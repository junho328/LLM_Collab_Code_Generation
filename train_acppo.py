"""
Training script for ACPPO (Agent-Chained Policy Optimization).

ACPPO combines:
- Simultaneous model updates (like MAPPO/MAGRPO)
- Per-agent value networks (decentralized, not centralized like HAVPPO)
- Agent chaining from MAGRPO
- KL-based similarity reward for auxiliary function prediction
- TD-based refined advantage calculation

Key features:
1. Simultaneous agent updates (not sequential like HAVPPO)
2. Per-agent value heads (2-layer MLP on frozen backbone)
3. TD residuals for advantage calculation with gamma' and lambda'
4. KL similarity reward: r_aux_sim = exp(-alpha * KL_token)
5. Agent chaining: Agent 2 predicts Agent 1's aux before generating main

Supported datasets:
- HumanEval / CoopHumanEval / MBPP
- BigCodeBench (via local JSON/JSONL file)
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Tuple

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset, Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

from rewards.code_rewards import execution_reward_aux, execution_reward_bigcodebench
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.acppo import ACPPOConfig, ACPPOTrainer
import external as external_ctx
from external import get_external_transition


def extract_function_params_from_prompt(prompt_text):
    """Extract function parameters from the prompt text."""
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def load_json_dataset(
    json_path: str,
    train_split: str,
    eval_split: str,
    verbose: bool = False,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
    seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Load dataset from a JSON or JSONL file.
    
    Args:
        json_path: Path to the JSON/JSONL file
        train_split: Split specification like "train[:500]" or "0:500"
        eval_split: Split specification like "train[500:550]" or "500:550"
        verbose: Whether to print loading information
        shuffle_train: Whether to shuffle the training dataset
        shuffle_eval: Whether to shuffle the evaluation dataset
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load data from JSON/JSONL file
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        # Check if it's JSONL (one JSON per line) or regular JSON
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Regular JSON array
            data = json.load(f)
        else:
            # JSONL format
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    if verbose:
        print(f"Loaded {len(data)} samples from {json_path}")
    
    # Parse split specifications
    def parse_split(split_str: str, total_len: int) -> Tuple[int, int]:
        """Parse split string like 'train[:500]' or '0:500' into (start, end)."""
        # Remove 'train' prefix if present
        split_str = re.sub(r'^(train|test|validation)\s*', '', split_str)
        # Remove brackets
        split_str = split_str.strip('[]')
        
        if ':' in split_str:
            parts = split_str.split(':')
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else total_len
        else:
            # Single index
            start = 0
            end = int(split_str) if split_str else total_len
        
        # Handle negative indices
        if start < 0:
            start = total_len + start
        if end < 0:
            end = total_len + end
            
        return max(0, start), min(total_len, end)
    
    total_len = len(data)
    
    train_start, train_end = parse_split(train_split, total_len)
    eval_start, eval_end = parse_split(eval_split, total_len)
    
    train_data = data[train_start:train_end]
    eval_data = data[eval_start:eval_end]
    
    if verbose:
        print(f"Train split [{train_start}:{train_end}]: {len(train_data)} samples")
        print(f"Eval split [{eval_start}:{eval_end}]: {len(eval_data)} samples")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    # Shuffle datasets if requested
    if shuffle_train:
        train_dataset = train_dataset.shuffle(seed=seed)
        if verbose:
            print(f"Train dataset shuffled with seed={seed}")
    
    if shuffle_eval:
        eval_dataset = eval_dataset.shuffle(seed=seed)
        if verbose:
            print(f"Eval dataset shuffled with seed={seed}")
    
    return train_dataset, eval_dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aux_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the auxiliary function generator (Agent 1) for code tasks."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    prompt_text = f"""Create a helper function for this coding problem.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Create a helper function named 'aux' that can assist the main function
- The function should return useful data for solving the problem

Your output should follow this format:

def aux(...):\n # your function code here\nreturn result\n"""

    return prompt_text


def main_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the main function generator (Agent 2) for code tasks."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    prompt_text = f"""Implement the '{entry_point}' function to solve this coding problem.

Problem:
{prompt}

You have access to a helper function aux() that you can call.

Requirements:
- Implement the function '{entry_point}' with parameters: {params_str}
- You may call aux() within your function if helpful
- Do NOT redefine aux()
- Include a return statement with actual computed values
- Do NOT include docstrings (no triple quotes)
- Output ONLY the Python function code, nothing else

Your output should follow this format:

def {entry_point}({params_str}):\n# your function code with aux fucntion call here\nreturn result"""

    return prompt_text


def chaining_main_function_formatter(example: Dict[str, Any]) -> str:
    """
    Formatter for Agent 2 in agent chaining mode.
    
    Agent 2 first predicts what Agent 1 will generate (auxiliary function),
    then generates the main function. Uses natural Python comment format
    instead of XML tags for better model understanding.
    """
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    prompt_text = f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You will use a helper function aux() that you need to predict first.

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the functions
- Do NOT include test cases or example usage
- First, write the aux() helper function you predict will be useful
- Then, write the '{entry_point}' function that calls aux()

Your output should follow this format:

# Predicted helper function
def aux(...):
    # helper function code here
    return result

# Main function
def {entry_point}({params_str}):
    # your function code with aux() call here
    return result
"""

    return prompt_text


# ============================================================================
# BigCodeBench Formatters
# ============================================================================

def _extract_imports_for_prompt(code_prompt: str) -> str:
    """Extract import statements from code_prompt for display in prompt."""
    if not code_prompt:
        return ""
    
    import_lines = []
    for line in code_prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(stripped)
        elif stripped.startswith("def "):
            break  # Stop at function definition
    
    return "\n".join(import_lines)


def bigcodebench_aux_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the auxiliary function generator (Agent 1) for BigCodeBench tasks."""
    code_prompt = example.get("code_prompt", "")
    complete_prompt = example.get("complete_prompt", "")
    entry_point = example.get("entry_point", "")

    # Extract imports to show available libraries
    available_imports = _extract_imports_for_prompt(code_prompt)
    imports_section = ""
    if available_imports:
        imports_section = f"""Available libraries (already imported):
{available_imports}
"""

    prompt_text = f"""Create a helper function for this coding problem.

Problem: 
{complete_prompt}

{imports_section}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Do NOT import additional libraries
- Create a helper function named 'aux' that can assist the main function
- The function should return useful data for solving the problem

Your output should follow this format:

def aux(...):\n # your aux function code here\nreturn result\n"""

    return prompt_text


def create_bigcodebench_main_formatter(force_aux_usage: bool = False):
    """Factory function to create a main formatter with configurable aux usage.
    
    Args:
        force_aux_usage: If True, prompt requires aux() usage. If False, aux() is optional.
    """
    def bigcodebench_main_formatter(example: Dict[str, Any]) -> str:
        """Formatter for the main function generator (Agent 2) for BigCodeBench tasks."""
        complete_prompt = example.get("complete_prompt", "")
        code_prompt = example.get("code_prompt", "")
        entry_point = example.get("entry_point", "")

        # Extract function signature from code_prompt
        func_signature = ""
        for line in code_prompt.split("\n"):
            if line.strip().startswith(f"def {entry_point}"):
                func_signature = line.strip()

        # Extract imports to show available libraries
        available_imports = _extract_imports_for_prompt(code_prompt)
        imports_section = ""
        if available_imports:
            imports_section = f"""Available libraries (already imported):
{available_imports}
"""

        # Choose aux usage instruction based on config
        if force_aux_usage:
            aux_note = "You MUST infer and call the helper function aux() to assign value to a variable within your function."
        else:
            aux_note = "You can infer and call the helper function aux() if helpful."

        prompt_text = f"""Implement the '{entry_point}' function for the following problem.

Problem: 
{complete_prompt}

{imports_section}

You have access to a helper function aux() that you can call.

Requirements:
- Implement the function 'task_func' with parameters: {params_str}
- You may call aux() within your function if helpful
- Do NOT redefine aux()
- Include a return statement with actual computed values
- Do NOT include docstrings (no triple quotes)
- Do NOT import additional libraries
- Output ONLY the Python function code, nothing else

Your output should follow this format:

{func_signature}\n # your function code with aux fucntion call here\nreturn result"""

        return prompt_text
    
    return bigcodebench_main_formatter


def chaining_bigcodebench_main_formatter(example: Dict[str, Any]) -> str:
    """
    Formatter for Agent 2 in agent chaining mode for BigCodeBench tasks.
    
    Agent 2 first predicts what Agent 1 will generate (auxiliary function),
    then generates the main function.
    """
    complete_prompt = example.get("complete_prompt", "")
    code_prompt = example.get("code_prompt", "")
    entry_point = example.get("entry_point", "")

    # Extract function signature from code_prompt
    func_signature = ""
    for line in code_prompt.split("\n"):
        if line.strip().startswith(f"def {entry_point}"):
            func_signature = line.strip()
            break

    # Extract imports to show available libraries
    available_imports = _extract_imports_for_prompt(code_prompt)
    imports_section = ""
    if available_imports:
        imports_section = f"""Available libraries (already imported):
{available_imports}
"""

    prompt_text = f"""Solve this coding problem by implementing the required function.

Problem: 
{complete_prompt}

{imports_section}

You will use a helper function aux() that you need to predict first.

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the functions
- Do NOT include test cases or example usage
- Do NOT import additional libraries
- First, write the aux() helper function you predict will be useful
- Then, write the 'task_func' function that calls aux()

Your output should follow this format:

# Predicted helper function
def aux(...):
    # helper function code here
    return result

# Main function
{func_signature}
    # your function code with aux() call here
    return result
"""

    return prompt_text


def get_formatters(dataset_type: str, num_agents: int, agent_chaining: bool = False, force_aux_usage: bool = False):
    """Get a list of per-agent formatters based on dataset type and agent count.
    
    Args:
        dataset_type: Type of dataset (humaneval, bigcodebench, etc.)
        num_agents: Number of agents
        agent_chaining: If True, use agent chaining formatters
        force_aux_usage: If True, main agent prompt requires aux() usage (BigCodeBench only)
    """
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"] and num_agents == 2:
        if agent_chaining:
            return [aux_function_formatter, chaining_main_function_formatter]
        return [aux_function_formatter, main_function_formatter]

    if dataset_type.lower() in ["bigcodebench", "bcb"] and num_agents == 2:
        if agent_chaining:
            return [bigcodebench_aux_formatter, chaining_bigcodebench_main_formatter]
        main_formatter = create_bigcodebench_main_formatter(force_aux_usage=force_aux_usage)
        return [bigcodebench_aux_formatter, main_formatter]

    raise NotImplementedError(f"Dataset type '{dataset_type}' with {num_agents} agents has not been implemented yet")


def get_logger_and_aggregator(dataset_type: str, is_multi_turn: bool = False):
    """
    Get the logger and aggregator functions based on dataset type.
    Returns a logger wrapper that passes dataset_type to enable proper test execution.
    """
    if dataset_type is None:
        return None, None

    # Use unified logger/aggregator for code datasets
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp", "bigcodebench", "bcb"]:
        # Create a wrapper that passes dataset_type to the logger
        def logger_wrapper(**kwargs):
            return mt_humaneval_logger(dataset_type=dataset_type, **kwargs)
        
        return logger_wrapper, aggregate_mt_humaneval_metrics_for_logging

    return None, None


def get_reward_function(dataset_type: str, num_agents: int):
    """Get a reward function compatible with variable number of agents.
    
    For code tasks, map N-agent completions to the existing aux/main reward by
    using the first agent as aux and the last agent as main.
    """
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval/bigcodebench' to the dataset section."
        )

    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:

        def reward_wrapper(*agent_completions, batch_items=None, prompts=None):
            if not agent_completions or len(agent_completions) < 1:
                return []

            if len(agent_completions) >= 2:
                completion1 = agent_completions[0]
                completion2 = agent_completions[-1]
            else:
                completion1 = [""] * len(agent_completions[0])
                completion2 = agent_completions[0]

            test_cases = []
            entry_points = []
            original_prompts = []

            if batch_items is not None:
                for item in batch_items:
                    test_cases.append(item["test"])
                    entry_points.append(item["entry_point"])
                    original_prompts.append(item.get("prompt", ""))
            else:
                raise ValueError("batch_items must be provided for reward calculation")

            return execution_reward_aux(
                completion1, completion2, test_cases, entry_points, original_prompts
            )

        return reward_wrapper

    if dataset_type.lower() in ["bigcodebench", "bcb"]:

        def bigcodebench_reward_wrapper(*agent_completions, batch_items=None, prompts=None):
            """Reward wrapper for BigCodeBench dataset."""
            if not agent_completions or len(agent_completions) < 1:
                return []

            # Choose aux from first agent, main from last agent
            if len(agent_completions) >= 2:
                completion1 = agent_completions[0]
                completion2 = agent_completions[-1]
            else:
                completion1 = [""] * len(agent_completions[0])
                completion2 = agent_completions[0]

            test_cases = []
            entry_points = []
            code_prompts = []
            complete_prompts = []

            if batch_items is not None:
                for item in batch_items:
                    # BigCodeBench uses 'test' for unittest code
                    test_cases.append(item.get("test", ""))
                    entry_points.append(item.get("entry_point", ""))
                    # BigCodeBench has code_prompt with imports and function signature
                    code_prompts.append(item.get("code_prompt", ""))
                    # BigCodeBench has complete_prompt with task description
                    complete_prompts.append(item.get("complete_prompt", ""))
            else:
                raise ValueError("batch_items must be provided for BigCodeBench reward calculation")

            return execution_reward_bigcodebench(
                completion1, completion2, test_cases, entry_points, code_prompts, complete_prompts
            )

        return bigcodebench_reward_wrapper

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main function to run the ACPPO training."""
    parser = argparse.ArgumentParser(
        description="Train ACPPO with configurable dataset (single-turn or multi-turn)"
    )
    add_config_args(parser)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config: load YAML and apply overrides
    # ------------------------------------------------------------------
    if args.config:
        config = Config(args.config)
    else:
        raise ValueError("Please provide a configuration file using --config")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    # ------------------------------------------------------------------
    # Config: model, dataset, output
    # ------------------------------------------------------------------
    model_config = config.get_model_config()
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")
    output_base_dir = config.get("output.base_dir")

    # Try to infer dataset type from dataset name if not specified
    if dataset_type is None:
        if "humaneval" in dataset_name.lower() and "coop" not in dataset_name.lower():
            dataset_type = "humaneval"
        elif "coophumaneval" in dataset_name.lower() or "coop" in dataset_name.lower():
            dataset_type = "coophumaneval"
        elif "mbpp" in dataset_name.lower():
            dataset_type = "mbpp"
        elif "bigcodebench" in dataset_name.lower() or "bcb" in dataset_name.lower():
            dataset_type = "bigcodebench"
        else:
            raise ValueError(
                f"Could not infer dataset type from dataset name '{dataset_name}'. "
                "Please specify 'type' in dataset config."
            )
        print(f"Dataset type not specified, inferred as: {dataset_type}")

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    # ------------------------------------------------------------------
    # Config: ACPPO training params
    # ------------------------------------------------------------------
    acppo_config = (
        config.get_section("acppo") if hasattr(config, "get_section") else {}
    )

    seed_value = int(config.get("seed", acppo_config.get("seed", 42)))
    num_turns = acppo_config.get("num_turns", 1)
    num_agents = acppo_config.get("num_agents", 2)
    is_multi_turn = num_turns > 1
    output_verbose = config.get("output.verbose", False)
    agent_chaining = acppo_config.get("agent_chaining", True)

    if output_verbose:
        if is_multi_turn:
            print(f"Multi-turn training enabled: num_turns={num_turns}")
        else:
            print(f"Single-turn training: num_turns={num_turns}")
        print("Using ACPPO algorithm with simultaneous agent updates and TD-based advantage")
        print(f"Agent chaining: {agent_chaining}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")

    if is_multi_turn:
        output_dir = os.path.join(output_base_dir, f"mt_acppo_job_{slurm_job_id}")
    else:
        output_dir = os.path.join(output_base_dir, f"acppo_job_{slurm_job_id}")

    os.makedirs(output_dir, exist_ok=True)

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)

    _set_seed(seed_value)

    train_dataset = None
    eval_dataset = None
    
    # Get shuffle options from config
    shuffle_train = config.get("dataset.shuffle_train", False)
    shuffle_eval = config.get("dataset.shuffle_eval", False)
    
    # Check if dataset_name is a local JSON/JSONL file
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        try:
            train_dataset, eval_dataset = load_json_dataset(
                dataset_name, train_split, eval_split, output_verbose,
                shuffle_train=shuffle_train,
                shuffle_eval=shuffle_eval,
                seed=seed_value,
            )
        except Exception as e:
            print(f"Error loading JSON dataset: {e}")
            return
    else:
        try:
            train_dataset = load_dataset(dataset_name, split=train_split)
            eval_dataset = load_dataset(dataset_name, split=eval_split)
            
            # Shuffle HuggingFace datasets if requested
            if shuffle_train:
                train_dataset = train_dataset.shuffle(seed=seed_value)
                if output_verbose:
                    print(f"Train dataset shuffled with seed={seed_value}")
            if shuffle_eval:
                eval_dataset = eval_dataset.shuffle(seed=seed_value)
                if output_verbose:
                    print(f"Eval dataset shuffled with seed={seed_value}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

    if output_verbose:
        print(f"\nUsing model: {model_name}")
        print(f"Model type: {model_config.type}")
        print(f"Max context window: {model_config.max_length} tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **model_config.tokenizer_kwargs
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding_side = config.get("tokenizer.padding_side")
    if padding_side:
        tokenizer.padding_side = padding_side

    if model_config.special_tokens:
        if output_verbose:
            print("Adding special tokens...")
        tokenizer.add_special_tokens(model_config.special_tokens)
        if output_verbose:
            print(
                f"Special tokens added: {model_config.special_tokens.get('additional_special_tokens', [])}"
            )

    temperature = acppo_config.get("temperature", 0.6)
    top_p = acppo_config.get("top_p", 0.6)

    # ------------------------------------------------------------------
    # Config: External transitions
    # ------------------------------------------------------------------
    external_cfg = config.get_section("external") if hasattr(config, "get_section") else {}

    def _normalize_prompt(p: str) -> str:
        return " ".join((p or "").split()).strip()

    context_map = {}

    _sandbox_val = external_cfg.get("sandbox_slice", 1)
    if isinstance(_sandbox_val, str):
        _sv = _sandbox_val.strip().lower()
        if _sv == "all":
            sandbox_slice = 0
        elif _sv.lstrip("-").isdigit():
            sandbox_slice = int(_sv)
        else:
            sandbox_slice = None
    elif isinstance(_sandbox_val, int):
        sandbox_slice = _sandbox_val
    else:
        sandbox_slice = None if _sandbox_val is None else 0

    def _make_sliced_assert_tests(test_code: str, n: int) -> str:
        if not isinstance(test_code, str) or not test_code.strip():
            return test_code

        if n is None or n == 0:
            return test_code

        lines = test_code.splitlines()
        preamble = []
        check_idx = None
        for idx, line in enumerate(lines):
            if re.match(r"\s*def\s+check\s*\(candidate\)\s*:\s*", line):
                check_idx = idx
                break
            preamble.append(line)

        asserts = []
        search_start = check_idx + 1 if check_idx is not None else 0
        for line in lines[search_start:]:
            s = line.strip()
            if s.startswith("assert") and "candidate" in s:
                asserts.append(s)

        if not asserts:
            return test_code

        preamble_text = "\n".join(preamble).strip()
        new_parts = []
        if preamble_text:
            new_parts.append(preamble_text)
        new_parts.append("def check(candidate):")
        selected = asserts[:n] if n > 0 else asserts[n:]
        for a in selected:
            new_parts.append(f"    {a}")
        return "\n".join(new_parts) + "\n"

    def _register_split(ds):
        try:
            for item in ds:
                key = _normalize_prompt(item.get("prompt", ""))
                if key and key not in context_map:
                    tests_eval = item.get("test", "")
                    tests_sandbox = (
                        _make_sliced_assert_tests(tests_eval, sandbox_slice)
                        if sandbox_slice is not None and sandbox_slice != 0
                        else tests_eval
                    )
                    context_map[key] = {
                        "entry_point": item.get("entry_point", ""),
                        "tests_eval": tests_eval,
                        "tests_sandbox": tests_sandbox,
                    }
        except Exception:
            pass

    if "train_dataset" in locals() and train_dataset is not None:
        _register_split(train_dataset)
    if "eval_dataset" in locals() and eval_dataset is not None:
        _register_split(eval_dataset)

    def _resolver(prompt: str):
        return context_map.get(_normalize_prompt(prompt))

    external_ctx.set_context_resolver(_resolver)

    # ------------------------------------------------------------------
    # Build training args with ACPPO-specific parameters
    # ------------------------------------------------------------------
    acppo_args_kwargs = {
        "output_dir": output_dir,
        "num_agents": num_agents,
        "num_train_epochs": acppo_config.get("num_train_epochs", 20),
        "per_device_train_batch_size": acppo_config.get(
            "per_device_train_batch_size", 1
        ),
        "learning_rate": acppo_config.get("learning_rate", 5e-6),
        "logging_steps": acppo_config.get("logging_steps", 50),
        "save_steps": acppo_config.get("save_steps", 200),
        "eval_interval": acppo_config.get("eval_interval", 16),
        "eval_num_samples": acppo_config.get("eval_num_samples", 4),
        "num_generations": acppo_config.get("num_generations", 4),
        "max_new_tokens": acppo_config.get("max_new_tokens", 256),
        "temperature": temperature,
        "top_p": top_p,
        # Multi-turn parameters
        "num_turns": num_turns,
        "discount": acppo_config.get("discount", 0.9),
        "joint_mode": acppo_config.get("joint_mode", "aligned"),
        "termination_threshold": acppo_config.get("termination_threshold", -0.2),
        "rollout_buffer_size": acppo_config.get("rollout_buffer_size", 2),
        "external_prompt_passthrough": True,
        # ACPPO-specific: per-agent value network parameters
        "value_head_hidden_dim": acppo_config.get("value_head_hidden_dim", 256),
        "value_learning_rate": acppo_config.get("value_learning_rate", 1e-4),
        "value_loss_coef": acppo_config.get("value_loss_coef", 0.5),
        # ACPPO-specific: TD-based advantage parameters
        "gamma_prime": acppo_config.get("gamma_prime", 0.99),
        "lambda_prime": acppo_config.get("lambda_prime", 0.95),
        "advantage_normalization": acppo_config.get("advantage_normalization", True),
        # Agent chaining parameters
        "agent_chaining": agent_chaining,
        # PPO Clip parameters
        "use_ppo_clip": acppo_config.get("use_ppo_clip", False),
        "ppo_clip_eps": acppo_config.get("ppo_clip_eps", 0.2),
    }

    # Handle max_new_tokens_per_agent
    max_tokens_per_agent = acppo_config.get("max_new_tokens_per_agent")
    if max_tokens_per_agent is not None:
        acppo_args_kwargs["max_new_tokens_per_agent"] = max_tokens_per_agent

    if "top_k" in acppo_config:
        acppo_args_kwargs["top_k"] = acppo_config.get("top_k")

    acppo_args = ACPPOConfig(**acppo_args_kwargs)

    # ------------------------------------------------------------------
    # Formatters, rewards, and logging
    # ------------------------------------------------------------------
    force_aux_usage = acppo_config.get("force_aux_usage", False)
    formatters = get_formatters(dataset_type, num_agents, agent_chaining, force_aux_usage)
    reward_func = get_reward_function(dataset_type, num_agents)
    eval_logger, eval_aggregator = get_logger_and_aggregator(
        dataset_type, is_multi_turn
    )

    # ------------------------------------------------------------------
    # W&B configuration
    # ------------------------------------------------------------------
    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )

    if is_multi_turn:
        wandb_name = wandb_section.get("name", f"mt_acppo_{dataset_type}")
    else:
        wandb_name = wandb_section.get("name", f"acppo_{dataset_type}")

    external_mode = external_cfg.get("mode", "level_feedback")
    default_tags = ["acppo", dataset_type or "code", f"turns_{num_turns}"]
    if agent_chaining:
        default_tags.append("agent-chaining")
    tags_from_cfg = wandb_section.get("tags", default_tags)
    tags = list(tags_from_cfg) if isinstance(tags_from_cfg, list) else default_tags

    if external_mode == "level_feedback":
        if "self-evolved" not in tags:
            tags.append("self-evolved")

    dataset_section = config.get_section("dataset") if hasattr(config, "get_section") else {}
    model_section = config.get_section("model") if hasattr(config, "get_section") else {}
    output_section = config.get_section("output") if hasattr(config, "get_section") else {}

    wandb_config = {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": f"{wandb_name}",
        "dir": wandb_section.get("dir", "../../../projects/bepg/sliu30"),
        "tags": tags,
        "config_sections": {
            "dataset": dataset_section,
            "model": model_section,
            "output": output_section,
            "external": external_cfg,
            "trainer": acppo_config,
        },
    }

    # Propagate verbosity to reward/external modules
    try:
        import rewards.code_rewards as code_rewards
        code_rewards.VERBOSE = bool(output_verbose)
    except Exception:
        pass
    try:
        import external as external_mod
        external_mod.VERBOSE = bool(output_verbose)
    except Exception:
        pass

    model_load_kwargs = dict(model_config.model_kwargs)
    if "attn_implementation" not in model_load_kwargs:
        model_load_kwargs["attn_implementation"] = "flash_attention_2"

    # ------------------------------------------------------------------
    # LoRA PEFT configuration (for policy models)
    # ------------------------------------------------------------------
    lora_config_section = (
        config.get_section("lora") if hasattr(config, "get_section") else {}
    )
    use_lora = lora_config_section.get("enabled", True)  # Default to True for ACPPO

    if use_lora:
        # LoRA configuration
        lora_r = lora_config_section.get("r", 32)
        lora_alpha = lora_config_section.get("lora_alpha", 32)
        lora_dropout = lora_config_section.get("lora_dropout", 0.0)
        lora_target_modules = lora_config_section.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        lora_bias = lora_config_section.get("bias", "none")

        if output_verbose:
            print("\n" + "=" * 60)
            print("LoRA Configuration (Policy Models)")
            print("=" * 60)
            print(f"  Enabled: {use_lora}")
            print(f"  Rank (r): {lora_r}")
            print(f"  Alpha: {lora_alpha}")
            print(f"  Dropout: {lora_dropout}")
            print(f"  Target Modules: {lora_target_modules}")
            print(f"  Bias: {lora_bias}")
            print("=" * 60 + "\n")

        # Create separate model instances with individual LoRA adapters for each agent
        agents = []
        for agent_idx in range(num_agents):
            # Load fresh base model for each agent
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_load_kwargs,
            )

            # Create LoRA config for this agent
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias=lora_bias,
            )

            # Wrap with PEFT - creates agent-specific adapter
            peft_model = get_peft_model(base_model, peft_config)

            if output_verbose:
                trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in peft_model.parameters())
                print(f"Agent {agent_idx}: Trainable params: {trainable_params:,} / {total_params:,} "
                      f"({100 * trainable_params / total_params:.2f}%)")

            agents.append(peft_model)
    else:
        # Full model training (not recommended due to memory)
        agents = [
            AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_load_kwargs,
            )
            for _ in range(num_agents)
        ]

    # ------------------------------------------------------------------
    # Print ACPPO-specific configuration
    # ------------------------------------------------------------------
    if output_verbose:
        print("\n" + "=" * 60)
        print("ACPPO Value Network Configuration (Per-Agent)")
        print("=" * 60)
        print(f"  Hidden Dim: {acppo_args.value_head_hidden_dim}")
        print(f"  Learning Rate: {acppo_args.value_learning_rate}")
        print(f"  Loss Coefficient: {acppo_args.value_loss_coef}")
        print("  Base Model: Frozen (only value head trainable)")
        print("=" * 60 + "\n")

        print("\n" + "=" * 60)
        print("ACPPO TD-based Advantage Configuration")
        print("=" * 60)
        print(f"  Gamma' (gamma_prime): {acppo_args.gamma_prime}")
        print(f"  Lambda' (lambda_prime): {acppo_args.lambda_prime}")
        print(f"  Advantage Normalization: {acppo_args.advantage_normalization}")
        print("=" * 60 + "\n")

        if agent_chaining:
            print("\n" + "=" * 60)
            print("ACPPO Agent Chaining Configuration")
            print("=" * 60)
            print(f"  Agent Chaining: {agent_chaining}")
            print("=" * 60 + "\n")

    reward_processor = None
    if config.get("reward_processor.enabled", True):
        scale_factor = config.get("reward_processor.scale_factor", 1.0)
        reward_processor = RewardProcessors.scale(factor=scale_factor)
        shift_val = config.get("reward_processor.shift", None)
        if shift_val is not None:
            try:
                shift_val_f = float(shift_val)
            except (TypeError, ValueError):
                shift_val_f = None
            if shift_val_f is not None:
                shift_proc = RewardProcessors.shift(value=shift_val_f)
                prev = reward_processor
                reward_processor = (lambda p=prev, s=shift_proc: (lambda x: s(p(x))))()

    # ------------------------------------------------------------------
    # Build trainer kwargs
    # ------------------------------------------------------------------
    trainer_kwargs = {
        # Model / data
        "agents": agents,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        # Reward / formatting
        "reward_func": reward_func,
        "formatters": formatters,
        # Logging / eval / config
        "wandb_config": wandb_config,
        "eval_logger": eval_logger,
        "eval_aggregator": eval_aggregator,
        # Training args
        "args": acppo_args,
        # LoRA configuration
        "use_lora": use_lora,
        # Model config
        "model_config": {
            "model_kwargs": model_load_kwargs,
            "tokenizer_kwargs": model_config.tokenizer_kwargs,
        },
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    # External transition for multi-turn
    if (
        is_multi_turn
        and dataset_type
        and dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]
    ):
        expert_model = external_cfg.get("expert_model", "deepseek-coder")

        def external_transition_wrapper(
            prompt,
            agent_completions,
            num_agents,
            prompt_history_per_agent=None,
            response_history_per_agent=None,
        ):
            return get_external_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=num_agents,
                expert_model=expert_model,
                mode=external_mode,
                prompt_history_per_agent=prompt_history_per_agent,
                response_history_per_agent=response_history_per_agent,
            )

        trainer_kwargs["external_transition"] = external_transition_wrapper

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    if output_verbose:
        print("\n" + "=" * 60)
        print("ACPPO Training Configuration")
        print("=" * 60)
        print(f"Algorithm: ACPPO (Simultaneous Agent Updates + TD-based Advantage)")
        print(f"Number of Agents: {num_agents}")
        print(f"Number of Turns: {num_turns}")
        print(f"Agent Chaining: {agent_chaining}")
        print(f"TD Advantage: gamma'={acppo_args.gamma_prime}, lambda'={acppo_args.lambda_prime}")
        print("=" * 60 + "\n")

    trainer = ACPPOTrainer(**trainer_kwargs)
    trainer.train()

    save_final = config.get("output.save_final_model", False)
    if save_final:
        save_path = config.get(
            "output.save_path", os.path.join(output_dir, "final_model")
        )
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
