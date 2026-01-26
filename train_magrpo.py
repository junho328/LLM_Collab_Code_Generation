"""
Unified training script for MAGRPO that supports both single-turn and multi-turn training.
The training mode is determined by the num_turns parameter in the config file.
Supports multiple datasets and configurations via YAML files.
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from typing import Any, Dict, Tuple, Optional

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Single-turn code logger no longer used directly; multi-turn logger handles all cases
from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

from rewards.code_rewards import execution_reward_aux, execution_reward_bigcodebench
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_dataset(
    json_path: str,
    train_split: str,
    eval_split: str,
    verbose: bool = False
) -> Tuple[Any, Any]:
    """
    Load dataset from a JSON or JSONL file.
    
    Args:
        json_path: Path to the JSON/JSONL file
        train_split: Split specification like "train[:500]" or "0:500"
        eval_split: Split specification like "train[500:550]" or "500:550"
        verbose: Whether to print loading information
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from datasets import Dataset
    
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
    
    return train_dataset, eval_dataset


def aux_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the auxiliary function generator (Agent 1) for code tasks."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

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
- Define actual parameters for the aux function (not "...")

Example format (replace with actual implementation):

def aux(param1, param2):
    # implementation here
    return result
"""

    return prompt_text


def main_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the main function generator (Agent 2) for code tasks."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    prompt_text = f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You have access to a helper function: aux()

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Do NOT redefine the aux() function
- Implement ONLY the '{entry_point}' function as specified
- You can call aux() to assign value to a variable within your function if helpful

Example format (replace with actual implementation):

def {entry_point}({params_str}):
    # implementation here
    return result
"""

    return prompt_text


# ============================================================================
# BigCodeBench Formatters
# ============================================================================

def bigcodebench_aux_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the auxiliary function generator (Agent 1) for BigCodeBench tasks."""
    # BigCodeBench uses instruct_prompt for task description
    instruct_prompt = example.get("instruct_prompt", "")
    code_prompt = example.get("code_prompt", "")
    entry_point = example.get("entry_point", "")

    if not instruct_prompt:
        return "Error: Could not extract task information from BigCodeBench data."

    prompt_text = f"""Create a helper function for this coding problem.

Problem:
{instruct_prompt}

Code template:
{code_prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Create a helper function named 'aux' that can assist the main function '{entry_point}'
- The function should return useful data for solving the problem
- Do NOT include import statements (they will be added separately)
- Define actual parameters for the aux function (not "...")

Example format (replace with actual implementation):

def aux(param1, param2):
    # implementation here
    return result
"""

    return prompt_text


def create_bigcodebench_main_formatter(force_aux_usage: bool = False):
    """Factory function to create a main formatter with configurable aux usage.
    
    Args:
        force_aux_usage: If True, prompt requires aux() usage. If False, aux() is optional.
    """
    def bigcodebench_main_formatter(example: Dict[str, Any]) -> str:
        """Formatter for the main function generator (Agent 2) for BigCodeBench tasks."""
        instruct_prompt = example.get("instruct_prompt", "")
        code_prompt = example.get("code_prompt", "")
        entry_point = example.get("entry_point", "")

        if not instruct_prompt or not entry_point:
            return "Error: Could not extract task information from BigCodeBench data."

        # Extract function signature from code_prompt
        func_signature = ""
        if code_prompt:
            for line in code_prompt.split("\n"):
                if line.strip().startswith(f"def {entry_point}"):
                    func_signature = line.strip()
                    break

        # Choose aux usage instruction based on config
        if force_aux_usage:
            aux_instruction = "- You MUST call aux() in your implementation (required, not optional)"
        else:
            aux_instruction = "- You can call aux() to help with the implementation"

        prompt_text = f"""Solve this coding problem by implementing the required function.

Problem:
{instruct_prompt}

Code template:
{code_prompt}

You have access to a helper function: aux()

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Do NOT redefine the aux() function
- Do NOT include import statements (they will be added separately)
- Implement ONLY the '{entry_point}' function as specified
{aux_instruction}

Example format (replace with actual implementation):

{func_signature if func_signature else f"def {entry_point}(param1, param2):"}
    # implementation here
    return result
"""

        return prompt_text
    
    return bigcodebench_main_formatter


# Keep backward-compatible default formatter
bigcodebench_main_formatter = create_bigcodebench_main_formatter(force_aux_usage=False)


def get_formatters(dataset_type: str, num_agents: int, force_aux_usage: bool = False):
    """Get a list of per-agent formatters based on dataset type and agent count.

    For code tasks, use aux formatters for all agents except the last, which uses main.
    
    Args:
        dataset_type: Type of dataset (humaneval, bigcodebench, etc.)
        num_agents: Number of agents
        force_aux_usage: If True, main agent prompt requires aux() usage (BigCodeBench only)
    """
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"] and num_agents == 2:
        return [aux_function_formatter, main_function_formatter]
    
    if dataset_type.lower() in ["bigcodebench", "bcb"] and num_agents == 2:
        main_formatter = create_bigcodebench_main_formatter(force_aux_usage=force_aux_usage)
        return [bigcodebench_aux_formatter, main_formatter]

    raise NotImplementedError(f"Dataset type '{dataset_type}' with {num_agents} agents has not been implemented yet")


def get_logger_and_aggregator(dataset_type: str):
    """
    Get the logger and aggregator functions based on dataset type.
    """
    if dataset_type is None:
        return None, None

    # Use unified logger/aggregator for code datasets
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp", "bigcodebench", "bcb"]:
        return mt_humaneval_logger, aggregate_mt_humaneval_metrics_for_logging

    return None, None


def get_reward_function(dataset_type: str, num_agents: int):
    """Get a reward function compatible with variable number of agents (single-turn).

    For code tasks, map N-agent completions to the existing aux/main reward by
    using the first agent as aux and the last agent as main.
    """
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval/bigcodebench' to the dataset section."
        )

    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:

        def reward_wrapper(*agent_completions, batch_items=None, prompts=None):
            # agent_completions: tuple of lists (one list per agent), each list contains strings per completion
            if not agent_completions or len(agent_completions) < 1:
                return []

            # Choose aux from first agent if available when >=2, otherwise empty list
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

            if batch_items is not None:
                for item in batch_items:
                    # BigCodeBench uses 'test' for unittest code
                    test_cases.append(item.get("test", ""))
                    entry_points.append(item.get("entry_point", ""))
                    # BigCodeBench has code_prompt with imports and function signature
                    code_prompts.append(item.get("code_prompt", ""))
            else:
                raise ValueError("batch_items must be provided for BigCodeBench reward calculation")

            return execution_reward_bigcodebench(
                completion1, completion2, test_cases, entry_points, code_prompts
            )

        return bigcodebench_reward_wrapper

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main function to run the unified MAGRPO training."""
    parser = argparse.ArgumentParser(
        description="Train MAGRPO with configurable dataset (single-turn or multi-turn)"
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

    # Apply command-line overrides
    

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
                f"Could not infer dataset type from dataset name '{dataset_name}'. Please specify 'type' in dataset config."
            )
        print(f"Dataset type not specified, inferred as: {dataset_type}")

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    # ------------------------------------------------------------------
    # Config: MAGRPO training params and verbosity
    # ------------------------------------------------------------------
    magrpo_config = (
        config.get_section("magrpo") if hasattr(config, "get_section") else {}
    )
    seed_value = int(config.get("seed", magrpo_config.get("seed", 42)))
    num_agents = magrpo_config.get("num_agents", 2)
    output_verbose = config.get("output.verbose", False)
    if output_verbose:
        print(f"Single-turn MAGRPO training with {num_agents} agents")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"job_{slurm_job_id}")

    os.makedirs(output_dir, exist_ok=True)

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)

    _set_seed(seed_value)

    train_dataset = None
    eval_dataset = None
    
    # Check if dataset_name is a local JSON/JSONL file
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        try:
            train_dataset, eval_dataset = load_json_dataset(
                dataset_name, train_split, eval_split, output_verbose
            )
        except Exception as e:
            print(f"Error loading JSON dataset: {e}")
            return
    else:
        try:
            train_dataset = load_dataset(dataset_name, split=train_split, trust_remote_code=True)
            eval_dataset = load_dataset(dataset_name, split=eval_split, trust_remote_code=True)
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

    # Add special tokens if needed (e.g., FIM tokens for StarCoder)
    if model_config.special_tokens:
        if output_verbose:
            print("Adding special tokens...")
        tokenizer.add_special_tokens(model_config.special_tokens)
        if output_verbose:
            print(
                f"Special tokens added: {model_config.special_tokens.get('additional_special_tokens', [])}"
            )

    temperature = magrpo_config.get("temperature", 0.6)
    top_p = magrpo_config.get("top_p", 0.6)

    # ------------------------------------------------------------------
    # Config: External transitions (mode, sandbox, expert model, context flags)
    # ------------------------------------------------------------------
    external_cfg = config.get_section("external") if hasattr(config, "get_section") else {}
    
    # Multi-turn mode: currently only single-turn is supported
    is_multi_turn = False
    external_mode = external_cfg.get("mode", "level_feedback")

    # Register external context resolver using dataset items
    def _normalize_prompt(p: str) -> str:
        return " ".join((p or "").split()).strip()

    context_map = {}

    # Optionally restrict sandbox tests to the first N eval asserts
    # Default: keep only the first assert (sandbox_slice=1)
    # Set external.sandbox_slice to an integer N (>0) to keep the first N asserts,
    # or to 0 / None / 'all' to keep all eval asserts.
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

    # re already imported at module level

    def _make_sliced_assert_tests(test_code: str, n: int) -> str:
        if not isinstance(test_code, str) or not test_code.strip():
            return test_code

        # n > 0: keep first n asserts; n < 0: keep last |n| asserts; n == 0: keep all
        if n is None or n == 0:
            return test_code

        lines = test_code.splitlines()
        # Collect import preamble before check definition
        preamble = []
        check_idx = None
        for idx, line in enumerate(lines):
            if re.match(r"\s*def\s+check\s*\(candidate\)\s*:\s*", line):
                check_idx = idx
                break
            preamble.append(line)

        # Find assert statements containing 'candidate'
        asserts = []
        search_start = check_idx + 1 if check_idx is not None else 0
        for line in lines[search_start:]:
            s = line.strip()
            if s.startswith("assert") and "candidate" in s:
                asserts.append(s)

        if not asserts:
            return test_code  # fallback when no asserts found

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

    # Use unified MAGRPOConfig which handles both single-turn and multi-turn
    # ------------------------------------------------------------------
    # Build training args
    # ------------------------------------------------------------------
    # LoRA settings can be at root level or in magrpo section
    # Check root level first, then fallback to magrpo section
    use_lora = config.get("use_lora", magrpo_config.get("use_lora", False))
    lora_r = config.get("lora_r", magrpo_config.get("lora_r", 16))
    lora_alpha = config.get("lora_alpha", magrpo_config.get("lora_alpha", 32))
    lora_dropout = config.get("lora_dropout", magrpo_config.get("lora_dropout", 0.05))
    lora_target_modules = config.get("lora_target_modules", magrpo_config.get("lora_target_modules", None))
    lora_path = config.get("lora_path", magrpo_config.get("lora_path", None))

    magrpo_args_kwargs = {
        "output_dir": output_dir,
        "num_agents": num_agents,
        "num_train_epochs": magrpo_config.get("num_train_epochs", 20),
        "per_device_train_batch_size": magrpo_config.get(
            "per_device_train_batch_size", 1
        ),
        "learning_rate": magrpo_config.get("learning_rate", 5e-6),
        "logging_steps": magrpo_config.get("logging_steps", 50),
        "save_steps": magrpo_config.get("save_steps", 200),
        "eval_interval": magrpo_config.get("eval_interval", 16),
        "eval_num_samples": magrpo_config.get("eval_num_samples", 4),
        "num_generations": magrpo_config.get("num_generations", 4),
        "max_new_tokens": magrpo_config.get("max_new_tokens", 256),
        "temperature": temperature,
        "top_p": top_p,
        "joint_mode": magrpo_config.get("joint_mode", "aligned"),
        "rollout_buffer_size": magrpo_config.get("rollout_buffer_size", 2),
        # Parallel reward computation
        "parallel_reward": magrpo_config.get("parallel_reward", True),
        "max_reward_workers": magrpo_config.get("max_reward_workers", 8),
        "reward_parallel_backend": magrpo_config.get("reward_parallel_backend", "process"),
        # LoRA configuration (from root or magrpo section)
        "use_lora": use_lora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        # Multi-GPU
        "use_distributed": magrpo_config.get("use_distributed", False),
    }
    if "top_k" in magrpo_config:
        magrpo_args_kwargs["top_k"] = magrpo_config.get("top_k")
    if lora_target_modules is not None:
        magrpo_args_kwargs["lora_target_modules"] = lora_target_modules
    if lora_path is not None:
        magrpo_args_kwargs["lora_path"] = lora_path
    magrpo_args = MAGRPOConfig(**magrpo_args_kwargs)

    # ------------------------------------------------------------------
    # Formatters, rewards, and logging
    # ------------------------------------------------------------------
    force_aux_usage = magrpo_config.get("force_aux_usage", False)
    formatters = get_formatters(dataset_type, num_agents, force_aux_usage=force_aux_usage)
    reward_func = get_reward_function(dataset_type, num_agents)
    eval_logger, eval_aggregator = get_logger_and_aggregator(dataset_type)

    # ------------------------------------------------------------------
    # W&B configuration and tags
    # ------------------------------------------------------------------
    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )
    wandb_name = wandb_section.get("name", f"magrpo_{dataset_type}")

    default_tags = ["magrpo", dataset_type or "code", f"agents_{num_agents}"]
    tags_from_cfg = wandb_section.get("tags", default_tags)
    tags = list(tags_from_cfg) if isinstance(tags_from_cfg, list) else default_tags

    # Collect full config sections for W&B searchability
    dataset_section = config.get_section("dataset") if hasattr(config, "get_section") else {}
    model_section = config.get_section("model") if hasattr(config, "get_section") else {}
    output_section = config.get_section("output") if hasattr(config, "get_section") else {}

    wandb_config = {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": f"{wandb_name}",
        "dir": wandb_section.get("dir", "../../../projects/bepg/sliu30"),
        "tags": tags,
        # Provide full sections for the trainer to log cleanly
        "config_sections": {
            "dataset": dataset_section,
            "model": model_section,
            "output": output_section,
            "external": external_cfg,
            "trainer": magrpo_config,
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

    # Initialize completion logger if enabled in config
    completion_log_enabled = config.get("output.log_completions", False)
    if completion_log_enabled:
        try:
            from loggers.completion_logger import CompletionLogger
            max_samples = config.get("output.log_max_samples_per_file", 100)
            CompletionLogger.initialize(
                output_dir=output_dir,
                enabled=True,
                max_samples_per_file=max_samples,
            )
            if output_verbose:
                print(f"Completion logging enabled. Logs will be saved to: {output_dir}/completion_logs/")
        except Exception as e:
            print(f"Warning: Failed to initialize completion logger: {e}")

    # Use num_agents from magrpo config (where it belongs for MAGRPO training)
    # Create a single shared model for all agents (same model instance)
        # Enable Flash Attention 2 if not explicitly disabled in config
    model_load_kwargs = dict(model_config.model_kwargs)
    if "attn_implementation" not in model_load_kwargs:
        model_load_kwargs["attn_implementation"] = "flash_attention_2"
    
    shared_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_load_kwargs,
    )
    agents = [shared_model for _ in range(num_agents)]


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
    # Build trainer kwargs (grouped: model/data, reward/formatting, logging, args)
    # ------------------------------------------------------------------
    trainer_kwargs = {
        # Model / data
        "agents": agents,
        "num_agents": num_agents,
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
        "dataset_type": dataset_type,
        # Training args
        "args": magrpo_args,
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    if (
        is_multi_turn
        and dataset_type
        and dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]
    ):
        expert_model = external_cfg.get("expert_model", "deepseek-coder")
        # external_mode already loaded above

        def external_transition_wrapper(
            prompt,
            agent_completions,
            num_agents,
            prompt_history_per_agent=None,
            response_history_per_agent=None,
        ):
            # Returns full next-turn prompts per agent (strings)
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

    trainer = MAGRPOTrainer(**trainer_kwargs)
    trainer.train()
    
    # Flush completion logger at the end of training
    if completion_log_enabled:
        try:
            from loggers.completion_logger import CompletionLogger
            logger = CompletionLogger.get_instance()
            if logger:
                logger.flush()
                stats = logger.get_stats()
                if output_verbose:
                    print(f"Completion logging complete. Total samples logged: {stats['total_logged']}")
                    print(f"Log files saved to: {stats['log_dir']}")
        except Exception:
            pass
    
    save_final = config.get("output.save_final_model", False)
    if save_final:
        save_path = config.get(
            "output.save_path", os.path.join(output_dir, "final_model")
        )
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
