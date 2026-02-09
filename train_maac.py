import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from config import Config, add_config_args, parse_overrides
from comlrl.trainers.maac import MAACConfig, MAACTrainer
from comlrl.utils.reward_processor import RewardProcessors
from rewards.code_rewards import execution_reward_aux, execution_reward_bigcodebench
from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)
import external as external_ctx
from external import get_external_transition


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


def extract_function_params_from_prompt(prompt_text: str) -> List[str]:
    """Extract function parameters from the prompt text."""
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


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

#     prompt_text = f"""Solve this coding problem by implementing the required function.

# Problem:
# {prompt}

# You have access to a helper function: aux(...)

# IMPORTANT INSTRUCTIONS:
# - Output ONLY the function code, no explanations or examples
# - Do NOT include markdown code blocks (```python)
# - Do NOT include any text before or after the function
# - Do NOT include test cases or example usage
# - Do NOT redefine the aux() function
# - Implement ONLY the '{entry_point}' function as specified
# - You can call aux() to assign value to a variable within your function if helpful

# Your output should follow this format:

# def {entry_point}({params_str}):\n # your function code here\nreturn result\n"""

    prompt_text = f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You have access to a helper function: aux(...)

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Implement ONLY the '{entry_point}' function as specified
- You MUST call aux() to assign value to a variable within your function
- Do NOT redefine the aux() function

Your output should follow this format:

def {entry_point}({params_str}):\n # your function code with aux() call here\nreturn result\n"""

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
                break

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
- Implement the function '{entry_point}' with the specified signature
- {aux_note}
- Do NOT redefine aux()
- Include a return statement with actual computed values
- Do NOT include docstrings (no triple quotes)
- Do NOT import additional libraries
- Output ONLY the Python function code, nothing else

Your output should follow this format:

{func_signature}\n # your function code with aux function call here\nreturn result"""

        return prompt_text
    
    return bigcodebench_main_formatter


def build_prompt_formatters(dataset_type: str = None, force_aux_usage: bool = False) -> List:
    """Get a list of per-agent formatters based on dataset type.
    
    Args:
        dataset_type: Type of dataset (humaneval, coophumaneval, mbpp, bigcodebench)
        force_aux_usage: If True, require aux() usage in main function (BigCodeBench only)
    
    Returns:
        List of formatter functions [aux_formatter, main_formatter]
    """
    if dataset_type is None or dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:
        return [aux_function_formatter, main_function_formatter]
    
    if dataset_type.lower() in ["bigcodebench", "bcb"]:
        main_formatter = create_bigcodebench_main_formatter(force_aux_usage=force_aux_usage)
        return [bigcodebench_aux_formatter, main_formatter]
    
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_logger_and_aggregator(dataset_type: str):
    """Get the logger and aggregator functions based on dataset type.
    
    For BigCodeBench, returns a wrapper that passes dataset_type to the logger.
    """
    if dataset_type is None:
        return None, None

    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:
        return mt_humaneval_logger, aggregate_mt_humaneval_metrics_for_logging

    # BigCodeBench uses the same logger with dataset_type parameter
    if dataset_type.lower() in ["bigcodebench", "bcb"]:
        def logger_wrapper(**kwargs):
            return mt_humaneval_logger(dataset_type=dataset_type, **kwargs)
        return logger_wrapper, aggregate_mt_humaneval_metrics_for_logging

    return None, None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_problem_from_prompt(formatted_prompt: str) -> str:
    match = re.search(
        r"Problem:\s*(.*?)\n\nIMPORTANT INSTRUCTIONS:", formatted_prompt, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return formatted_prompt.strip()


def build_prompt_lookup(dataset) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for item in dataset:
        raw_prompt = item.get("prompt", "")
        if not raw_prompt:
            continue
        lookup[raw_prompt.strip()] = {
            "prompt": raw_prompt,
            "entry_point": item.get("entry_point", ""),
            "test": item.get("test", ""),
        }
    return lookup


def make_prompt_reward_fn(prompt_lookup: Dict[str, Dict[str, str]]):
    def _reward(
        prompts: List[str], aux_outputs: List[str], main_outputs: List[str]
    ) -> List[float]:
        if not prompts:
            return []

        problem_text = extract_problem_from_prompt(prompts[0])
        meta = prompt_lookup.get(problem_text) or prompt_lookup.get(problem_text.strip())
        if meta is None:
            raise KeyError("Failed to find metadata for provided prompt text.")

        count = min(len(aux_outputs), len(main_outputs))
        if count == 0:
            return []

        test_cases = [meta["test"]] * count
        entry_points = [meta["entry_point"]] * count
        raw_prompts = [meta["prompt"]] * count

        return execution_reward_aux(
            aux_outputs[:count],
            main_outputs[:count],
            test_cases,
            entry_points,
            raw_prompts,
        )

    return _reward


# ============================================================================
# BigCodeBench Reward Functions
# ============================================================================

def extract_problem_from_bcb_prompt(formatted_prompt: str) -> str:
    """Extract problem text from BigCodeBench formatted prompt."""
    match = re.search(
        r"Problem:\s*(.*?)\n\n(?:Available libraries|IMPORTANT INSTRUCTIONS)", 
        formatted_prompt, 
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return formatted_prompt.strip()


def build_prompt_lookup_bigcodebench(dataset) -> Dict[str, Dict[str, str]]:
    """Build a prompt lookup for BigCodeBench dataset.
    
    BigCodeBench uses 'complete_prompt' instead of 'prompt'.
    """
    lookup: Dict[str, Dict[str, str]] = {}
    for item in dataset:
        complete_prompt = item.get("complete_prompt", "")
        if not complete_prompt:
            continue
        lookup[complete_prompt.strip()] = {
            "complete_prompt": complete_prompt,
            "code_prompt": item.get("code_prompt", ""),
            "entry_point": item.get("entry_point", ""),
            "test": item.get("test", ""),
        }
    return lookup


def make_bigcodebench_reward_fn(
    prompt_lookup: Dict[str, Dict[str, str]],
    enforce_collaboration: bool = True,
    self_aux_penalty: float = 0.0,
):
    """Create a reward function for BigCodeBench dataset.
    
    Args:
        prompt_lookup: Dictionary mapping complete_prompt to metadata
        enforce_collaboration: If True, penalize main defining its own aux
        self_aux_penalty: Reward value when collaboration is violated
    """
    def _reward(
        prompts: List[str], aux_outputs: List[str], main_outputs: List[str]
    ) -> List[float]:
        if not prompts:
            return []

        problem_text = extract_problem_from_bcb_prompt(prompts[0])
        meta = prompt_lookup.get(problem_text) or prompt_lookup.get(problem_text.strip())
        if meta is None:
            raise KeyError("Failed to find metadata for provided BigCodeBench prompt text.")

        count = min(len(aux_outputs), len(main_outputs))
        if count == 0:
            return []

        test_cases = [meta["test"]] * count
        entry_points = [meta["entry_point"]] * count
        code_prompts = [meta["code_prompt"]] * count

        return execution_reward_bigcodebench(
            aux_outputs[:count],
            main_outputs[:count],
            test_cases,
            entry_points,
            code_prompts,
            enforce_collaboration=enforce_collaboration,
            self_aux_penalty_value=self_aux_penalty,
        )

    return _reward


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-turn MAAC (shared critic) training for cooperative code generation."
    )
    add_config_args(parser)
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Config: load YAML and apply overrides
    # ------------------------------------------------------------------ #
    if args.config:
        config = Config(args.config)
    else:
        default_config_path = Path(__file__).parent / "configs" / "maac_che_config.yaml"
        if default_config_path.exists():
            config = Config(str(default_config_path))
        else:
            raise ValueError("Please provide a configuration file using --config")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    # ------------------------------------------------------------------ #
    # Config: model, dataset, output
    # ------------------------------------------------------------------ #
    model_config = config.get_model_config()
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")
    train_split = config.get("dataset.train_split") or config.get(
        "dataset.split", "train"
    )
    eval_split = config.get("dataset.eval_split")
    train_size = config.get("dataset.size")
    eval_size = config.get("dataset.eval_size")
    output_base_dir = config.get("output.base_dir", "output")
    output_verbose = config.get("output.verbose", False)

    # Try to infer dataset type if missing
    if dataset_type is None and dataset_name:
        if "humaneval" in dataset_name.lower() and "coop" not in dataset_name.lower():
            dataset_type = "humaneval"
        elif "coophumaneval" in dataset_name.lower() or "coop" in dataset_name.lower():
            dataset_type = "coophumaneval"
        elif "mbpp" in dataset_name.lower():
            dataset_type = "mbpp"
        elif "bigcodebench" in dataset_name.lower() or "bcb" in dataset_name.lower():
            dataset_type = "bigcodebench"
    if dataset_type is None:
        raise ValueError("dataset.type must be specified or inferrable from dataset.name")
    
    if output_verbose:
        print(f"Dataset type: {dataset_type}")

    # ------------------------------------------------------------------ #
    # MAAC-specific config (needed early for seed)
    # ------------------------------------------------------------------ #
    maac_cfg = config.get_section("maac") if hasattr(config, "get_section") else {}
    seed_value = int(config.get("seed", maac_cfg.get("seed", 42)))

    # ------------------------------------------------------------------ #
    # Output directory handling
    # ------------------------------------------------------------------ #
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"maac_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "config.yaml")

    # ------------------------------------------------------------------ #
    # Tokenizer / dataset
    # ------------------------------------------------------------------ #
    _set_seed(seed_value)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **model_config.tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get shuffle options from config
    shuffle_train = config.get("dataset.shuffle_train", False)
    shuffle_eval = config.get("dataset.shuffle_eval", False)

    train_dataset = None
    eval_dataset = None

    # Check if dataset_name is a JSON/JSONL file path
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        train_dataset, eval_dataset = load_json_dataset(
            json_path=dataset_name,
            train_split=train_split,
            eval_split=eval_split,
            verbose=output_verbose,
            shuffle_train=shuffle_train,
            shuffle_eval=shuffle_eval,
            seed=seed_value,
        )
        train_size = len(train_dataset)
        eval_size = len(eval_dataset) if eval_dataset else None
    else:
        # Load from HuggingFace Hub
        train_dataset = load_dataset(dataset_name, split=train_split)
        if train_size is not None:
            train_size = min(int(train_size), len(train_dataset))
            train_dataset = train_dataset.select(range(train_size))
        else:
            train_size = len(train_dataset)
        
        # Shuffle training dataset if requested
        if shuffle_train:
            train_dataset = train_dataset.shuffle(seed=seed_value)
            if output_verbose:
                print(f"Train dataset shuffled with seed={seed_value}")

        if eval_split:
            eval_dataset = load_dataset(dataset_name, split=eval_split)
            if eval_size is not None:
                eval_size = min(int(eval_size), len(eval_dataset))
                eval_dataset = eval_dataset.select(range(eval_size))
            else:
                eval_size = len(eval_dataset)
            
            # Shuffle eval dataset if requested
            if shuffle_eval:
                eval_dataset = eval_dataset.shuffle(seed=seed_value)
                if output_verbose:
                    print(f"Eval dataset shuffled with seed={seed_value}")

    if output_verbose:
        print(f"Using model: {model_name}")
        print(f"Train dataset: {dataset_name} split={train_split} size={train_size}")
        if eval_dataset is not None:
            print(f"Eval dataset: {dataset_name} split={eval_split} size={eval_size}")

    config.update(
        {
            "dataset": {
                "type": dataset_type,
                "train_split": train_split,
                "eval_split": eval_split,
                "size": train_size,
                "eval_size": eval_size,
            }
        }
    )
    if hasattr(config, "save"):
        config.save(config_save_path)

    # ------------------------------------------------------------------ #
    # External context resolver (for multi-turn transitions)
    # ------------------------------------------------------------------ #
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

    if train_dataset is not None:
        _register_split(train_dataset)
    if eval_dataset is not None:
        _register_split(eval_dataset)

    def _resolver(prompt: str):
        return context_map.get(_normalize_prompt(prompt))

    external_ctx.set_context_resolver(_resolver)

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

    # ------------------------------------------------------------------ #
    # Formatters and Reward Function (based on dataset type)
    # ------------------------------------------------------------------ #
    force_aux_usage = maac_cfg.get("force_aux_usage", False)
    formatters = build_prompt_formatters(dataset_type=dataset_type, force_aux_usage=force_aux_usage)
    
    # Load collaboration enforcement settings from config
    collaboration_cfg = (
        config.get_section("collaboration") if hasattr(config, "get_section") else {}
    )
    enforce_collaboration = collaboration_cfg.get("enforce", True)
    self_aux_penalty = collaboration_cfg.get("self_aux_penalty", 0.0)
    
    if output_verbose:
        print(f"[Collaboration] enforce={enforce_collaboration}, self_aux_penalty={self_aux_penalty}")
    
    # Build prompt lookup and reward function based on dataset type
    is_bigcodebench = dataset_type and dataset_type.lower() in ["bigcodebench", "bcb"]
    
    if is_bigcodebench:
        prompt_lookup = build_prompt_lookup_bigcodebench(train_dataset)
        if eval_dataset is not None:
            prompt_lookup.update(build_prompt_lookup_bigcodebench(eval_dataset))
        reward_fn = make_bigcodebench_reward_fn(
            prompt_lookup,
            enforce_collaboration=enforce_collaboration,
            self_aux_penalty=self_aux_penalty,
        )
    else:
        prompt_lookup = build_prompt_lookup(train_dataset)
        if eval_dataset is not None:
            prompt_lookup.update(build_prompt_lookup(eval_dataset))
        reward_fn = make_prompt_reward_fn(prompt_lookup)

    # Get eval logger and aggregator for detailed metrics (fully_passed_rate, etc.)
    eval_logger, eval_aggregator = get_logger_and_aggregator(dataset_type)

    reward_processor = None
    shift_val = maac_cfg.get("reward_shift", -4)
    if shift_val is not None:
        try:
            shift_val_f = float(shift_val)
        except (TypeError, ValueError):
            shift_val_f = None
        if shift_val_f is not None:
            reward_processor = RewardProcessors.shift(value=shift_val_f)

    # ------------------------------------------------------------------ #
    # MAAC-specific config
    # ------------------------------------------------------------------ #
    if "do_sample" in maac_cfg:
        use_sampling = bool(maac_cfg.get("do_sample"))
    else:
        use_sampling = bool(
            "temperature" in maac_cfg
            or "top_p" in maac_cfg
            or "top_k" in maac_cfg
        )
    top_k = maac_cfg.get("top_k")
    temperature = maac_cfg.get("temperature", 0.6)
    top_p = maac_cfg.get("top_p", 0.6)
    critic_model = (
        maac_cfg.get("critic_model")
        or maac_cfg.get("critic_model_name_or_path")
        or model_name
    )
    num_turns = maac_cfg.get("num_turns", 2)
    discount = maac_cfg.get("discount", 0.9)

    external_transition_fn = None
    if num_turns > 1:
        external_mode = external_cfg.get("mode", "level_feedback")
        expert_model = external_cfg.get("expert_model", "deepseek-coder")

        def external_transition_fn(
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

    trainer = MAACTrainer(
        model=model_name,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        reward_processor=reward_processor,
        formatters=formatters,
        metrics_callback=None,
        external_transition=external_transition_fn,
        eval_logger=eval_logger,
        eval_aggregator=eval_aggregator,
        args=MAACConfig(
            output_dir=os.path.join(output_dir, "maac"),
            actor_learning_rate=maac_cfg.get("actor_learning_rate", 5e-6),
            critic_learning_rate=maac_cfg.get("critic_learning_rate", 5e-6),
            value_loss_coef=maac_cfg.get("value_loss_coef", 0.6),
            rollout_buffer_size=maac_cfg.get("rollout_buffer_size", 8),
            max_new_tokens=maac_cfg.get("max_new_tokens", 256),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=use_sampling,
            num_train_epochs=maac_cfg.get("num_train_epochs", 40),
            per_device_train_batch_size=maac_cfg.get("per_device_train_batch_size", 1),
            num_agents=maac_cfg.get("num_agents", 2),
            num_return_sequences=1,
            critic_model_name_or_path=critic_model,
            num_turns=num_turns,
            discount=discount,
            critic_type=maac_cfg.get("critic_type", "v"),
            early_termination_threshold=maac_cfg.get(
                "early_termination_threshold", -0.2
            ),
            eval_interval=maac_cfg.get("eval_interval", 16),
            eval_num_samples=maac_cfg.get("eval_num_samples", 4),
            logging_steps=maac_cfg.get("logging_steps", 1),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_config={
            "tokenizer_kwargs": model_config.tokenizer_kwargs,
            "model_kwargs": model_config.model_kwargs,
            "critic_model_kwargs": maac_cfg.get(
                "critic_model_kwargs", model_config.model_kwargs
            ),
        },
        wandb_config=_build_wandb_config(
            config, dataset_name, train_split, eval_split, train_size, eval_size
        ),
    )
    trainer.train()

    if config.get("output.save_final_model", False):
        save_path = config.get("output.save_path", os.path.join(output_dir, "final_model"))
        trainer.save_model(save_path)
        if output_verbose:
            print(f"Model saved to: {save_path}")

    if wandb.run is not None:
        wandb.finish()


def _build_wandb_config(
    config: Config,
    dataset_name: str,
    train_split: str,
    eval_split: str,
    train_size: int,
    eval_size: int | None,
):
    wandb_section = config.get_section("wandb") if hasattr(config, "get_section") else {}
    maac_section = config.get_section("maac") if hasattr(config, "get_section") else {}
    output_section = (
        config.get_section("output") if hasattr(config, "get_section") else {}
    )
    tags = wandb_section.get("tags", ["maac", dataset_name or "code", "turns_2"])
    return {
        "project": wandb_section.get("project", "maac"),
        "entity": wandb_section.get("entity"),
        "name": wandb_section.get("name", "maac_two_turn"),
        "dir": wandb_section.get("dir"),
        "tags": tags,
        "config_sections": {
            "dataset": {
                "name": dataset_name,
                "train_split": train_split,
                "eval_split": eval_split,
                "train_size": train_size,
                "eval_size": eval_size,
            },
            "output": output_section,
            "trainer": {
                "num_turns": maac_section.get("num_turns", 2),
                "max_new_tokens": maac_section.get("max_new_tokens", 256),
                "temperature": maac_section.get("temperature", 0.6),
                "top_p": maac_section.get("top_p", 0.6),
                "top_k": maac_section.get("top_k"),
                "discount": maac_section.get("discount", 0.9),
                "critic_type": maac_section.get("critic_type", "v"),
            },
        },
    }


if __name__ == "__main__":
    main()
