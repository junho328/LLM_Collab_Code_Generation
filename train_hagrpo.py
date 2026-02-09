"""
Training script for HAGRPO (Heterogeneous-Agent Group Relative Policy Optimization).

HAGRPO combines:
- HAPPO's sequential update scheme with importance ratio accumulation
- GRPO's group-relative advantage estimation (no Value network)

Key features:
1. Sequential agent updates (not simultaneous like MAGRPO)
2. M factor accumulates importance ratios across agents
3. PPO-clip objective with M factor
4. Group reward normalization for advantage estimation
"""

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

from rewards.code_rewards import execution_reward_aux
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.hagrpo import HAGRPOConfig, HAGRPOTrainer
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


def get_formatters(dataset_type: str, num_agents: int):
    """Get a list of per-agent formatters based on dataset type and agent count."""
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"] and num_agents == 2:
        return [aux_function_formatter, main_function_formatter]

    raise NotImplementedError("Other number of agents have not been implemented yet")


def get_logger_and_aggregator(dataset_type: str, is_multi_turn: bool = False):
    """Get the logger and aggregator functions based on dataset type."""
    if dataset_type is None:
        return None, None

    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:
        return mt_humaneval_logger, aggregate_mt_humaneval_metrics_for_logging

    return None, None


def get_reward_function(
    dataset_type: str,
    num_agents: int,
    enforce_collaboration: bool = True,
    self_aux_penalty: float = 0.0,
):
    """Get a reward function compatible with variable number of agents.
    
    Args:
        dataset_type: Type of dataset (humaneval, coophumaneval, mbpp)
        num_agents: Number of agents in the collaboration
        enforce_collaboration: If True, penalize main function defining its own aux
        self_aux_penalty: Reward value when collaboration violation is detected
    """
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval' to the dataset section."
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
                completion1,
                completion2,
                test_cases,
                entry_points,
                original_prompts,
                enforce_collaboration=enforce_collaboration,
                self_aux_penalty_value=self_aux_penalty,
            )

        return reward_wrapper

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main function to run the HAGRPO training."""
    parser = argparse.ArgumentParser(
        description="Train HAGRPO with configurable dataset (single-turn or multi-turn)"
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
        else:
            raise ValueError(
                f"Could not infer dataset type from dataset name '{dataset_name}'. "
                "Please specify 'type' in dataset config."
            )
        print(f"Dataset type not specified, inferred as: {dataset_type}")

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    # ------------------------------------------------------------------
    # Config: HAGRPO training params
    # ------------------------------------------------------------------
    hagrpo_config = (
        config.get_section("hagrpo") if hasattr(config, "get_section") else {}
    )
    # Fall back to magrpo section if hagrpo not found (for compatibility)
    if not hagrpo_config:
        hagrpo_config = (
            config.get_section("magrpo") if hasattr(config, "get_section") else {}
        )

    seed_value = int(config.get("seed", hagrpo_config.get("seed", 42)))
    num_turns = hagrpo_config.get("num_turns", 2)
    num_agents = hagrpo_config.get("num_agents", 2)
    is_multi_turn = num_turns > 1
    output_verbose = config.get("output.verbose", False)

    if output_verbose:
        if is_multi_turn:
            print(f"Multi-turn training enabled: num_turns={num_turns}")
        else:
            print(f"Single-turn training: num_turns={num_turns}")
        print("Using HAGRPO algorithm with sequential agent updates")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")

    if is_multi_turn:
        output_dir = os.path.join(output_base_dir, f"mt_hagrpo_job_{slurm_job_id}")
    else:
        output_dir = os.path.join(output_base_dir, f"hagrpo_job_{slurm_job_id}")

    os.makedirs(output_dir, exist_ok=True)

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)

    _set_seed(seed_value)

    train_dataset = None
    eval_dataset = None
    try:
        train_dataset = load_dataset(dataset_name, split=train_split)
        eval_dataset = load_dataset(dataset_name, split=eval_split)
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

    temperature = hagrpo_config.get("temperature", 0.6)
    top_p = hagrpo_config.get("top_p", 0.6)

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
    # Build training args with HAGRPO-specific parameters
    # ------------------------------------------------------------------
    hagrpo_args_kwargs = {
        "output_dir": output_dir,
        "num_agents": num_agents,
        "num_train_epochs": hagrpo_config.get("num_train_epochs", 20),
        "per_device_train_batch_size": hagrpo_config.get(
            "per_device_train_batch_size", 1
        ),
        "learning_rate": hagrpo_config.get("learning_rate", 5e-6),
        "logging_steps": hagrpo_config.get("logging_steps", 50),
        "save_steps": hagrpo_config.get("save_steps", 200),
        "eval_interval": hagrpo_config.get("eval_interval", 16),
        "eval_num_samples": hagrpo_config.get("eval_num_samples", 4),
        "num_generations": hagrpo_config.get("num_generations", 4),
        "max_new_tokens": hagrpo_config.get("max_new_tokens", 256),
        "temperature": temperature,
        "top_p": top_p,
        # Multi-turn parameters
        "num_turns": num_turns,
        "discount": hagrpo_config.get("discount", 0.9),
        "joint_mode": hagrpo_config.get("joint_mode", "aligned"),
        "termination_threshold": hagrpo_config.get("termination_threshold", -0.2),
        "rollout_buffer_size": hagrpo_config.get("rollout_buffer_size", 2),
        "external_prompt_passthrough": True,
        # HAGRPO-specific parameters
        "ppo_clip_eps": hagrpo_config.get("ppo_clip_eps", 0.2),
        "m_clip_min": hagrpo_config.get("m_clip_min", 0.1),
        "m_clip_max": hagrpo_config.get("m_clip_max", 10.0),
        "shuffle_agent_order": hagrpo_config.get("shuffle_agent_order", False),
        "reverse_agent_order": hagrpo_config.get("reverse_agent_order", False),
        "use_ppo_clip": hagrpo_config.get("use_ppo_clip", True),
        "normalize_log_prob_by_length": hagrpo_config.get("normalize_log_prob_by_length", False),
    }

    if "top_k" in hagrpo_config:
        hagrpo_args_kwargs["top_k"] = hagrpo_config.get("top_k")

    hagrpo_args = HAGRPOConfig(**hagrpo_args_kwargs)

    # ------------------------------------------------------------------
    # Formatters, rewards, and logging
    # ------------------------------------------------------------------
    formatters = get_formatters(dataset_type, num_agents)

    # Load collaboration enforcement settings from config
    collaboration_cfg = (
        config.get_section("collaboration") if hasattr(config, "get_section") else {}
    )
    enforce_collaboration = collaboration_cfg.get("enforce", True)
    self_aux_penalty = collaboration_cfg.get("self_aux_penalty", 0.0)

    print(f"[Collaboration] enforce={enforce_collaboration}, self_aux_penalty={self_aux_penalty}")

    reward_func = get_reward_function(
        dataset_type,
        num_agents,
        enforce_collaboration=enforce_collaboration,
        self_aux_penalty=self_aux_penalty,
    )
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
        wandb_name = wandb_section.get("name", f"mt_hagrpo_{dataset_type}")
    else:
        wandb_name = wandb_section.get("name", f"hagrpo_{dataset_type}")

    external_mode = external_cfg.get("mode", "level_feedback")
    default_tags = ["hagrpo", dataset_type or "code", f"turns_{num_turns}"]
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
            "trainer": hagrpo_config,
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
    # LoRA PEFT configuration
    # ------------------------------------------------------------------
    lora_config_section = (
        config.get_section("lora") if hasattr(config, "get_section") else {}
    )
    use_lora = lora_config_section.get("enabled", False)

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
            print("LoRA Configuration")
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
        # Original full model training
        agents = [
            AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_load_kwargs,
            )
            for _ in range(num_agents)
        ]

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
        "args": hagrpo_args,
        # LoRA configuration
        "use_lora": use_lora,
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

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
        print("HAGRPO Training Configuration")
        print("=" * 60)
        print(f"Algorithm: HAGRPO (Sequential Agent Updates)")
        print(f"Number of Agents: {num_agents}")
        print(f"Number of Turns: {num_turns}")
        # Agent update order
        if hagrpo_args.shuffle_agent_order:
            order_desc = "Random (shuffled each batch)"
        elif hagrpo_args.reverse_agent_order:
            order_desc = "Reverse (Main -> Helper)"
        else:
            order_desc = "Default (Helper -> Main)"
        print(f"Agent Update Order: {order_desc}")
        # Loss function type
        if hagrpo_args.use_ppo_clip:
            loss_desc = f"PPO-Clip (eps={hagrpo_args.ppo_clip_eps})"
        else:
            loss_desc = "Policy Gradient (MAGRPO-style)"
        print(f"Loss Function: {loss_desc}")
        print(f"M Factor Clip Range: [{hagrpo_args.m_clip_min}, {hagrpo_args.m_clip_max}]")
        print("=" * 60 + "\n")

    trainer = HAGRPOTrainer(**trainer_kwargs)
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
