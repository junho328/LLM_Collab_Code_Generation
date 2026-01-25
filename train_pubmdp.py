"""
Training script for PuB-MDP (Public Belief MDP) collaborative code generation.

This script trains a Public Agent that generates prescriptions (specifications)
for two worker agents. The Public Agent learns a meta-policy for coordination,
while worker agents remain frozen and execute code based on prescriptions.

Features:
- Separate model configurations for Public Agent and Worker Agents
- Flash Attention 2 support
- Multi-GPU data parallelism
- Parallel generation and reward computation

System Flow:
1. Public Agent receives task T
2. Public Agent generates z1 (prescription for Agent 1 - auxiliary function)
3. Public Agent generates z2 (prescription for Agent 2 - main function), conditioned on T and z1
4. Worker Agent 1 (frozen): z1 + instruction -> auxiliary function code
5. Worker Agent 2 (frozen): z2 + instruction -> main function code
6. Reward computed from combined execution
7. Only Public Agent is updated via GRPO

Usage:
    # Single GPU
    python train_pubmdp.py --config configs/pubmdp_he_config.yaml
    
    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 train_pubmdp.py --config configs/pubmdp_he_config.yaml
"""

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

from rewards.code_rewards import execution_reward_aux
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.pubmdp import PuBMDPConfig, PuBMDPTrainer


def extract_function_params_from_prompt(prompt_text: str) -> list:
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


def task_formatter(example: Dict[str, Any]) -> str:
    """Format the task for the Public Agent."""
    return example.get("prompt", "")


def create_prescription_prompt_template() -> str:
    """
    Create the prescription prompt template for the Public Agent (z1 generation).
    
    This generates z1 (aux function specification) which is shared with BOTH:
    - Agent 1: Uses z1 to implement the aux function
    - Agent 2: Sees z1 to understand how aux works when implementing main function
    
    Note: Only uses {task} placeholder. Entry point is handled separately
    when generating z2 prescriptions.
    """
    return """You are a coding team leader (Public Agent) coordinating two developers on a collaborative coding task.

## Task Description:
{task}

## Your Role:
You coordinate two developers to solve this problem collaboratively:
- **Developer 1**: Creates a helper function named 'aux'
- **Developer 2**: Creates the main solution function that uses aux()

IMPORTANT: 
- Developer 2 cannot see Developer 1's actual code implementation
- Developer 2 CAN see your specification for the aux function
- Your specification must be detailed enough for both developers to create compatible code

## Instructions:
Write a detailed specification for the auxiliary function 'aux' that Developer 1 will implement.
This specification will be shared with BOTH developers so they understand how aux() works.

Include:
- Purpose: What should the aux function accomplish?
- Parameters: What inputs should it accept? (types and descriptions)
- Return value: What should it return? (type and description)
- Algorithm: Key implementation steps or logic
- Edge cases: How to handle special cases

[SPECIFICATION FOR AUX FUNCTION]
"""


def create_aux_instruction_template():
    """Create the instruction template for Worker Agent 1 (auxiliary function)."""
    def aux_template(task: str, prescription: str, entry_point: str) -> str:
        return f"""You are Developer 1 on a coding team. Create a helper function based on the specification from your team leader.

## Problem Context:
{task}

## Team Leader's Specification for Your Helper Function:
{prescription}

## Requirements:
- Create a helper function named 'aux'
- Follow the specification exactly
- Output ONLY the function code
- Do NOT include markdown code blocks (```python)
- Do NOT include explanations, examples, or test cases
- Do NOT include any text before or after the function

## Output Format:
def aux(...):\n    # implementation\n    return result
"""
    return aux_template


def create_main_instruction_template():
    """
    Create the instruction template for Worker Agent 2 (main function).
    
    Agent 2 receives:
    - z1 (aux_prescription): Specification of how aux() works (shared with Agent 1)
    - z2 (main_prescription): Specific instructions for the main function
    """
    def main_template(task: str, prescription: str, entry_point: str, aux_prescription: str = None) -> str:
        params = extract_function_params_from_prompt(task)
        params_str = ", ".join(params) if params else "..."
        
        # If aux_prescription is provided separately, include it
        aux_spec_section = ""
        if aux_prescription:
            aux_spec_section = f"""## Available Helper Function - aux():
The following specification describes what the aux() function does.
Developer 1 implemented this function based on this specification:

{aux_prescription}

---

"""
        
        return f"""You are Developer 2 on a coding team. Implement the main function based on the specifications from your team leader.

## Problem:
{task}

{aux_spec_section}## Team Leader's Specification for Your Main Function:
{prescription}

## Requirements:
- Implement the '{entry_point}' function as specified
- You SHOULD use the aux() helper function according to the specifications above
- Output ONLY the function code
- Do NOT include markdown code blocks (```python)
- Do NOT include explanations, examples, or test cases
- Do NOT redefine the aux() function
- Do NOT include any text before or after the function

## Output Format:
def {entry_point}({params_str}):\n    # implementation\n    return result
"""
    return main_template


def get_reward_function(dataset_type: str):
    """Get the reward function for code evaluation."""
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:
        def reward_wrapper(completion1, completion2, batch_items=None):
            if batch_items is None:
                raise ValueError("batch_items must be provided for reward calculation")
            
            test_cases = []
            entry_points = []
            original_prompts = []
            
            for item in batch_items:
                test_cases.append(item["test"])
                entry_points.append(item["entry_point"])
                original_prompts.append(item.get("prompt", ""))
            
            return execution_reward_aux(
                completion1, completion2, test_cases, entry_points, original_prompts
            )
        
        return reward_wrapper
    
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_logger_and_aggregator(dataset_type: str):
    """Get the logger and aggregator functions."""
    if dataset_type is None:
        return None, None
    
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:
        return mt_humaneval_logger, aggregate_mt_humaneval_metrics_for_logging
    
    return None, None


def load_model_with_config(
    model_name: str,
    model_config: Dict[str, Any],
    use_flash_attention: bool = True,
    device_map: Optional[str] = None,
    local_rank: int = -1,
) -> AutoModelForCausalLM:
    """
    Load a model with the specified configuration.
    
    Args:
        model_name: HuggingFace model name or path
        model_config: Model configuration dictionary
        use_flash_attention: Whether to use Flash Attention 2
        device_map: Device map for model placement
        local_rank: Local rank for distributed training
    
    Returns:
        Loaded model
    """
    model_kwargs = dict(model_config.get("model_kwargs", {}))
    
    # Handle torch_dtype
    if "torch_dtype" in model_kwargs:
        dtype_str = model_kwargs["torch_dtype"]
        if dtype_str == "auto":
            model_kwargs["torch_dtype"] = "auto"
        elif dtype_str == "float16" or dtype_str == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype_str == "bfloat16" or dtype_str == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype_str == "float32" or dtype_str == "fp32":
            model_kwargs["torch_dtype"] = torch.float32
    
    # Enable Flash Attention 2
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        # Flash Attention 2 requires bfloat16 or float16
        if model_kwargs.get("torch_dtype") == "auto":
            model_kwargs["torch_dtype"] = torch.bfloat16
    
    # Handle device placement for distributed training
    if local_rank != -1:
        # For distributed training, load to specific GPU
        model_kwargs["device_map"] = {"": local_rank}
    elif device_map:
        model_kwargs["device_map"] = device_map
    
    # Remove device_map if it would conflict
    if "device_map" in model_kwargs and model_kwargs["device_map"] is None:
        del model_kwargs["device_map"]
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    
    return model


def load_tokenizer_with_config(
    model_name: str,
    tokenizer_config: Dict[str, Any],
) -> AutoTokenizer:
    """Load a tokenizer with the specified configuration."""
    tokenizer_kwargs = dict(tokenizer_config.get("tokenizer_kwargs", {}))
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **tokenizer_kwargs,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def setup_distributed():
    """Setup distributed training environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"Initialized distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
    else:
        world_size = 1
        rank = 0
    
    return local_rank, world_size, rank


def main():
    """Main function to run PuB-MDP training."""
    parser = argparse.ArgumentParser(
        description="Train PuB-MDP Public Agent for collaborative code generation"
    )
    add_config_args(parser)
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    is_main_process = rank == 0
    
    # Load config
    if args.config:
        config = Config(args.config)
    else:
        raise ValueError("Please provide a configuration file using --config")
    
    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)
    
    # Get configurations
    model_config = config.get_model_config()
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")
    output_base_dir = config.get("output.base_dir")
    
    # Infer dataset type if not specified
    if dataset_type is None:
        if "humaneval" in dataset_name.lower() and "coop" not in dataset_name.lower():
            dataset_type = "humaneval"
        elif "coophumaneval" in dataset_name.lower() or "coop" in dataset_name.lower():
            dataset_type = "coophumaneval"
        elif "mbpp" in dataset_name.lower():
            dataset_type = "mbpp"
        else:
            raise ValueError(
                f"Could not infer dataset type from '{dataset_name}'. Please specify 'type' in dataset config."
            )
        if is_main_process:
            print(f"Dataset type inferred as: {dataset_type}")
    
    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")
    
    # PuB-MDP specific config
    pubmdp_config = config.get_section("pubmdp") if hasattr(config, "get_section") else {}
    seed_value = int(config.get("seed", pubmdp_config.get("seed", 42)))
    output_verbose = config.get("output.verbose", False)
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"pubmdp_job_{slurm_job_id}")
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    if is_main_process and hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)
    
    _set_seed(seed_value + rank)  # Different seed per rank for diversity
    
    # Load datasets
    train_dataset = None
    eval_dataset = None
    try:
        train_dataset = load_dataset(dataset_name, split=train_split)
        eval_dataset = load_dataset(dataset_name, split=eval_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Model configurations - separate configs for public and worker agents
    public_model_config = pubmdp_config.get("public_model_config", {})
    worker_model_config = pubmdp_config.get("worker_model_config", {})
    
    # Model names
    public_model_name = pubmdp_config.get("public_model", model_config.name)
    worker_model_name = pubmdp_config.get("worker_model", model_config.name)
    
    # Use Flash Attention 2 by default
    use_flash_attention = pubmdp_config.get("use_flash_attention", True)
    
    if is_main_process and output_verbose:
        print(f"\nPublic Agent model: {public_model_name}")
        print(f"Worker Agent model: {worker_model_name}")
        print(f"Flash Attention 2: {use_flash_attention}")
        print(f"World size: {world_size}")
    
    # Merge model configs with defaults
    default_model_kwargs = dict(model_config.model_kwargs) if hasattr(model_config, 'model_kwargs') else {}
    default_tokenizer_kwargs = dict(model_config.tokenizer_kwargs) if hasattr(model_config, 'tokenizer_kwargs') else {}
    
    public_model_cfg = {
        "model_kwargs": {**default_model_kwargs, **public_model_config.get("model_kwargs", {})},
        "tokenizer_kwargs": {**default_tokenizer_kwargs, **public_model_config.get("tokenizer_kwargs", {})},
    }
    
    worker_model_cfg = {
        "model_kwargs": {**default_model_kwargs, **worker_model_config.get("model_kwargs", {})},
        "tokenizer_kwargs": {**default_tokenizer_kwargs, **worker_model_config.get("tokenizer_kwargs", {})},
    }
    
    # Load Public Agent tokenizer
    if is_main_process and output_verbose:
        print("Loading Public Agent tokenizer...")
    public_tokenizer = load_tokenizer_with_config(public_model_name, public_model_cfg)
    
    # Load Worker Agent tokenizer (may be same as public)
    if worker_model_name == public_model_name:
        worker_tokenizer = public_tokenizer
    else:
        if is_main_process and output_verbose:
            print("Loading Worker Agent tokenizer...")
        worker_tokenizer = load_tokenizer_with_config(worker_model_name, worker_model_cfg)
    
    padding_side = config.get("tokenizer.padding_side")
    if padding_side:
        public_tokenizer.padding_side = padding_side
        worker_tokenizer.padding_side = padding_side
    
    # Add special tokens if needed
    if model_config.special_tokens:
        if is_main_process and output_verbose:
            print("Adding special tokens...")
        public_tokenizer.add_special_tokens(model_config.special_tokens)
        if worker_tokenizer is not public_tokenizer:
            worker_tokenizer.add_special_tokens(model_config.special_tokens)
    
    # Load Public Agent (trainable)
    if is_main_process and output_verbose:
        print("Loading Public Agent (trainable)...")
    public_agent = load_model_with_config(
        public_model_name,
        public_model_cfg,
        use_flash_attention=use_flash_attention,
        local_rank=local_rank,
    )
    
    # Resize embeddings if special tokens were added
    if model_config.special_tokens:
        public_agent.resize_token_embeddings(len(public_tokenizer))
    
    # Load Worker Agents (frozen)
    if is_main_process and output_verbose:
        print("Loading Worker Agents (frozen)...")
    worker_agents = []
    for i in range(2):
        worker = load_model_with_config(
            worker_model_name,
            worker_model_cfg,
            use_flash_attention=use_flash_attention,
            local_rank=local_rank,
        )
        if model_config.special_tokens and worker_tokenizer is not public_tokenizer:
            worker.resize_token_embeddings(len(worker_tokenizer))
        elif model_config.special_tokens:
            worker.resize_token_embeddings(len(public_tokenizer))
        worker_agents.append(worker)
    
    # Build PuB-MDP config
    train_workers = pubmdp_config.get("train_workers", False)
    worker_learning_rate = pubmdp_config.get("worker_learning_rate", None)
    
    pubmdp_args = PuBMDPConfig(
        output_dir=output_dir,
        num_workers=2,
        num_train_epochs=pubmdp_config.get("num_train_epochs", 4),
        per_device_train_batch_size=pubmdp_config.get("per_device_train_batch_size", 1),
        learning_rate=pubmdp_config.get("learning_rate", 5e-6),
        logging_steps=pubmdp_config.get("logging_steps", 50),
        save_steps=pubmdp_config.get("save_steps", 200),
        eval_interval=pubmdp_config.get("eval_interval", 16),
        eval_num_samples=pubmdp_config.get("eval_num_samples", 4),
        num_generations=pubmdp_config.get("num_generations", 4),
        max_prescription_tokens=pubmdp_config.get("max_prescription_tokens", 512),
        max_code_tokens=pubmdp_config.get("max_code_tokens", 256),
        temperature=pubmdp_config.get("temperature", 0.7),
        top_p=pubmdp_config.get("top_p", 0.9),
        rollout_buffer_size=pubmdp_config.get("rollout_buffer_size", 64),
        num_reward_workers=pubmdp_config.get("num_reward_workers", 4),
        # Worker training options
        train_workers=train_workers,
        worker_learning_rate=worker_learning_rate,
    )
    
    if "top_k" in pubmdp_config:
        pubmdp_args.top_k = pubmdp_config.get("top_k")
    
    # Get reward function and loggers
    reward_func = get_reward_function(dataset_type)
    eval_logger, eval_aggregator = get_logger_and_aggregator(dataset_type)
    
    # Reward processor
    reward_processor = None
    if config.get("reward_processor.enabled", True):
        scale_factor = config.get("reward_processor.scale_factor", 1.0)
        reward_processor = RewardProcessors.scale(factor=scale_factor)
        shift_val = config.get("reward_processor.shift", None)
        if shift_val is not None:
            try:
                shift_val_f = float(shift_val)
                shift_proc = RewardProcessors.shift(value=shift_val_f)
                prev = reward_processor
                reward_processor = lambda x, p=prev, s=shift_proc: s(p(x))
            except (TypeError, ValueError):
                pass
    
    # W&B configuration
    wandb_section = config.get_section("wandb") if hasattr(config, "get_section") else {}
    wandb_name = wandb_section.get("name", f"pubmdp_{dataset_type}")
    
    default_tags = ["pubmdp", dataset_type or "code", "meta-policy"]
    tags = wandb_section.get("tags", default_tags)
    if isinstance(tags, list) and "pubmdp" not in tags:
        tags.append("pubmdp")
    if use_flash_attention and isinstance(tags, list):
        tags.append("flash-attention-2")
    
    wandb_config = {
        "project": wandb_section.get("project", "pubmdp-coding"),
        "entity": wandb_section.get("entity", "contrl"),
        "name": wandb_name,
        "dir": wandb_section.get("dir", "./wandb_logs"),
        "tags": tags,
        "config_sections": {
            "dataset": config.get_section("dataset") if hasattr(config, "get_section") else {},
            "model": config.get_section("model") if hasattr(config, "get_section") else {},
            "output": config.get_section("output") if hasattr(config, "get_section") else {},
            "pubmdp": pubmdp_config,
        },
    }
    
    # Create trainer
    trainer = PuBMDPTrainer(
        public_agent=public_agent,
        worker_agents=worker_agents,
        public_tokenizer=public_tokenizer,
        worker_tokenizers=[worker_tokenizer, worker_tokenizer],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_type=dataset_type,
        reward_func=reward_func,
        reward_processor=reward_processor,
        task_formatter=task_formatter,
        prescription_prompt_template=create_prescription_prompt_template(),
        aux_instruction_template=create_aux_instruction_template(),
        main_instruction_template=create_main_instruction_template(),
        wandb_config=wandb_config,
        eval_logger=eval_logger,
        eval_aggregator=eval_aggregator,
        args=pubmdp_args,
        local_rank=local_rank,
    )
    
    # Set verbosity
    trainer.verbose = output_verbose
    
    # Train
    if is_main_process and output_verbose:
        print("\n" + "="*60)
        print("Starting PuB-MDP training...")
        print("="*60)
        print(f"Public Agent: {public_model_name}")
        print(f"Worker Agents: {worker_model_name}")
        print(f"Train Workers: {train_workers}")
        if train_workers:
            print(f"Worker Learning Rate: {worker_learning_rate or pubmdp_args.learning_rate}")
        print(f"Flash Attention 2: {use_flash_attention}")
        print(f"World size: {world_size}")
        print(f"Num generations per task: {pubmdp_args.num_generations}")
        print(f"Learning rate (Public): {pubmdp_args.learning_rate}")
        print(f"Batch size per device: {pubmdp_args.per_device_train_batch_size}")
        print(f"Rollout buffer size: {pubmdp_args.rollout_buffer_size}")
        print("="*60 + "\n")
    
    try:
        trainer.train()
    finally:
        trainer.cleanup()
    
    # Save model
    save_final = config.get("output.save_final_model", False)
    if save_final and is_main_process:
        save_path = config.get(
            "output.save_path", os.path.join(output_dir, "final_model")
        )
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
