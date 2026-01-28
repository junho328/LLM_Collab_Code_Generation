"""
Prescription-based GRPO Training Script.

This is a single-agent GRPO where only the Public Agent is trained.
Worker agents are frozen and used only for code generation (inference).

Architecture:
- Public Agent (TRAINED via GRPO): Generates prescriptions/guidelines for workers
- Worker Agent 1 (FROZEN): Generates aux() function based on prescription
- Worker Agent 2 (FROZEN): Generates main() function based on prescription

Training Flow:
1. Public Agent generates N prescriptions (N = num_generations)
2. Each prescription is parsed into aux/main guidelines
3. Worker agents generate code for each prescription (1 code per prescription)
4. Rewards are computed based on code execution
5. Public Agent is updated via standard GRPO

Note: We reuse MAGRPOConfig for convenience, but this is fundamentally
single-agent GRPO since only one model's parameters are updated.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from config import Config, add_config_args, parse_overrides
from rewards.code_rewards import execution_reward_aux
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.magrpo import MAGRPOConfig
import external as external_ctx


# =============================================================================
# Prescription Format Constants
# =============================================================================

PRESCRIPTION_FORMAT = """<AUX_PRESCRIPTION>
Function Name: aux
Purpose: {aux_purpose}
Input Parameters: {aux_inputs}
Output: {aux_output}
Implementation Guideline: {aux_guideline}
</AUX_PRESCRIPTION>

<MAIN_PRESCRIPTION>
Function Name: {entry_point}
Purpose: {main_purpose}
Input Parameters: {main_inputs}
Output: {main_output}
How to use aux(): {aux_usage}
Implementation Guideline: {main_guideline}
</MAIN_PRESCRIPTION>"""


# =============================================================================
# Utility Functions
# =============================================================================

def extract_function_params_from_prompt(prompt_text: str) -> List[str]:
    """Extract function parameters from the prompt text."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", prompt_text)
    if match:
        params_str = match.group(1)
        if params_str.strip():
            params = [p.strip().split(":")[0].strip() for p in params_str.split(",") if p.strip()]
            return params
    return []


def parse_prescription(prescription_text: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Parse the prescription text into aux and main prescription dictionaries.
    
    Returns:
        Tuple of (aux_prescription, main_prescription) dictionaries.
        Each contains: purpose, inputs, output, guideline, and optionally aux_usage for main.
    """
    aux_prescription = None
    main_prescription = None
    
    # Parse AUX_PRESCRIPTION block
    aux_match = re.search(
        r"<AUX_PRESCRIPTION>(.*?)</AUX_PRESCRIPTION>",
        prescription_text,
        re.DOTALL
    )
    if aux_match:
        aux_block = aux_match.group(1)
        aux_prescription = {
            "purpose": _extract_field(aux_block, "Purpose"),
            "inputs": _extract_field(aux_block, "Input Parameters"),
            "output": _extract_field(aux_block, "Output"),
            "guideline": _extract_field(aux_block, "Implementation Guideline"),
        }
    
    # Parse MAIN_PRESCRIPTION block
    main_match = re.search(
        r"<MAIN_PRESCRIPTION>(.*?)</MAIN_PRESCRIPTION>",
        prescription_text,
        re.DOTALL
    )
    if main_match:
        main_block = main_match.group(1)
        main_prescription = {
            "purpose": _extract_field(main_block, "Purpose"),
            "inputs": _extract_field(main_block, "Input Parameters"),
            "output": _extract_field(main_block, "Output"),
            "aux_usage": _extract_field(main_block, "How to use aux()"),
            "guideline": _extract_field(main_block, "Implementation Guideline"),
        }
    
    return aux_prescription, main_prescription


def _extract_field(block: str, field_name: str) -> str:
    """Extract a field value from a prescription block."""
    pattern = rf"{re.escape(field_name)}:\s*(.+?)(?=\n[A-Z]|\n<|$)"
    match = re.search(pattern, block, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# =============================================================================
# Public Agent Formatter
# =============================================================================

def public_agent_formatter(example: Dict[str, Any]) -> str:
    """
    Formatter for the Public Agent that generates prescriptions.
    
    The Public Agent analyzes the problem and outputs structured prescriptions
    for how worker agents should implement aux() and main() functions.
    """
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")
    
    params = extract_function_params_from_prompt(prompt)
    params_str = ", ".join(params) if params else "..."
    
    formatter_prompt = f"""You are a senior software architect. Analyze the following coding problem and create detailed implementation prescriptions for two developers:
- Developer 1 will implement a helper function called 'aux()'
- Developer 2 will implement the main function '{entry_point}()' using the aux() helper

Problem:
{prompt}

Your task: Create clear, specific prescriptions that guide each developer on exactly how to implement their function.

IMPORTANT INSTRUCTIONS:
- Be specific about input types, output types, and the algorithm to use
- The aux() function should handle a meaningful subtask that helps solve the main problem
- Explain exactly how main() should use aux() to solve the complete problem
- Do NOT suggest using external libraries (math, numpy, etc.)
- Use only built-in Python operations (int(), //, %, list comprehensions, etc.)
- Output ONLY the prescription in the exact format below, no other text

Output format:
<AUX_PRESCRIPTION>
Function Name: aux
Purpose: [Describe what aux() should accomplish]
Input Parameters: [List parameters with types]
Output: [Describe return value and type]
Implementation Guideline: [Step-by-step implementation guide]
</AUX_PRESCRIPTION>

<MAIN_PRESCRIPTION>
Function Name: {entry_point}
Purpose: [Describe what {entry_point}() should accomplish]
Input Parameters: {params_str}
Output: [Describe return value with types]
How to use aux(): [Explain exactly how to call and use aux() result]
Implementation Guideline: [Step-by-step implementation guide]
</MAIN_PRESCRIPTION>
"""
    return formatter_prompt


# =============================================================================
# Worker Agent Formatters
# =============================================================================

def worker_aux_formatter(
    example: Dict[str, Any],
    aux_prescription: Optional[Dict] = None
) -> str:
    """
    Formatter for Worker Agent 1 that generates the aux() function.
    Uses the prescription from the Public Agent.
    """
    prompt = example.get("prompt", "")
    
    if aux_prescription is None:
        prescription_text = "Create a useful helper function."
    else:
        prescription_text = f"""Purpose: {aux_prescription.get('purpose', 'N/A')}
Input Parameters: {aux_prescription.get('inputs', 'N/A')}
Output: {aux_prescription.get('output', 'N/A')}
Implementation Guideline: {aux_prescription.get('guideline', 'N/A')}"""
    
    formatter_prompt = f"""You are implementing a helper function based on the following prescription.

Original Problem:
{prompt}

=== YOUR PRESCRIPTION ===
{prescription_text}
=========================

IMPORTANT INSTRUCTIONS:
- Output ONLY the complete, working function code
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT output placeholder code like "# your implementation" or "return result"
- You MUST write the actual implementation with real logic
- Follow the prescription exactly
- The function MUST be named 'aux'
- Do NOT import or use external libraries (math, numpy, etc.)
- Use only built-in Python operations (int(), //, %, etc.)

Write the complete aux() function now:
"""
    return formatter_prompt


def worker_main_formatter(
    example: Dict[str, Any],
    main_prescription: Optional[Dict] = None,
    aux_prescription: Optional[Dict] = None,
) -> str:
    """
    Formatter for Worker Agent 2 that generates the main() function.
    Uses both main_prescription and aux_prescription from the Public Agent.
    
    Args:
        example: Dataset item with prompt, entry_point, etc.
        main_prescription: Guidelines for implementing main()
        aux_prescription: Guidelines for aux() - needed to know how to call it
    """
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")
    params = extract_function_params_from_prompt(prompt)
    params_str = ", ".join(params) if params else "..."
    
    # Build aux function description so main worker knows how to use it
    if aux_prescription is None:
        aux_description = "aux(...) - A helper function is available."
    else:
        aux_inputs = aux_prescription.get('inputs', '...')
        aux_output = aux_prescription.get('output', 'result')
        aux_purpose = aux_prescription.get('purpose', 'Helper function')
        aux_description = f"""def aux({aux_inputs}):
    \"\"\"
    {aux_purpose}
    Returns: {aux_output}
    \"\"\""""
    
    if main_prescription is None:
        prescription_text = "Implement the main function using aux() helper."
    else:
        prescription_text = f"""Purpose: {main_prescription.get('purpose', 'N/A')}
Input Parameters: {main_prescription.get('inputs', 'N/A')}
Output: {main_prescription.get('output', 'N/A')}
How to use aux(): {main_prescription.get('aux_usage', 'N/A')}
Implementation Guideline: {main_prescription.get('guideline', 'N/A')}"""
    
    formatter_prompt = f"""You are implementing the main function based on the following prescription.

Original Problem:
{prompt}

=== AVAILABLE HELPER FUNCTION ===
{aux_description}
=================================

=== YOUR PRESCRIPTION ===
{prescription_text}
=========================

IMPORTANT INSTRUCTIONS:
- Output ONLY the complete, working function code
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT output placeholder code like "# your implementation" or "return result"
- You MUST write the actual implementation with real logic
- Do NOT redefine the aux() function
- Follow the prescription exactly
- Implement ONLY the '{entry_point}' function
- Use the aux() function as described above if it is helpful
- Do NOT import or use external libraries (math, numpy, etc.)
- Use only built-in Python operations (int(), //, %, etc.)

Write the complete {entry_point}() function now:
"""
    return formatter_prompt


# =============================================================================
# Prescription GRPO Trainer (Single-Agent: Public Agent Only)
# =============================================================================

class PrescriptionGRPOTrainer:
    """
    Prescription-based GRPO Trainer.
    
    This is a single-agent GRPO where only the Public Agent is trained.
    Worker agents are frozen and used only for inference.
    
    Architecture:
    - public_agent: Generates prescriptions (TRAINED via GRPO)
    - worker_agents: Generate code based on prescriptions (FROZEN, inference only)
    
    Training flow:
    1. Public Agent generates num_generations prescriptions
    2. Each prescription is parsed into aux/main guidelines
    3. Worker agents generate code for each prescription (1 code per prescription)
    4. Rewards are computed for each (aux, main) pair
    5. Public Agent is updated via GRPO using the rewards
    
    Note: Despite using multiple models, this is GRPO (not MAGRPO) because
    only one agent (public_agent) is being trained.
    """
    
    def __init__(
        self,
        public_agent: PreTrainedModel,
        worker_agents: List[PreTrainedModel],
        tokenizer,
        train_dataset,
        eval_dataset=None,
        reward_func=None,
        reward_processor=None,
        wandb_config=None,
        args: Optional[MAGRPOConfig] = None,
        dataset_type: str = None,
        max_new_tokens_prescription: int = 512,
        max_new_tokens_code: int = 256,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not found. PrescriptionGRPOTrainer requires GPU.")
        
        self.args = args if args is not None else MAGRPOConfig()
        self.env_step = 0
        
        # Separate max_new_tokens for prescription and code generation
        self.max_new_tokens_prescription = max_new_tokens_prescription
        self.max_new_tokens_code = max_new_tokens_code
        
        # LoRA setup for public agent
        self.use_lora = use_lora
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT library is required for LoRA training. "
                    "Install with: pip install peft"
                )
            
            # Default LoRA config
            default_lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "bias": "none",
            }
            if lora_config:
                default_lora_config.update(lora_config)
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=default_lora_config["r"],
                lora_alpha=default_lora_config["lora_alpha"],
                lora_dropout=default_lora_config["lora_dropout"],
                target_modules=default_lora_config["target_modules"],
                bias=default_lora_config["bias"],
            )
            
            # Wrap public agent with LoRA
            public_agent = get_peft_model(public_agent, peft_config)
            public_agent.print_trainable_parameters()
        
        # Public agent (trained)
        self.public_agent = public_agent
        
        # Worker agents (frozen) - expects 2 workers: aux and main
        if len(worker_agents) != 2:
            raise ValueError("PrescriptionGRPOTrainer requires exactly 2 worker agents")
        self.worker_agents = worker_agents
        
        # Freeze worker agents
        for worker in self.worker_agents:
            for param in worker.parameters():
                param.requires_grad = False
        
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Reward setup
        if reward_func is None or not callable(reward_func):
            raise ValueError("reward_func must be a callable")
        self.reward_func = reward_func
        self.reward_processor = reward_processor if reward_processor else (lambda x: x)
        
        # Optimizer only for public agent (trainable parameters only)
        trainable_params = [p for p in self.public_agent.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        
        # W&B setup
        self.wandb_config = wandb_config
        self.wandb_initialized = False
        self.dataset_type = dataset_type
        if self.wandb_config is not None:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases for tracking."""
        if not self.wandb_initialized and self.wandb_config:
            wandb_project = self.wandb_config.get("project", "comlrl")
            wandb_entity = self.wandb_config.get("entity", "OpenMLRL")
            wandb_name = self.wandb_config.get("name", "prescription-magrpo")
            wandb_dir = self.wandb_config.get("dir", None)
            
            config_dict = {
                "architecture": "prescription-magrpo",
                "public_agent_trained": True,
                "worker_agents_frozen": True,
                "num_generations": self.args.num_generations,
                "learning_rate": self.args.learning_rate,
                "max_new_tokens": self.args.max_new_tokens,
            }
            
            sections = self.wandb_config.get("config_sections", {})
            if isinstance(sections, dict):
                config_dict.update(sections)
            
            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "config": config_dict,
            }
            if wandb_dir:
                os.makedirs(wandb_dir, exist_ok=True)
                init_kwargs["dir"] = wandb_dir
            
            tags = self.wandb_config.get("tags", ["prescription-magrpo"])
            if isinstance(tags, list):
                init_kwargs["tags"] = tags
            
            wandb.init(**init_kwargs)
            self.wandb_initialized = True
    
    def _generate_from_agent(
        self,
        agent: PreTrainedModel,
        prompts: List[str],
        num_return_sequences: int = 1,
        max_new_tokens: int = 256,
        do_sample: bool = True,
    ) -> List[List[str]]:
        """
        Generate completions from an agent given prompts.
        
        Args:
            agent: The model to generate from
            prompts: List of prompt strings
            num_return_sequences: Number of completions per prompt
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            
        Returns:
            List of list of completions (outer: prompts, inner: sequences)
        """
        device = agent.device
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store original state
        training_mode = agent.training
        original_requires_grad = {
            name: param.requires_grad for name, param in agent.named_parameters()
        }
        
        # Disable gradients for generation
        for param in agent.parameters():
            param.requires_grad = False
        agent.eval()
        
        all_completions = []
        
        try:
            for prompt in prompts:
                encodings = self.tokenizer(
                    prompt,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                generation_kwargs = {
                    "input_ids": encodings.input_ids,
                    "attention_mask": encodings.attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "num_return_sequences": num_return_sequences,
                }
                
                if do_sample and num_return_sequences > 1:
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": self.args.temperature,
                        "top_p": self.args.top_p,
                        "num_beams": 1,
                    })
                    if self.args.top_k:
                        generation_kwargs["top_k"] = self.args.top_k
                elif do_sample:
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": self.args.temperature,
                        "top_p": self.args.top_p,
                    })
                
                outputs = agent.generate(**generation_kwargs)
                
                prompt_len = encodings.input_ids.shape[1]
                completions = []
                for seq in outputs:
                    completion_tokens = seq[prompt_len:]
                    completion_text = self.tokenizer.decode(
                        completion_tokens, skip_special_tokens=True
                    )
                    completions.append(completion_text)
                
                all_completions.append(completions)
        
        finally:
            # Restore original state
            agent.train(training_mode)
            for name, param in agent.named_parameters():
                if name in original_requires_grad:
                    param.requires_grad = original_requires_grad[name]
        
        return all_completions
    
    def _compute_rewards(
        self,
        aux_completions: List[str],
        main_completions: List[str],
        batch_item: Dict[str, Any],
    ) -> List[float]:
        """Compute rewards for aux/main completion pairs."""
        rewards = []
        
        test_case = batch_item.get("test", "")
        entry_point = batch_item.get("entry_point", "")
        prompt = batch_item.get("prompt", "")
        
        for aux_code, main_code in zip(aux_completions, main_completions):
            raw_rewards = self.reward_func(
                [aux_code], [main_code],
                batch_items=[batch_item]
            )
            processed = self.reward_processor(raw_rewards[0]) if raw_rewards else 0.0
            rewards.append(float(processed))
        
        return rewards
    
    def _compute_grpo_loss(
        self,
        prompt: str,
        completions: List[str],
        rewards: List[float],
    ) -> torch.Tensor:
        """
        Compute GRPO loss for the public agent.
        
        Args:
            prompt: The input prompt
            completions: List of prescription completions
            rewards: List of rewards for each completion
            
        Returns:
            The computed loss tensor
        """
        device = self.public_agent.device
        
        if not rewards:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        
        # Group-relative advantage (GRPO baseline)
        mean_reward = rewards_tensor.mean()
        advantages = rewards_tensor - mean_reward
        
        self.public_agent.train()
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0
        
        # Encode prompt
        prompt_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(device)
        prompt_ids = prompt_encoding.input_ids[0]
        prompt_len = len(prompt_ids)
        
        for seq_idx, completion in enumerate(completions):
            if seq_idx >= len(advantages):
                break
            
            advantage = advantages[seq_idx]
            
            # Encode completion
            completion_encoding = self.tokenizer(
                completion,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=False,
            ).to(device)
            completion_ids = completion_encoding.input_ids[0]
            
            if len(completion_ids) == 0:
                continue
            
            # Build full sequence
            input_ids = torch.cat([prompt_ids, completion_ids[:-1]])
            attention_mask = torch.ones(len(input_ids), device=device)
            
            # Forward pass
            outputs = self.public_agent(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            )
            
            # Get logits for completion positions
            completion_logits = outputs.logits[0, prompt_len - 1:-1, :]
            
            # Compute log probabilities
            log_probs = []
            for i, token_id in enumerate(completion_ids):
                if i < completion_logits.size(0):
                    token_log_prob = torch.log_softmax(completion_logits[i], dim=-1)[token_id]
                    log_probs.append(token_log_prob)
            
            if log_probs:
                sequence_log_prob = torch.stack(log_probs).sum()
                loss = -sequence_log_prob * advantage
                total_loss = total_loss + loss
                num_samples += 1
        
        if num_samples > 0:
            total_loss = total_loss / num_samples
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)
        
        return total_loss
    
    def _train_step(self, batch_item: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Execute one training step.
        
        1. Public agent generates num_generations prescriptions
        2. Parse each prescription into aux/main guidelines
        3. Worker agents generate code for each prescription
        4. Compute rewards
        5. Update public agent via GRPO
        
        Returns:
            Tuple of (loss_value, metrics_dict)
        """
        num_gens = self.args.num_generations
        
        # Step 1: Generate prescriptions from public agent
        public_prompt = public_agent_formatter(batch_item)
        prescriptions = self._generate_from_agent(
            self.public_agent,
            [public_prompt],
            num_return_sequences=num_gens,
            max_new_tokens=self.max_new_tokens_prescription,  # Prescription: longer
            do_sample=True,
        )[0]  # [0] because single prompt
        
        # Step 2 & 3: Parse prescriptions and generate code from workers
        aux_completions = []
        main_completions = []
        
        for prescription in prescriptions:
            aux_presc, main_presc = parse_prescription(prescription)
            
            # Worker 1: Generate aux code
            aux_prompt = worker_aux_formatter(batch_item, aux_presc)
            aux_code = self._generate_from_agent(
                self.worker_agents[0],
                [aux_prompt],
                num_return_sequences=1,
                max_new_tokens=self.max_new_tokens_code,  # Code: shorter
                do_sample=False,  # Deterministic for workers
            )[0][0]
            aux_completions.append(aux_code)
            
            # Worker 2: Generate main code (with aux_presc so it knows how to use aux)
            main_prompt = worker_main_formatter(batch_item, main_presc, aux_presc)
            main_code = self._generate_from_agent(
                self.worker_agents[1],
                [main_prompt],
                num_return_sequences=1,
                max_new_tokens=self.max_new_tokens_code,  # Code: shorter
                do_sample=False,
            )[0][0]
            main_completions.append(main_code)
        
        # Step 4: Compute rewards
        rewards = self._compute_rewards(aux_completions, main_completions, batch_item)
        self.env_step += len(rewards)
        
        # Step 5: Update public agent
        self.optimizer.zero_grad()
        loss = self._compute_grpo_loss(public_prompt, prescriptions, rewards)
        loss.backward()
        self.optimizer.step()
        
        # Metrics
        metrics = {
            "loss": float(loss.item()),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "min_reward": float(min(rewards)) if rewards else 0.0,
        }
        
        return float(loss.item()), metrics
    
    def train(self):
        """Main training loop."""
        if self.wandb_config and not self.wandb_initialized:
            self._init_wandb()
        
        device = torch.device("cuda")
        self.public_agent.to(device)
        for worker in self.worker_agents:
            worker.to(device)
        
        self.public_agent.train()
        for worker in self.worker_agents:
            worker.eval()
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        
        for epoch in range(int(self.args.num_train_epochs)):
            epoch_losses = []
            epoch_rewards = []
            
            for batch_idx, batch in enumerate(dataloader):
                batch_item = batch[0]
                
                # Periodic evaluation
                if (
                    self.args.eval_interval > 0 and
                    batch_idx % self.args.eval_interval == 0 and
                    self.eval_dataset is not None
                ):
                    self.evaluate(num_samples=self.args.eval_num_samples)
                
                loss, metrics = self._train_step(batch_item)
                epoch_losses.append(loss)
                epoch_rewards.append(metrics["mean_reward"])
                
                # Logging
                if (
                    self.args.logging_steps > 0 and
                    batch_idx % self.args.logging_steps == 0
                ):
                    log_dict = {
                        "train/loss": loss,
                        "train/mean_reward": metrics["mean_reward"],
                        "train/max_reward": metrics["max_reward"],
                    }
                    if self.wandb_initialized:
                        wandb.log(log_dict, step=self.env_step)
                    print(f"[Epoch {epoch+1}][Batch {batch_idx}] Loss: {loss:.4f}, "
                          f"Mean Reward: {metrics['mean_reward']:.4f}")
            
            # Epoch summary
            epoch_log = {
                "epoch/mean_loss": float(np.mean(epoch_losses)),
                "epoch/mean_reward": float(np.mean(epoch_rewards)),
            }
            if self.wandb_initialized:
                wandb.log(epoch_log, step=self.env_step)
    
    def evaluate(self, num_samples: int = 4) -> Dict[str, float]:
        """Evaluate the current model."""
        if self.eval_dataset is None:
            return {}
        
        eval_rewards = []
        
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        
        with torch.no_grad():
            for eval_idx, batch in enumerate(dataloader):
                if eval_idx >= num_samples:
                    break
                
                batch_item = batch[0]
                
                # Generate single prescription (greedy)
                public_prompt = public_agent_formatter(batch_item)
                prescriptions = self._generate_from_agent(
                    self.public_agent,
                    [public_prompt],
                    num_return_sequences=1,
                    max_new_tokens=self.max_new_tokens_prescription,  # Prescription
                    do_sample=False,
                )[0]
                
                if not prescriptions:
                    continue
                
                prescription = prescriptions[0]
                aux_presc, main_presc = parse_prescription(prescription)
                
                # Generate code from workers
                aux_prompt = worker_aux_formatter(batch_item, aux_presc)
                aux_code = self._generate_from_agent(
                    self.worker_agents[0],
                    [aux_prompt],
                    num_return_sequences=1,
                    max_new_tokens=self.max_new_tokens_code,  # Code
                    do_sample=False,
                )[0][0]
                
                main_prompt = worker_main_formatter(batch_item, main_presc, aux_presc)
                main_code = self._generate_from_agent(
                    self.worker_agents[1],
                    [main_prompt],
                    num_return_sequences=1,
                    max_new_tokens=self.max_new_tokens_code,  # Code
                    do_sample=False,
                )[0][0]
                
                rewards = self._compute_rewards([aux_code], [main_code], batch_item)
                if rewards:
                    eval_rewards.append(rewards[0])
        
        metrics = {
            "eval/mean_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            "eval/max_reward": float(max(eval_rewards)) if eval_rewards else 0.0,
        }
        
        if self.wandb_initialized:
            wandb.log(metrics, step=self.env_step)
        
        return metrics
    
    def save_model(self, output_dir: str):
        """Save the trained public agent model (or LoRA adapter if using LoRA)."""
        os.makedirs(output_dir, exist_ok=True)
        
        public_dir = os.path.join(output_dir, "public_agent")
        os.makedirs(public_dir, exist_ok=True)
        
        if self.use_lora:
            # Save only the LoRA adapter weights
            self.public_agent.save_pretrained(public_dir)
            print(f"LoRA adapter saved to: {public_dir}")
        else:
            # Save the full model
            self.public_agent.save_pretrained(public_dir)
            print(f"Full model saved to: {public_dir}")
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(public_dir)
        
        if self.wandb_initialized and wandb.run is not None:
            wandb.log({"final_model_saved": output_dir, "use_lora": self.use_lora})


# =============================================================================
# Reward Function
# =============================================================================

def prescription_reward_func(aux_completions, main_completions, batch_items=None):
    """
    Compute execution-based rewards for prescription-generated code.
    """
    test_cases = []
    entry_points = []
    prompts = []
    
    if batch_items:
        for item in batch_items:
            test_cases.append(item.get("test", ""))
            entry_points.append(item.get("entry_point", ""))
            prompts.append(item.get("prompt", ""))
    else:
        return [0.0] * len(aux_completions)
    
    return execution_reward_aux(
        aux_completions, main_completions, test_cases, entry_points, prompts
    )


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run Prescription-based MAGRPO training."""
    parser = argparse.ArgumentParser(
        description="Train Prescription-based MAGRPO"
    )
    add_config_args(parser)
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config(args.config)
    else:
        raise ValueError("Please provide a configuration file using --config")
    
    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)
    
    # Model configuration
    model_config = config.get_model_config()
    public_model_name = model_config.name
    
    # Worker model can be different - check config
    worker_model_name = config.get("model.worker_model", public_model_name)
    
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")
    output_base_dir = config.get("output.base_dir")
    
    # Infer dataset type if not specified
    if dataset_type is None:
        if "humaneval" in dataset_name.lower():
            dataset_type = "humaneval"
        elif "mbpp" in dataset_name.lower():
            dataset_type = "mbpp"
        else:
            raise ValueError(f"Could not infer dataset type from '{dataset_name}'")
    
    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")
    
    # Training config
    trainer_config = config.get_section("prescription_magrpo") if hasattr(config, "get_section") else {}
    if not trainer_config:
        trainer_config = config.get_section("magrpo") if hasattr(config, "get_section") else {}
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"prescription_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(config, "save"):
        config.save(os.path.join(output_dir, "config.yaml"))
    
    # Load datasets
    try:
        train_dataset = load_dataset(dataset_name, split=train_split)
        eval_dataset = load_dataset(dataset_name, split=eval_split)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {public_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        public_model_name, **model_config.tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models with Flash Attention 2 support
    model_kwargs = dict(model_config.model_kwargs)
    
    # Enable Flash Attention 2 if not explicitly disabled
    use_flash_attn = config.get("model.use_flash_attention_2", True)
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Flash Attention 2 enabled")
    
    print(f"\nLoading public agent from {public_model_name}...")
    public_agent = AutoModelForCausalLM.from_pretrained(
        public_model_name, **model_kwargs
    )
    
    print(f"\nLoading worker agents from {worker_model_name}...")
    worker_agents = [
        AutoModelForCausalLM.from_pretrained(
            worker_model_name, **model_kwargs
        )
        for _ in range(2)
    ]
    
    # Build training args
    training_args = MAGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=trainer_config.get("num_train_epochs", 20),
        per_device_train_batch_size=1,
        learning_rate=trainer_config.get("learning_rate", 5e-6),
        logging_steps=trainer_config.get("logging_steps", 50),
        save_steps=trainer_config.get("save_steps", 200),
        eval_interval=trainer_config.get("eval_interval", 16),
        eval_num_samples=trainer_config.get("eval_num_samples", 4),
        num_generations=trainer_config.get("num_generations", 4),
        max_new_tokens=trainer_config.get("max_new_tokens", 512),
        temperature=trainer_config.get("temperature", 0.6),
        top_p=trainer_config.get("top_p", 0.6),
    )
    
    # Reward processor
    reward_processor = None
    if config.get("reward_processor.enabled", False):
        scale = config.get("reward_processor.scale_factor", 1.0)
        reward_processor = RewardProcessors.scale(factor=scale)
    
    # W&B config
    wandb_section = config.get_section("wandb") if hasattr(config, "get_section") else {}
    wandb_config = {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_section.get("name", f"prescription-magrpo-{dataset_type}"),
        "dir": wandb_section.get("dir"),
        "tags": ["prescription-magrpo", dataset_type],
        "config_sections": {
            "dataset": config.get_section("dataset") if hasattr(config, "get_section") else {},
            "model": config.get_section("model") if hasattr(config, "get_section") else {},
            "trainer": trainer_config,
        },
    }
    
    # Get separate max_new_tokens for prescription and code
    max_new_tokens_prescription = trainer_config.get("max_new_tokens_prescription", 512)
    max_new_tokens_code = trainer_config.get("max_new_tokens_code", 256)
    
    # LoRA configuration for public agent
    use_lora = trainer_config.get("use_lora", False)
    lora_config = None
    if use_lora:
        lora_section = trainer_config.get("lora", {})
        lora_config = {
            "r": lora_section.get("r", 16),
            "lora_alpha": lora_section.get("lora_alpha", 32),
            "lora_dropout": lora_section.get("lora_dropout", 0.05),
            "target_modules": lora_section.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            "bias": lora_section.get("bias", "none"),
        }
        print(f"LoRA enabled with config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}")
    
    # Create trainer
    trainer = PrescriptionGRPOTrainer(
        public_agent=public_agent,
        worker_agents=worker_agents,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_func=prescription_reward_func,
        reward_processor=reward_processor,
        wandb_config=wandb_config,
        args=training_args,
        dataset_type=dataset_type,
        max_new_tokens_prescription=max_new_tokens_prescription,
        max_new_tokens_code=max_new_tokens_code,
        use_lora=use_lora,
        lora_config=lora_config,
    )
    
    # Train
    trainer.train()
    
    # Save model
    if config.get("output.save_final_model", False):
        save_path = config.get("output.save_path", os.path.join(output_dir, "final_model"))
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
