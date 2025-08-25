# Experiments

This directory contains training scripts and configurations for paper _"LLM Collaboration with Multi-Agent Reinforcement Learning"_.

## Directory

```
experiments/
├── train_grpo.py           # Single-agent GRPO training
├── train_magrpo.py         # Multi-agent MAGRPO training
├── train_mt_magrpo.py      # Multi-turn MAGRPO training
├── config_loader.py        # Configuration utility
└── configs/               # YAML configuration files
    ├── grpo_*.yaml        # Single-agent configs
    ├── magrpo_*.yaml      # Multi-agent configs
    └── magrpo_mt_*.yaml   # Multi-turn configs
```

## Training Scripts

### 1. **train_grpo.py** - Single-Agent Training
Trains a single model using GRPO (Group Relative Policy Optimization) on various datasets.

### 2. **train_magrpo.py** - Multi-Agent Training
Trains multiple collaborative agents using MAGRPO on various datasets.

### 3. **train_mt_magrpo.py** - Multi-Turn Multi-Agent Training
Trains multiple agents with multi-turn interactions and expert feedback.

## Supported Datasets

All scripts support the following datasets:
- **HumanEval**: Code generation benchmark
- **CoopHumanEval**: Cooperative code generation
- **ArXiv**: Scientific abstract expansion
- **TLDR**: Reddit post summarization

## Configuration Files

### Naming Convention
- `grpo_*` - Single-agent configurations
- `magrpo_*` - Multi-agent configurations  
- `magrpo_mt_*` - Multi-turn multi-agent configurations

### Available Configurations

#### Single-Agent (GRPO)
- `grpo_he_config.yaml` - HumanEval single-agent
- `grpo_che_config.yaml` - CoopHumanEval single-agent
- `grpo_arxiv_config.yaml` - ArXiv single-agent
- `grpo_tldr_config.yaml` - TLDR single-agent

#### Multi-Agent (MAGRPO)
- `magrpo_he_config.yaml` - HumanEval multi-agent
- `magrpo_che_config.yaml` - CoopHumanEval multi-agent
- `magrpo_arxiv_config.yaml` - ArXiv multi-agent
- `magrpo_tldr_config.yaml` - TLDR multi-agent

#### Multi-Turn (MT-MAGRPO)
- `magrpo_mt_he_config.yaml` - Multi-turn HumanEval
- `magrpo_mt_che_config.yaml` - Multi-turn CoopHumanEval

## Usage

### Basic Training

```bash
# Single-agent HumanEval
python experiments/train_grpo.py --config experiments/configs/grpo_he_config.yaml

# Multi-agent CoopHumanEval
python experiments/train_magrpo.py --config experiments/configs/magrpo_che_config.yaml

# Multi-turn HumanEval
python experiments/train_mt_magrpo.py --config experiments/configs/mt_magrpo_he_config.yaml
```

### With Parameter Overrides

You can override any configuration parameter using the `--override` flag:

```bash
# Change model
python experiments/train_magrpo.py --config experiments/configs/magrpo_he_config.yaml \
    --override model_name=bigcode/starcoder2-3b

# Modify training parameters
python experiments/train_grpo.py --config experiments/configs/grpo_che_config.yaml \
    --override grpo.num_train_epochs=20 grpo.learning_rate=5e-6

# Change dataset splits
python experiments/train_magrpo.py --config experiments/configs/magrpo_tldr_config.yaml \
    --override dataset.train_split="train[:1000]" dataset.eval_split="test[:100]"

# Multiple overrides for multi-turn training
python experiments/train_mt_magrpo.py --config experiments/configs/mt_magrpo_che_config.yaml \
    --override mt_magrpo.num_turns=2 mt_magrpo.turn_gradient_weights=[1.5,0.5]
```

### Legacy Command-Line Arguments

For backward compatibility, you can also use direct command-line arguments:

```bash
# Override model name
python experiments/train_magrpo.py --config experiments/configs/magrpo_he_config.yaml \
    --model_name Qwen/Qwen2.5-Coder-7B

# For multi-turn, override specific parameters
python experiments/train_mt_magrpo.py --config experiments/configs/mt_magrpo_he_config.yaml \
    --num_epochs 10 --turn_gradient_weights 1.2 0.8
```
