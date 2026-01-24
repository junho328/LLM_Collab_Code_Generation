# MAGRPO Baseline Progress

## Project Overview
- **Baseline**: MAGRPO (Multi-Agent Group Relative Policy Optimization)
- **Model**: Qwen/Qwen3-0.6B
- **Task**: Writing (Article - ArXiv Abstract Expansion)
- **Setup**: Single-turn only
- **Branch**: `koh-dev/MAGRPO-baseline`

---

## Quick Start Guide (New Server Setup)

### Step 1: Clone and Checkout Branch
```bash
git clone https://github.com/junho328/LLM_Collab_Writing.git
cd LLM_Collab_Writing
git checkout koh-dev/MAGRPO-baseline
git lfs install
```

### Step 2: Install Dependencies
```bash
pip install comlrl "transformers>=4.51.0" datasets torch accelerate wandb
```

### Step 3: Configure Accelerate (for multi-GPU)
```bash
accelerate config default --mixed_precision bf16
```

### Step 4: Set Environment Variables
```bash
export HF_TOKEN=""
export WANDB_API_KEY=""
export OPENAI_API_KEY=""
wandb login --relogin $WANDB_API_KEY
```

### Step 5: Run Training

**Option A: Single GPU (Recommended for stability)**
```bash
CUDA_VISIBLE_DEVICES=0 python train_magrpo.py --config configs/magrpo_arxiv_config.yaml
```

**Option B: Multi-GPU (4 GPUs) - May encounter OOM**
```bash
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 train_magrpo.py --config configs/magrpo_arxiv_config.yaml
```

**Option C: Multi-GPU with reduced memory (if OOM occurs)**

Edit `configs/magrpo_arxiv_config.yaml`:
- Change `num_generations: 4` to `num_generations: 2`
- Change `max_new_tokens: 256` to `max_new_tokens: 128`

---

## Shared Parameter Training

**Experiment Branch**: `koh-dev/shared-param` ([GitHub](https://github.com/junho328/LLM_Collab_Writing/tree/koh-dev/shared-param))

**CoMLRL Branch**: `koh-dev/shared` ([GitHub](https://github.com/junho328/CoMLRL/tree/koh-dev/shared))

Shared parameter training uses a single model instance for all agents, reducing memory by ~50%.

### Setup

**Step 1: Clone LLM_Collab_Writing**
```bash
git clone https://github.com/junho328/LLM_Collab_Writing.git
cd LLM_Collab_Writing
git checkout koh-dev/shared-param
```

**Step 2: Install CoMLRL with Shared Parameter Support**
```bash
# Option A: Install from forked branch (recommended)
pip install git+https://github.com/junho328/CoMLRL.git@koh-dev/shared

# Option B: Local editable install (for development)
cd /workspace
git clone https://github.com/junho328/CoMLRL.git
cd CoMLRL
git checkout koh-dev/shared
pip install -e .
```

**Step 3: Install other dependencies**
```bash
pip install "transformers>=4.51.0" datasets torch accelerate wandb
```

### Run Training
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python sharedparam_train_magrpo.py --config configs/magrpo_arxiv_config.yaml

# Multi-GPU
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 sharedparam_train_magrpo.py --config configs/magrpo_arxiv_config.yaml
```

### How It Works

The shared parameter support is implemented **natively in CoMLRL** (`junho328/CoMLRL` branch `koh-dev/shared`):

1. **Automatic Detection**: `MAGRPOTrainer` detects when all agents share the same model instance
2. **Single Optimizer**: Creates one optimizer instead of per-agent optimizers
3. **Gradient Accumulation**: 
   - First agent zeros gradients
   - All agents accumulate gradients (scaled by `1 / (num_samples * num_agents)`)
   - Last agent triggers optimizer step
4. **Efficient Saving**: Saves model once to `shared_model/` directory

**Console Output** (expected):
```
[SharedParam] Loading single shared model: Qwen/Qwen3-0.6B
[SharedParam] Model loaded once, shared by 2 agents
[SharedParams] Detected shared model across 2 agents
```

See full documentation: [CoMLRL Progress.md](https://github.com/junho328/CoMLRL/blob/koh-dev/shared/Progress.md)

---

## WandB Configuration
- **Project**: MA-GRPO
- **Entity**: namhokoh-korea-advanced-institute-of-science-and-technology
- **Dashboard**: https://wandb.ai/namhokoh-korea-advanced-institute-of-science-and-technology/MA-GRPO

---

## Configuration Summary

### Model Config
- Model: Qwen/Qwen3-0.6B (0.6B params, 28 layers, 32K context)
- Temperature: 0.7
- Top-p: 0.9 (original baseline)
- Top-k: null (original baseline)

### Dataset
- Name: OpenMLRL/arXiv_abstract
- Train split: train[:1100]
- Eval split: val[:1100]

### Training Parameters (Default)
- Num epochs: 2
- Batch size: 1 (per device)
- Learning rate: 5e-6
- Num generations: 4
- Max new tokens: 256

---

## Known Issues

### Issue 1: Multiple WandB Runs
**Fixed** in `train_magrpo.py` - only main process initializes WandB.

### Issue 2: CUDA OOM with Multi-GPU
Training crashed at step 31/1100 on 4x A100-40GB. Solutions:
1. Use single GPU mode (Option A)
2. Reduce `num_generations` to 2
3. Reduce `max_new_tokens` to 128

---

## Git Push Command
```bash
git add -A && git commit -m "message"
git push https://<YOUR_GITHUB_PAT>@github.com/junho328/LLM_Collab_Writing.git koh-dev/MAGRPO-baseline
```

---

## Pending Steps
1. Complete MAGRPO baseline training
2. Combine with agent chaining via prompting
3. Implement LLM inference of other agent's state

## Notes
- Writing experiments are strictly single-turn (num_turns=1)
- Two agents: Background (motivation) + Complementary (methodology)
- Qwen3-0.6B requires transformers>=4.51.0
- Model does NOT use thinking mode during training