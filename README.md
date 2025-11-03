# LLM Collaboration - Code Generation

Training scripts and configs for _"LLM Collaboration with Multi‑Agent Reinforcement Learning"_.

## Benchmarks

- MBPP: 427 problems on split `sanitized`
- HumanEval: 164 problems on split `test`
- CoopHumanEval: 82 problems on split `test`

## Training Scripts

```bash
python LLM_Collaboration_with_MARL/train_grpo.py \
  --config LLM_Collaboration_with_MARL/configs/grpo_he_config.yaml

python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_che_config.yaml
```

You can always override any configuration parameter using `--override`:

```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_he_config.yaml \
  --override model.name='bigcode/starcoder2-3b' magrpo.num_turns=1
```

## Settings

### Joint Action

`magrpo.joint_mode` determines how to combine each agent’s $G$ generations into joint actions at each turn. Two modes are supported: 'align' (default), which pairs the $g$‑th generation of every agent to form $G$ joint actions per node; and 'cross', which forms the Cartesian product within a node, yielding $G^N$ joint actions per node ($N$ agents). Total leaf joint trajectories after $T$ turns (no early termination): align → $G^T$; cross → $G^{N\cdot T}$. 

Aligned is faster in wall‑time (fewer sibling evaluations per node), while cross is more sample‑efficient (better value estimation) without extra VRAM because it reuses the same G generations per agent and only crosses them within the node. We never cross across different nodes/prompts; this preserves causal state consistency (actions are conditioned on the same prompts), keeps siblings comparable for the baseline/advantage, maintains correct credit assignment (log‑probs matched to rewards from the same state), and remains computationally tractable.

### Advantage

Advantages are used to optimize the agents policies, which use a mean baseline without any standard‑deviation normalization to make training unbiased (see [Dr. GRPO](https://arxiv.org/pdf/2503.20783)). We do not apply importance sampling ratios either, since our training is in an on-policy manner (hence also no need for epsilon clipping).

### Number of Samples

`magrpo.num_turns` is the number of turns in training and evaluation, and `magrpo.num_generations` is the number of samples per generation. Leaf (total samples at current turn) counts grow with $T$: `aligned` → $G^T$; `cross` → $G^{N\cdot T}$. At each node, the sibling set (competing joint actions under the same prompt/context/turn) has size $G$ for `aligned`, and $G^N$ for `cross`. The policy‑gradient baseline is the mean return over these siblings at that node.

### Termination

`magrpo.termination_threshold` is used to incentivize agents to find high‑reward solutions quickly instead of expanding the full Monte Carlo tree. At each node (branch, turn), we compute the mean immediate reward across that node’s sibling joint actions; if the mean exceeds the threshold, that branch stops expanding at this turn and the trainer backpropagates from the truncated subtree. Other branches continue.

### History

Agents always receive their own full history on later turns: all of their previous prompts and all of their previous responses, plus the new external prompt/context for the turn. No cross‑agent history is inserted. The behavior is fixed to full history and is not configurable.

### External Modes

`external.mode` controls how environment feedback shapes next‑turn prompts. Supported modes:

- 'level_feedback' (default): attaches static/dynamic diagnostics (syntax, tests, aux usage) and instructs code‑only revision.
- 'expert_edits': uses an external expert LLM to propose compact edits, then asks for a code‑only revision.
- 'plain': history‑only follow‑up with a concise "revise, code‑only" instruction (no diagnostics).

Modes 'level_passed' and 'passed' have been removed.

Specific setting for 'level_feedback' is `external.sandbox_slice`, which controls how many eval tests to include in the feedback. By default, sandbox executes only the first assert (sandbox_slice=1). Use all eval tests by setting `external.sandbox_slice` to 0, None, or 'all'. Negative values use the last asserts. `external.sandbox_slice` affects analysis-based mode 'level_feedback' (not used by 'expert_edits' or 'plain').

Specific settings for 'expert_edits' is `external.expert_edits_model`, which controls which LLM to use for proposing edits. By default, it uses DeepSeek-Coder. You can also change it to Claude, GPT, and other models, once you have keys/tokens in your environment.

### Output

`output.save_model` is set to 'false' by default because of the huge storage required by multiple LLMs. `output.verbose` is used for debug printing on cluster if set to be true, but it is default to be false and you can only see a tqdm bar that shows the training progress.
