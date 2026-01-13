# Qwen3 8B - Code (Submission v1)

Federated instruction tuning for the **Code** challenge using `Qwen/Qwen3-8B` on `flwrlabs/code-alpaca-20k`. Round-10 PEFT adapter (`peft_10`) is used for leaderboard evaluation.

## Project Structure
```
.
├── mmfl/                         # Source code for ClientApp, ServerApp, and Strategy
├── flowertune-eval-code/         # Evaluation scripts and instructions
├── pyproject.toml                # Project configuration and dependencies
└── README.md                     # This file
```

## Methodology

### Changes from Baseline
- Base model: `mistralai/Mistral-7B-v0.3` → `Qwen/Qwen3-8B` with `trust_remote_code=true`.
- Rounds: 200 → 10; save-every-round kept at 5.
- LoRA/DoRA: `r=8`, `alpha=16`, target `q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj` (baseline 32/64, default targets, no DoRA).
- Optim/precision: `paged_adamw_8bit`, `bf16=true`, `tf32=true`; 4-bit quantization (nf4) retained.
- Batch/steps: per-device batch 2, accum 4 (effective 8; baseline 16/1); 3 epochs, max_steps 10 per round.
- Runtime stack: torch 2.4.0 / peft 0.14.0 / transformers 4.51.0 (baseline torch 2.9.1, peft 0.6.2).
- Strategy: custom communication tracker retained; FlexLoRA disabled.

### Model Configuration
- Base: `Qwen/Qwen3-8B`
- Quantization: 4-bit (bnb, nf4)
- PEFT: LoRA + DoRA, r=8, alpha=16, targets q/k/v/o/gate/down/up
- Seq length: 512
- Fractions: fraction_fit=0.2, fraction_evaluate=0

### Training Configuration (from K8s env & pyproject)
- Num rounds: 10
- Per-device train batch: 2
- Grad accumulation: 4
- LR: max 5e-5 / min 5e-6, scheduler constant
- Epochs: 3, max_steps: 10
- Save every: 5 rounds
- BF16/TF32 enabled

> `[tool.flwr.app.config.static]` and `options.num-supernodes` remain at the Code challenge fixed values.

## Evaluation

See `evaluation/README.md` for the exact command (bigcode harness). Benchmarks: HumanEval, MBPP, MultiPL-E (JS), MultiPL-E (C++). Evaluation run used fp32 (training was 4-bit).

| Benchmark | pass@1 |
|---|---:|
| HumanEval | 0.7073 |
| MBPP | 0.5620 |
| MultiPL-E (JS) | 0.6832 |
| MultiPL-E (C++) | 0.6584 |
| **Average** | **0.6527** |

**Communication budget: 3549.28 MB**

> Run uses `peft_10`(batch=4, max_len=2048, temp=0.2, top_p=0.95, max_mem=24GiB).
> Patched `main.py` supports multi-GPU via `--max_memory_per_gpu`; this run used two GPUs (e.g., 4090+3090) with `device_map=auto`.

## Reproducing Evaluation

See `evaluation/README.md` for a single multi-task command and options (`--trust_remote_code`, `--max_memory_per_gpu`) matching our run settings.

## Checkpoints

- Round 10 PEFT adapter: [Google Drive link](https://drive.google.com/file/d/1oJyTu4VbRHtv74EYX6jxPX4B-JVk2Ahz/view?usp=sharing)
