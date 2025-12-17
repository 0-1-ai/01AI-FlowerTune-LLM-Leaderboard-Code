# Code Evaluation (Qwen3-8B, Code challenge)

Evaluate the round-10 PEFT adapter (`peft_10`) on four benchmarks: HumanEval, MBPP, MultiPL-E (JS), MultiPL-E (C++) using the BigCode evaluation harness.

## Environment
1) Clone the official eval harness wrapper:
```bash
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/code ./flowertune-eval-code && rm -rf flower && cd flowertune-eval-code
```
2) Install deps (Python 3.11 recommended) and log in to Hugging Face:
```bash
pip install -r requirements.txt
huggingface-cli login
```
3) Install Node.js and g++ for MultiPL-E JS/C++:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
exec $SHELL
nvm install 20
sudo apt-get install -y g++
```
4) Fetch `main.py` from the BigCode harness:
```bash
git clone https://github.com/yan-gao-GY/bigcode-evaluation-harness.git && cd bigcode-evaluation-harness && mv main.py ../ && cd .. && rm -rf bigcode-evaluation-harness
```

## Run (single command example)
- Base model: `Qwen/Qwen3-8B`
- PEFT adapter: `./results/<timestamp>/peft_10`
- Batch: 4 (adjust as needed)
- Max length: 2048 (MBPP needs 2048; OK for others)
- Multi-GPU: pass `--max_memory_per_gpu=24GiB` (example) to enable `device_map=auto`

```bash
python main.py \
  --model=Qwen/Qwen3-8B \
  --peft_model=./results/<timestamp>/peft_10 \
  --max_length_generation=2048 \
  --batch_size=4 \
  --use_auth_token \
  --allow_code_execution \
  --save_generations \
  --save_references \
  --tasks=humaneval,mbpp,multiple-js,multiple-cpp \
  --metric_output_path=./evaluation_results_peft_10.json \
  --trust_remote_code \
  --max_memory_per_gpu=24GiB
```

## Outputs
- Generations: `generations_{task}.json`
- Metrics: `evaluation_results_{task}.json`

Submit pass@1 for all four tasks (HumanEval, MBPP, MultiPL-E JS, MultiPL-E C++) to the leaderboard.

## Notes on patched `main.py`
- 4-bit uses nf4 + double quant; compute dtype is bf16 when supported (else fp16).
- Device map/max_memory handling unified (auto when `max_memory_per_gpu` is set, otherwise per-process device_map).
- Tokenizer safety tweaks (pad/eos, WizardCoder BOS fix) and `trust_remote_code` support are retained.
