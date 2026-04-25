# DUIP with Qwen3.5-2B on Amazon Reviews 2023 (Games)

A from-scratch PyTorch implementation of the paper
**"Enhancing User Intent for Recommendation Systems via Large Language Models"**
([Xu et al., 2025, arXiv:2501.10871](https://arxiv.org/abs/2501.10871)) — *DUIP*
(Dynamic User Intent Prediction).

The original paper combined an LSTM with **GPT-2**. Here we replace GPT-2 with the
much larger **`Qwen/Qwen3.5-2B`**, kept fully **frozen** so that only the LSTM,
the soft-prompt projector, and the item embedding table are trained. Thanks to
this, the whole pipeline trains on a single 12 GB consumer GPU
(e.g. RTX 5070 Ti) using BF16 + gradient checkpointing.

The dataset is the **Amazon Reviews 2023, Video Games** category from
[`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

---

## 1. Architecture

```
session items x_1..x_{t-1}
        │
        ▼
   item embedding lookup  (warm-started from Qwen title embeddings)
        │
        ▼
       LSTM  ─────►  hidden state h_t
                          │
                          ▼
              soft-prompt projector f(·)
                          │
                          ▼
       K learnable pseudo-token embeddings  ┐
                                            │ spliced into the embedding stream
hard-prompt text "User has interacted ..."  ┘
                          │
                          ▼
          Qwen3.5-2B (frozen, BF16, grad-checkpointed)
                          │
                          ▼
   score every candidate item title by Σ log P(token | prompt)
                          │
                          ▼
          HR@1, HR@5, NDCG@1, NDCG@5
```

Equation references in the source code map directly to the paper
(Eqs. 1–7, §3).

---

## 2. Hardware / memory budget

Targeted at **RTX 5070 Ti, 12 GB**:

| Component                          | Approx. VRAM |
|------------------------------------|--------------|
| Qwen3.5-2B in BF16 (frozen)        | ~5 GB        |
| Activations (no grad-checkpoint)   | ~3–4 GB      |
| Item embeddings + LSTM + projector | ~0.3 GB      |
| Optimizer state (AdamW, small)     | ~0.2 GB      |
| **Peak**                           | **~9–10 GB** |

The forward pass batches **all** candidates of a step (`B*C` sequences)
into a single LLM call. To bound peak activation memory the chunk size is
exposed as `model.cand_chunk_size` (default 32). If you OOM:

- Lower `model.cand_chunk_size` (e.g. 16 or 8).
- Lower `train.micro_batch_size`.
- Re-enable `model.gradient_checkpointing: true`.
- Lower `data.max_session_len`.

---

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Log in once so training streams to your dashboard. Skip this and set
# `logging.wandb.mode: offline` (or `disabled`) in configs/games.yaml if
# you don't want online tracking.
wandb login
```

> The first call will pull `Qwen/Qwen3.5-2B` (~5 GB) from the Hugging Face Hub.
> If your network needs a token, run `huggingface-cli login` first.

> **Transformers version**: Qwen3.5 is a new architecture (Gated DeltaNet
> + sparse MoE + vision encoder). If `requirements.txt`'s `transformers>=4.46`
> is too old to recognize it, install from main:
> ```bash
> pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@main"
> ```
> Our `_load_qwen_lm` helper transparently handles both the pure-causal-LM
> and the multimodal `ImageTextToText` checkpoint variants.

> **Flash-Attention-2 (optional)**: `flash-attn` is *optional*. If
> installed, the LLM is loaded with `attn_implementation="flash_attention_2"`;
> otherwise the code falls back to PyTorch SDPA (which itself uses FA2
> kernels when available on your GPU), and finally to the eager attention
> kernel. Override the chain via `model.attn_implementation` in
> [`configs/games.yaml`](configs/games.yaml) (`flash_attention_2` |
> `sdpa` | `eager` | `auto`).

---

## 4. End-to-end run

```bash
# 1. Download + sessionize the Amazon Reviews 2023 Games subset
python -m scripts.prepare_data --config configs/games.yaml

# 2. Train (LSTM + soft-prompt projector + item embeddings; Qwen frozen)
python -m scripts.run_training --config configs/games.yaml

# 3. Evaluate on the test split
python -m scripts.run_eval --config configs/games.yaml \
    --checkpoint checkpoints/games/best.pt
```

Or run everything:

```bash
bash scripts/run_all.sh
```

Outputs:

- `data/processed/{train,val,test}.jsonl`  – sessionized splits
- `data/processed/items.json`              – item id ↔ title catalog
- `checkpoints/games/best.pt`              – best validation checkpoint
- `results/games.json`                     – HR@1, HR@5, NDCG@1, NDCG@5
- `logs/games/<run_name>/run.log`          – plain-text per-run log
- `logs/games/<run_name>/metrics.jsonl`    – structured per-step metrics
- W&B project `duip` (when `logging.wandb.enabled: true`) – live charts
  for `train/loss`, `train/loss_ema`, `train/lr`, `train/grad_norm`,
  `train/samples_per_sec`, `train/tokens_per_sec`, `train/gpu_mem_*`,
  `train/in_batch_HR@K`, `train/in_batch_NDCG@K`, plus `val/*` and
  `test/*` rollups.

---

## 5. Files

```
project1/
├── configs/games.yaml          # all hyperparameters
├── src/
│   ├── data/
│   │   ├── download.py         # HF datasets -> raw cache
│   │   ├── preprocess.py       # sessionize daily, 80/10/10 split
│   │   └── dataset.py          # SessionDataset + collator
│   ├── models/
│   │   ├── lstm_encoder.py     # LSTM + item embedding
│   │   ├── soft_prompt.py      # h_t -> K x d_llm pseudo-tokens
│   │   └── duip.py             # frozen Qwen + soft-prompt splicer
│   ├── train.py                # InfoNCE training loop
│   ├── evaluate.py             # HR@K / NDCG@K eval
│   └── utils.py                # metrics, seeding, logging
└── scripts/
    ├── prepare_data.py
    ├── run_training.py
    ├── run_eval.py
    └── run_all.sh
```

---

## 6. Notes / assumptions

- **Sessionization** follows the paper §4.1 ("daily" interactions per user).
- **Evaluation protocol** is the standard 1-positive + 99-sampled-negatives
  Hit-Rate / NDCG. The paper does not state otherwise; the negative seed is
  fixed in `configs/games.yaml` for reproducibility.
- The Qwen3.5-2B model is multimodal; we explicitly load only its
  text/language-model component (vision tower is dropped after load).

---

## 7. Performance & logging knobs

All under [`configs/games.yaml`](configs/games.yaml):

| Block | Key | Notes |
|-------|-----|-------|
| `model` | `attn_implementation` | `flash_attention_2` (default) → `sdpa` → `eager` fallback chain. |
| `model` | `cand_chunk_size` | Cap on `B*C` sequences per LLM call. Lower if you OOM. |
| `model` | `gradient_checkpointing` | Default `false`. Re-enable for tighter VRAM at the cost of compute. |
| `train` | `micro_batch_size`, `grad_accum_steps` | Effective batch = product. Bump `micro_batch_size` first to feed the GPU. |
| `train` | `num_workers`, `persistent_workers`, `prefetch_factor` | Keep the GPU fed from the data side. |
| `eval`  | `micro_batch_size`, `num_workers` | Eval uses `inference_mode`, so VRAM headroom is larger than training. |
| `eval`  | `cand_chunk_size` | Overrides `model.cand_chunk_size` during eval only. Eval is forward-only so a larger value (e.g. 32–64) is usually safe and noticeably faster than the training default. |
| `runtime` | `tf32`, `cudnn_benchmark`, `matmul_precision` | One-line throughput wins on Ampere+ GPUs. |
| `logging` | `wandb.{enabled,project,entity,run_name,mode}` | `mode: disabled` skips W&B entirely; `offline` queues runs locally. |
| `logging` | `log_every_steps`, `jsonl` | Detailed metrics (loss / EMA / LR / grad-norm / throughput / VRAM / in-batch HR & NDCG) are emitted every N optimizer steps to console + JSONL + W&B. |
