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
| Activations (gradient checkpointed)| ~3–4 GB      |
| Item embeddings + LSTM + projector | ~0.3 GB      |
| Optimizer state (AdamW, small)     | ~0.2 GB      |
| **Peak**                           | **~9–10 GB** |

If you run out of memory, reduce `model.max_session_len` and/or
`train.micro_batch_size` in `configs/games.yaml`.

---

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
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
