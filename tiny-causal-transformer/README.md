# tiny-causal-transformer

A small GPT-style decoder that learns to reverse three digits.

## Objective

Every example is a reverse-copy sequence of length 9:

```text
[BOS, a, b, c, SEP, c, b, a, EOS]
```

`a`, `b`, and `c` are digits `0–9`. Special tokens sit above that range: `BOS=10`, `SEP=11`, `EOS=12` (vocab size 13). There are `10³ = 1,000` such sequences. After a seeded shuffle they split **800 / 200** train / validation.

The model sees the prompt `[BOS, a, b, c, SEP]` and must emit the reversed digits plus `EOS`. Causal language modeling is used throughout: inputs are `sequences[:, :-1]` (length 8) and targets are `sequences[:, 1:]`.

## Setup, train, test, generate

Python 3.10+ is required. Use the **repo-root** virtualenv:

```bash
# from the ai-fluency repo root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cd tiny-causal-transformer

python train.py          # writes checkpoint.pt and metrics.json
pytest -q                # 48 tests
```

Greedy decode from a saved checkpoint:

```python
import torch
from data import BOS, SEP
from generate import greedy_generate
from train import load_checkpoint

model, _ = load_checkpoint("checkpoint.pt")
prompt = torch.tensor([BOS, 1, 2, 3, SEP])
print(greedy_generate(model, prompt).tolist())
# [10, 1, 2, 3, 11, 3, 2, 1, 12]
```

Architecture: 2 pre-norm decoder blocks, `d_model=64`, 4 heads, `d_ff=256`, context length 8, dropout 0.1, AdamW (`lr=3e-3`, `weight_decay=1e-4`), max 150 epochs, early stopping patience 15.

## QKV shape progression

With batch `B`, time `T=8`, channels `C=64`, heads `H=4`, head dim `D=C/H=16`:

```text
x                  [B, T, C]          # 64-d token vectors
qkv_projection     [B, T, 3C]         # fused Q, K, V
chunk → q, k, v    [B, T, C] each
split_heads        [B, H, T, D]       # 4 heads of 16
scores = Q Kᵀ / √D [B, H, T, T]       # attention over time
probs @ V          [B, H, T, D]
merge_heads        [B, T, C]
```

`[B, H, T, T]` is one `T×T` attention map per head per batch item. The causal mask broadcasts as `[1, 1, T, T]`.

## Why divide by `sqrt(head_dim)`

`Q Kᵀ` is a sum of `D` products. If Q and K entries have variance ~1, the scores have variance ~`D`. Softmax then saturates: one position gets almost all the mass, gradients vanish. Dividing by `√D` (`√16 = 4`) keeps the logits in a range where softmax stays informative. That is the scaling in `compute_attention_scores`.

## Why mask before softmax

The upper triangle of the score matrix is filled with `-inf` **before** softmax. Softmax of `-inf` is 0, so future tokens get no weight, and the remaining (past + self) weights still sum to 1.

Masking after softmax would zero the future but leave the rest unnormalized (they would no longer sum to 1). Those rows would be a broken distribution and would not match the causal LM objective.

## Causal mask vs loss mask

| | causal mask | loss mask |
|---|---|---|
| Lives on | attention scores `[B, H, T, T]` | next-token targets `[B, T]` |
| Shape here | `[1, 1, 8, 8]`, lower-triangular `bool` buffer | `[N, 8]`, `False False False False True True True True` |
| Job | token `t` may not attend to `t+1, …` | which target positions enter the loss |

The causal mask is required even on prompt positions: the model still runs a full causal forward pass. The loss mask is only about **which predictions we train**.

## Prompt tokens stay visible; prompt targets do not

Targets are `[a, b, c, SEP, c, b, a, EOS]`. Loss is applied only to the last four (`c, b, a, EOS`).

The prompt tokens remain in the input, so when the model predicts the first reversed digit it has already seen `BOS a b c SEP`. We do not train it to predict `a`, `b`, `c`, or `SEP` — those are given. Training on them would reward copying the prompt instead of reversing after `SEP`.

## Teacher forcing vs autoregressive generation

**Teacher forcing** (training and `evaluate_teacher_forced`): every step’s input is the **gold** prefix. The model predicts all 8 next tokens in one forward pass. That is fast and gives `teacher_forced` metrics.

**Autoregressive generation** (`greedy_generate`): the model sees only the prompt, then appends `argmax` of the **last** time step, one token at a time, until `EOS` or 4 new tokens. Errors can compound. `generation` metrics score that loop against `sequence[5:]`.

Both are 100% on this tiny reverse task after training, but they measure different things. Teacher forcing can look solved while greedy decode still fails if decoding uses the wrong time index.

## Pre-norm and residuals

Each `DecoderBlock` is pre-norm:

```text
x = x + attention(LayerNorm(x))
x = x + feedforward(LayerNorm(x))
```

**Residual.** The block output is an update added to `x`, so the original stream is still there if the sublayer is noisy. Gradients also have a skip path to earlier layers.

**Pre-norm.** LayerNorm runs *before* attention / MLP, which keeps residual-stream scale stable as blocks stack. This 2-layer model would train without it; the same recipe is what larger GPT-style models use.

## Why the best checkpoint can beat the final epoch

`fit` tracks the lowest validation loss (patience 15, min improvement `1e-6`) and reloads those weights at the end. Training can continue after the minimum: later epochs may overfit the 800 training triples or jitter from dropout. `best_epoch` was **145** of 150; the saved `checkpoint.pt` is that snapshot, not epoch 150.

## Final metrics

From `metrics.json` (seed 42, ~9–11 s on CPU):

| | value |
|---|---|
| Train / val / total | 800 / 200 / 1000 |
| Best epoch / epochs ran | 145 / 150 |
| Best val loss | `1.43e-5` |
| Teacher-forced token / exact acc | 1.0 / 1.0 |
| Generation token / exact acc | 1.0 / 1.0 |
| Parameters | 102,272 |
| Reloaded logits match | true |

## Debugging narrative

**Symptom.** `python train.py` crashed immediately with `ImportError: cannot import name 'VOCABULARY' from 'data'`.

**Hypothesis.** Either `data.py` never defined the vocab map, or `train.py` imported a name that had been renamed (`VOCAB_SIZE` vs `VOCABULARY`).

**Diagnostic.** `grep VOCABULARY tiny-causal-transformer` showed the name only in `train.py` (`from data import VOCABULARY` and the checkpoint payload). `data.py` had `BOS` / `SEP` / `EOS` / `VOCAB_SIZE` only.

**Cause.** The checkpoint spec asked to store a vocabulary mapping, but that dict was never added next to the token constants. Training never got as far as `save_checkpoint`.

**Fix.** Define `VOCABULARY` in `data.py` as digits `"0"`–`"9"` plus `"BOS"`, `"SEP"`, `"EOS"`, and import it once with the other data helpers. After that, `train.py` writes `checkpoint.pt` whose reload matches logits, and `pytest -q` stays green.
