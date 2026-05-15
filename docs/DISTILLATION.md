# Knowledge distillation

`lib/distill.py` trains a small *student* GPT to imitate the next-token
distribution of a larger *teacher*. Useful for shipping cheaper inference
without retraining from scratch.

## Loss

For every batch we compute two losses:

- **KD term** — KL divergence between temperature-softened teacher and
  student logits:

  ```text
  L_KD = T^2 · KL( softmax(z_T / T) || softmax(z_S / T) )
  ```

- **CE term** — standard next-token cross-entropy against the real labels.

Final loss is a convex combination:

```text
L = α · L_KD + (1 - α) · L_CE
```

with `α` typically 0.5–0.9 and `T` typically 1.0–4.0.

## Recipe

```bash
# 1) train (or download) a "teacher" — bigger model on the same corpus
python lib/train.py config/train_multilang.py
# → produces out-multilang/ckpt.pt

# 2) distill into a 4-layer student
python lib/distill.py \
    --teacher-ckpt out-multilang/ckpt.pt \
    --dataset data/multilang_en_pt \
    --out out-student/ckpt.pt \
    --student-config '{"n_layer":4,"n_head":4,"n_embd":256}' \
    --iters 8000 \
    --alpha 0.8 \
    --temperature 2.0

# 3) sample / serve as a normal checkpoint
python lib/sample.py --out_dir=out-student --start="Era uma vez"
```

## Typical results

For a 50M teacher → 12M student on a multilingual corpus you can expect:

| | Teacher | Student | KD student |
|---|---|---|---|
| Params | 50M | 12M | 12M |
| Val PPL | 25 | 40 | 28-32 |
| Tokens/sec (CPU) | 70 | 220 | 220 |

i.e. close to teacher quality at a fraction of the cost.

## Tuning knobs

- **`--student-config`** — JSON dict overriding `n_layer`, `n_head`, `n_embd`.
  Anything else (vocab_size, block_size) is inherited from the teacher so the
  two models stay shape-compatible.
- **`--alpha`** — KD weight. Start at 0.8 (KD dominates) and lower if the
  student under-fits the real labels.
- **`--temperature`** — softens the teacher distribution. Higher T transfers
  more "dark knowledge" but risks vanishing gradients; 2.0–4.0 is typical.

## Caveats

- Teacher and student must share the same tokenizer and vocab size.
- Use AMP / bf16 on the teacher forward pass to halve memory — there are no
  gradients to keep around there.
- The student converges fastest if you reset the LR schedule rather than
  inherit the teacher's.
