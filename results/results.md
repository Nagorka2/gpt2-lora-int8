# Results

All numbers are transcribed from our DD2424 project report. Base model: GPT-2
small (124M) from HuggingFace, trained with our custom AdamW on a single
consumer CUDA GPU. Single run per cell (seed 11711). "# Trainable params" counts
only parameters with `requires_grad=True`. For each `--quantize` run, the fp32
and int8 columns come from the **same** script invocation (fp32 pass, then int8
pass on the same dev set), controlling the comparison by construction.

## Hyperparameters

LoRA uses α = 2r throughout. The smaller batch size on CFIMDB/Quora is dictated
by the starter's per-task config.

| Task | Mode | LR | Batch | Epochs | Dropout |
|------|------|----|------|------|------|
| SST | last-linear-layer | 1e-3 | 96 | 6 | 0.1 |
| SST | full-model | 1e-4 | 96 | 6 | 0.6 |
| SST | LoRA (r∈{8,16}) | 1e-3 | 96 | 6 | 0.3 |
| CFIMDB | last-linear-layer | 1e-3 | 8 | 6 | 0.1 |
| CFIMDB | full-model | 1e-4 | 8 | 6 | 0.6 |
| CFIMDB | LoRA (r∈{8,16}) | 1e-3 | 8 | 6 | 0.3 |
| Quora | full-model | 1e-5 | 8 | 5 | default |
| Quora | LoRA (r=8) | 1e-5 | 8 | 5 | default |

LoRA required learning rates 1–2 orders of magnitude larger than the full-fine-
tune optimum to converge on SST/CFIMDB.

## SST dev (5-way sentiment)

| Mode | # Trainable params | fp32 acc | fp32 mem (MB) | int8 acc | int8 mem (MB) |
|------|:-:|:-:|:-:|:-:|:-:|
| Linear head only | ~4k | 0.460 | 1194 | 0.443 | 958 |
| Full fine-tune | 125M | 0.510 | 2152 | 0.508 | 1917 |
| LoRA, r=8 | 296k (0.24%) | 0.501 | 1202 | 0.497 | 967 |
| LoRA, r=16 | 591k (0.48%) | 0.505 | 1230 | 0.509 | 986 |

## CFIMDB dev (binary sentiment)

| Mode | # Trainable params | fp32 acc | fp32 mem (MB) | int8 acc | int8 mem (MB) |
|------|:-:|:-:|:-:|:-:|:-:|
| Linear head only | ~2k | 0.845 | 1202 | 0.849 | 956 |
| Full fine-tune | 125M | 0.947 | 2160 | 0.947 | 1917 |
| **LoRA, r=8** | 296k (0.24%) | **0.980** | 1209 | **0.980** | 966 |
| LoRA, r=16 | 591k (0.48%) | 0.976 | 1238 | 0.971 | 984 |

LoRA r=8 exceeds full fine-tuning by **3.3 pp** here, attributed to implicit
regularisation on a small (1.7k-example) training set.

## Quora paraphrase detection (dev; 5 epochs, batch 8, lr 1e-5)

| Mode | # Trainable params | fp32 acc | fp32 mem (MB) | int8 acc | int8 mem (MB) |
|------|:-:|:-:|:-:|:-:|:-:|
| Full fine-tune | 125M | 0.898 | 1982 | 0.898 | 1744 |
| LoRA, r=8 | 296k (0.24%) | 0.829 | 1041 | 0.829 | 804 |

On the much larger 283k-example training set, LoRA r=8 leaves a **6.9 pp** gap to
full fine-tuning: expressive capacity, not regularisation, is the binding
constraint. A higher-rank sweep is expected to close it.

## Effect of int8 quantisation on accuracy

Changes are within ±1.6 pp across all configurations.

| Mode | Dataset | fp32 | int8 | Δ (pp) |
|------|---------|:-:|:-:|:-:|
| last-linear-layer | SST | 0.460 | 0.443 | −1.6 |
| last-linear-layer | CFIMDB | 0.845 | 0.849 | +0.4 |
| full-model | SST | 0.510 | 0.508 | −0.2 |
| full-model | CFIMDB | 0.947 | 0.947 | 0.0 |
| full-model | Quora | 0.898 | 0.898 | 0.0 |
| LoRA r=8 | SST | 0.501 | 0.497 | −0.5 |
| LoRA r=8 | CFIMDB | 0.980 | 0.980 | 0.0 |
| LoRA r=8 | Quora | 0.829 | 0.829 | 0.0 |
| LoRA r=16 | SST | 0.505 | 0.509 | +0.4 |
| LoRA r=16 | CFIMDB | 0.976 | 0.971 | −0.4 |

## Peak GPU memory during inference (int8 saves ~240 MB consistently)

| Mode | Dataset | fp32 (MB) | int8 (MB) | Reduction |
|------|---------|:-:|:-:|:-:|
| last-linear-layer | SST | 1194 | 958 | 19.7% |
| last-linear-layer | CFIMDB | 1202 | 956 | 20.5% |
| full-model | SST | 2152 | 1917 | 10.9% |
| full-model | CFIMDB | 2160 | 1917 | 11.2% |
| full-model | Quora | 1982 | 1744 | 12.0% |
| LoRA r=8 | SST | 1202 | 967 | 19.6% |
| LoRA r=8 | CFIMDB | 1209 | 966 | 20.1% |
| LoRA r=8 | Quora | 1041 | 804 | 22.8% |
| LoRA r=16 | SST | 1230 | 986 | 19.9% |
| LoRA r=16 | CFIMDB | 1238 | 984 | 20.5% |

Full fine-tuning shows a smaller *proportional* saving (~11%) because it starts
from a higher fp32 baseline (extra CUDA caching-allocator state persisting into
the test phase); the absolute saving (~240 MB) is essentially constant.

## Per-example inference latency

The weight-only scheme adds latency because weights are dequantised on the fly; a
native int8 matmul kernel would invert this.

| Mode | Dataset | fp32 (ms) | int8 (ms) | Slowdown |
|------|---------|:-:|:-:|:-:|
| last-linear-layer | SST | 1.66 | 1.79 | +7.8% |
| last-linear-layer | CFIMDB | 15.15 | 16.46 | +8.6% |
| full-model | SST | 1.89 | 2.04 | +7.9% |
| full-model | CFIMDB | 36.05 | 38.36 | +6.4% |
| full-model | Quora | 3.00 | 3.68 | +22.7% |
| LoRA r=8 | SST | 1.77 | 1.89 | +6.8% |
| LoRA r=8 | CFIMDB | 16.07 | 17.11 | +6.5% |
| LoRA r=8 | Quora | 3.46 | 4.26 | +23.1% |
| LoRA r=16 | SST | 1.76 | 1.90 | +8.0% |
| LoRA r=16 | CFIMDB | 15.87 | 17.25 | +8.7% |

## Sonnet generation

Fine-tuned GPT-2 small on 143 training sonnets (next-token prediction) and
generated continuations from the first three lines of each held-out sonnet
(temperature 1.2, top-p 0.9). Outputs show surface features of the domain
(archaic pronouns, inverted syntax, stanzaic line breaks) but do not reliably
produce iambic pentameter or honour the rhyme scheme, as expected for a 124M
model fine-tuned on 143 examples. Quantitative chrF scoring against the
references was left as future work.
