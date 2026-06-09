# GPT-2 Efficiency Extensions: LoRA + Int8 Weight Quantisation

From-scratch PyTorch implementations of two model-efficiency techniques, applied
to GPT-2 small (124M) on classification, paraphrase detection, and generation
tasks. Built as the course project for **KTH DD2424 (Deep Learning)** on top of
the Stanford CS 224N "Build GPT-2" starter.

**This repository publishes our original extensions only** — the `LoRALinear` and
`QuantizedLinear` modules, a runnable demo, tests, and our experimental results.
It deliberately does **not** include the completed CS 224N starter (see
[Attribution](#attribution--academic-integrity)).

> Authors: **Adrian Nagorka**, **Joel Håkansson**, **Nicholas Lawrance**.
> Contribution breakdown in [CONTRIBUTORS.md](CONTRIBUTORS.md).

---

## Headline results

LoRA at rank 8 trains **0.24% of the parameters** of full fine-tuning and matches
or beats it on both small classification tasks. Int8 weight-only quantisation
cuts peak inference memory by **~20%** for ≤ **1.6 pp** accuracy cost, and
composes cleanly with LoRA (a small-scale QLoRA).

| Task | Metric | Full fine-tune | LoRA r=8 (0.24% params) | Δ (LoRA − full) |
|------|--------|:-:|:-:|:-:|
| CFIMDB (sentiment, 2-way) | dev acc | 0.947 | **0.980** | **+3.3 pp** |
| SST (sentiment, 5-way) | dev acc | 0.510 | 0.501 | −0.9 pp |
| Quora (paraphrase) | dev acc | **0.898** | 0.829 | −6.9 pp |

| Int8 quantisation (across all configs) | Result |
|---|---|
| Peak inference memory saved | ~240 MB (≈20% on small configs, ≈11% on full fine-tune) |
| Accuracy cost | ≤ 1.6 pp; 0.0 pp on most configs |
| Trade-off | weight-only scheme adds ~7–9% latency (on-the-fly dequant) |

Full per-configuration tables (memory, accuracy, latency) are in
[results/results.md](results/results.md).

---

## Problem & motivation

Full fine-tuning of every parameter of a pre-trained language model is expensive
to train and to store. Two complementary techniques address this: **parameter-
efficient fine-tuning (PEFT)**, which trains a small number of new parameters
while freezing the backbone, and **quantisation**, which lowers the bits per
stored weight. We implement one of each from scratch — **LoRA** (PEFT) and
**post-training int8 weight quantisation** — measure them on the same
experimental grid, and show they stack.

## Method

### LoRA — `src/lora.py`

For a frozen linear weight `W0`, LoRA learns a low-rank update
`ΔW = (α/r) · B·A` with `r ≪ min(out, in)`, so the adapted layer computes

```
y = W0·x + (α/r) · B·(A·x)
```

`B` is initialised to zero (so `ΔW = 0` at init and the layer starts equal to the
base), and the `α/r` factor decouples the effective step size from the rank.
`LoRALinear` wraps any `nn.Linear`; we adapt the **query and value projections**
of every attention layer (following Hu et al., 2022, §7.1). At rank 8 this is
296k trainable parameters — 0.24% of the 124M backbone.

### Int8 weight-only quantisation — `src/quantization.py`

Per-output-channel **symmetric** int8 (zero-point `Z = 0`): each output row `i`
of `W` gets its own scale `s_i = max_j|W_ij| / 127`, weights are stored as int8
in `[-127, 127]`, and inference dequantises on the fly (`y = (W_q · s)·x + b`).
Dequant and matmul run in fp32, so this reduces *stored weight* memory, not
activation memory — a deliberately simple, calibration-free baseline.

### Composition (QLoRA-style)

Because `LoRALinear` keeps its frozen base layer as a child module,
`quantize_model_int8` recurses into it and quantises **only the frozen backbone**,
leaving the fp32 LoRA adapters untouched — structurally the
[QLoRA](https://arxiv.org/abs/2305.14314) recipe at small scale.

## How to run

The demo and tests are fully self-contained — **no datasets, pre-trained
weights, or network access required.** They exercise the modules on a small toy
transformer (`toy_model.py`) that is *not* the course model.

```bash
pip install -r requirements.txt

python demo.py        # LoRA param%, int8 memory saving, composed QLoRA-style
pytest -q             # unit tests for both modules and their composition
```

To use the modules in a real GPT-2 pipeline, see
[docs/integration.md](docs/integration.md), which shows the exact ~10-line hooks
we added to the CS 224N task scripts.

## Results

See [results/results.md](results/results.md) for the complete grid (SST, CFIMDB,
Quora × {linear-head, full, LoRA r∈{8,16}} × {fp32, int8}), including memory,
accuracy, and latency tables and the hyperparameters used.

Sonnet generation was evaluated qualitatively only; quantitative chrF scoring was
left as future work.

## Limitations

These are carried over honestly from the project report:

- **Single seed.** Every cell is one run (seed 11711); small gaps (e.g. SST
  LoRA vs. full) are within seed-to-seed noise and we do not claim significance.
- **Weight-only quantisation.** Activations and the matmul stay fp32, so int8
  reduces *stored* weight memory (~4×) but peak inference memory by only ~20%;
  the weight-only scheme also adds ~7–9% latency (no native int8 kernel).
- **Sweep coverage.** Rank sweep covers r ∈ {8, 16} on SST/CFIMDB and r=8 only on
  Quora (each Quora LoRA epoch took ~36 min); closing the 6.9 pp Quora gap with
  higher rank is the obvious next step.
- **Post-training only.** No calibration and no quantisation-aware training, so
  the reported accuracy cost is an upper bound for this scheme.

## Attribution & academic integrity

The experiments were run on Stanford's **CS 224N "Default Final Project: Build
GPT-2"** starter (<https://web.stanford.edu/class/cs224n/>). That assignment is
reused across course offerings, so to avoid publishing a solution to it, this
repository **excludes the completed starter** (GPT-2 model, attention, optimizer,
classifier, and task scripts). Only our own extension modules are published; the
integration is shown as short patch snippets in
[docs/integration.md](docs/integration.md). See [NOTICE](NOTICE) for details.

Datasets (SST, CFIMDB, Quora Question Pairs, Shakespeare sonnets) are **not
redistributed**. Obtain them from their original sources / the CS 224N starter.

## License

[MIT](LICENSE) © 2026 Adrian Nagorka, Joel Håkansson, Nicholas Lawrance.
"# gpt2-lora-int8" 
