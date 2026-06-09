# GPT-2 Efficiency Extensions: LoRA + Int8 Weight Quantisation

From-scratch PyTorch implementations of two model-efficiency techniques, applied
to GPT-2 small (124M) on classification, paraphrase detection, and generation
tasks. This started as our course project for KTH DD2424 (Deep Learning), built
on top of the Stanford CS 224N "Build GPT-2" starter.

This repository publishes our original extensions only: the `LoRALinear` and
`QuantizedLinear` modules, a runnable demo, tests, and our experimental results.
It does not include the completed CS 224N starter (see
[Attribution](#attribution-and-academic-integrity) for why).

Authors: Adrian Nagorka, Joel Håkansson, Nicholas Lawrence. The per-author
contribution breakdown is in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Headline results

At rank 8, LoRA trains 0.24% of the parameters of full fine-tuning and still
matches or beats it on both small classification tasks. Int8 weight-only
quantisation cuts peak inference memory by roughly 20% for at most 1.6 pp of
accuracy, and the two techniques compose into a small-scale QLoRA.

| Task | Metric | Full fine-tune | LoRA r=8 (0.24% params) | Δ (LoRA vs full) |
|------|--------|:-:|:-:|:-:|
| CFIMDB (sentiment, 2-way) | dev acc | 0.947 | 0.980 | +3.3 pp |
| SST (sentiment, 5-way) | dev acc | 0.510 | 0.501 | -0.9 pp |
| Quora (paraphrase) | dev acc | 0.898 | 0.829 | -6.9 pp |

| Int8 quantisation (across all configs) | Result |
|---|---|
| Peak inference memory saved | ~240 MB (about 20% on small configs, 11% on full fine-tune) |
| Accuracy cost | at most 1.6 pp; 0.0 pp on most configs |
| Trade-off | the weight-only scheme adds about 7 to 9% latency from on-the-fly dequant |

The complete per-configuration tables (memory, accuracy, latency) are in
[results/results.md](results/results.md).

## Problem and motivation

Fine-tuning every parameter of a pre-trained language model is expensive to train
and to store. Two complementary techniques help. Parameter-efficient fine-tuning
(PEFT) trains a small number of new parameters while keeping the backbone frozen.
Quantisation lowers the number of bits per stored weight. We implement one of
each from scratch, LoRA for PEFT and post-training int8 weight quantisation,
measure them on the same experimental grid, and show that they stack.

## Method

### LoRA (`src/lora.py`)

For a frozen linear weight `W0`, LoRA learns a low-rank update
`ΔW = (α/r) · B·A` with `r` much smaller than `min(out, in)`, so the adapted
layer computes

```
y = W0·x + (α/r) · B·(A·x)
```

`B` starts at zero, so `ΔW` is zero at initialisation and the layer begins
equal to the base. The `α/r` factor keeps the effective step size from depending
on the rank. `LoRALinear` wraps any `nn.Linear`. We adapt the query and value
projections of every attention layer, following Hu et al. (2022), §7.1. At rank 8
that comes to 296k trainable parameters, or 0.24% of the 124M backbone.

### Int8 weight-only quantisation (`src/quantization.py`)

Per-output-channel symmetric int8 with no zero-point (`Z = 0`). Each output row
`i` of `W` gets its own scale `s_i = max_j|W_ij| / 127`, weights are stored as
int8 in `[-127, 127]`, and inference dequantises on the fly
(`y = (W_q · s)·x + b`). Both the dequant and the matmul run in fp32, so this
reduces stored weight memory rather than activation memory. It is a deliberately
simple, calibration-free baseline.

### Composition (QLoRA-style)

Because `LoRALinear` keeps its frozen base layer as a child module,
`quantize_model_int8` recurses into it and quantises only the frozen backbone,
leaving the fp32 LoRA adapters in place. That is the same structure as
[QLoRA](https://arxiv.org/abs/2305.14314), just at a smaller scale.

## How to run

The demo and tests are self-contained. They need no datasets, pre-trained
weights, or network access, and run on a small toy transformer (`toy_model.py`)
that is not the course model.

```bash
pip install -r requirements.txt

python demo.py        # LoRA param%, int8 memory saving, composed QLoRA-style
pytest -q             # unit tests for both modules and their composition
```

To wire the modules into a real GPT-2 pipeline, see
[docs/integration.md](docs/integration.md), which shows the roughly ten-line
hooks we added to the CS 224N task scripts.

## Results

See [results/results.md](results/results.md) for the full grid: SST, CFIMDB, and
Quora across the linear-head, full, and LoRA (r of 8 and 16) modes, each with and
without int8, plus the memory, accuracy, and latency tables and the
hyperparameters used.

Sonnet generation was evaluated qualitatively only; quantitative chrF scoring was
left as future work.

## Limitations

These carry over honestly from the project report.

Single seed. Every cell is one run (seed 11711), so small gaps such as SST LoRA
versus full are within seed-to-seed noise and we do not claim significance.

Weight-only quantisation. Activations and the matmul stay in fp32, so int8 shrinks
the stored weights by about 4x but cuts peak inference memory by only around 20%.
The same scheme also adds roughly 7 to 9% latency, since there is no native int8
kernel.

Sweep coverage. The rank sweep covers r of 8 and 16 on SST and CFIMDB, and r=8
only on Quora (each Quora LoRA epoch took about 36 minutes). Closing the 6.9 pp
Quora gap with a higher rank is the obvious next step.

Post-training only. We do no calibration and no quantisation-aware training, so
the reported accuracy cost is an upper bound for this scheme.

## Attribution and academic integrity

The experiments ran on Stanford's CS 224N "Default Final Project: Build GPT-2"
starter (<https://web.stanford.edu/class/cs224n/>). That assignment is reused
across course offerings, so to avoid publishing a solution to it, this repository
leaves out the completed starter (the GPT-2 model, attention, optimizer,
classifier, and task scripts). Only our own extension modules are published, and
the integration is shown as short patch snippets in
[docs/integration.md](docs/integration.md). See [NOTICE](NOTICE) for details.

Datasets (SST, CFIMDB, Quora Question Pairs, Shakespeare sonnets) are not
redistributed. Obtain them from their original sources or the CS 224N starter.

## License

[MIT](LICENSE), 2026 Adrian Nagorka, Joel Håkansson, Nicholas Lawrence.
