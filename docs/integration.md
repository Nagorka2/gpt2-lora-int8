# Integrating the modules into a GPT-2 pipeline

This page shows how `src/lora.py` and `src/quantization.py` plugged into the
CS 224N "Build GPT-2" task scripts. We show the **hooks only** — short snippets,
not the completed starter — so the integration is reproducible without
republishing the assignment solution. To run end-to-end you need your own copy of
the CS 224N starter.

The modules themselves are model-agnostic: `LoRALinear` wraps any `nn.Linear` and
`quantize_model_int8` walks any `nn.Module`. The whole integration is roughly ten
lines per task script plus CLI flags.

## 1. LoRA on the attention query/value projections

In each task model's constructor, freeze the backbone and wrap the attention
projections. The starter GPT-2 exposes per-layer `self_attention.{query,value}`,
so `apply_lora_to_gpt` targets those directly:

```python
from src.lora import apply_lora_to_gpt

# inside the task model's __init__, when LoRA mode is selected:
if config.fine_tune_mode == "lora":
    for p in self.gpt.parameters():
        p.requires_grad = False
    self.gpt = apply_lora_to_gpt(
        self.gpt, r=config.lora_r, alpha=config.lora_alpha,
        target=("query", "value"),
    )
```

For any other architecture, use the generic applier instead:

```python
from src.lora import apply_lora

for p in model.parameters():
    p.requires_grad = False
apply_lora(model, target_modules=("query", "value"), r=8, alpha=16)
```

## 2. Post-training int8 quantisation

Applied *after* training, inside each task's eval/`test()` path, gated by a flag:

```python
from src.quantization import quantize_model_int8

measure_inference(dev_dataloader, model, device, label="fp32")
if args.quantize:
    model = quantize_model_int8(model)
    model = model.to(device)
    measure_inference(dev_dataloader, model, device, label="int8")
```

Because a LoRA-adapted model holds its frozen linear as a child named `base`,
`quantize_model_int8` recurses into it and quantises only the backbone, leaving
the fp32 LoRA adapters intact — the QLoRA structure.

## 3. CLI flags

```python
parser.add_argument("--fine-tune-mode",
                    choices=("last-linear-layer", "full-model", "lora"),
                    default="last-linear-layer")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--quantize", action="store_true",
                    help="Apply int8 weight quantisation before evaluation.")
```

## 4. Measuring memory and latency

The two-pass measurement that produced the memory/latency tables resets the CUDA
peak-memory counter between passes so the fp32 and int8 numbers are directly
comparable:

```python
import time, torch

def measure_inference(dataloader, model, device, label):
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    acc, *_ = evaluate(dataloader, model, device)   # task-specific eval
    if on_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    if on_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[{label}] acc {acc:.4f} | peak {peak_mb:.1f} MB | {elapsed:.2f}s")
    return acc, elapsed
```
