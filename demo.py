"""End-to-end demo of the LoRA and int8-quantisation modules.

Runs on a small self-contained toy transformer (see ``toy_model.py``) so it
needs no datasets, no pre-trained weights, and no network access. It shows:

  1. LoRA on the query/value projections -> a tiny trainable-parameter footprint.
  2. Int8 weight-only quantisation -> smaller stored weight memory.
  3. The two composed (QLoRA-style) -> quantised backbone, fp32 adapters.

Run:  python demo.py
"""

from __future__ import annotations

import torch

from src.lora import apply_lora, trainable_parameter_summary
from src.quantization import quantize_model_int8, QuantizedLinear
from toy_model import ToyTransformer


def stored_weight_bytes(model: torch.nn.Module) -> int:
    """Sum the bytes of weight tensors that dominate stored size."""
    total = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            total += m.weight_q.numel() * m.weight_q.element_size()
            total += m.scale.numel() * m.scale.element_size()
        elif isinstance(m, torch.nn.Linear):
            total += m.weight.numel() * m.weight.element_size()
    return total


def main() -> None:
    torch.manual_seed(11711)
    x = torch.randint(0, 256, (4, 32))

    # --- 1. LoRA -----------------------------------------------------------
    model = ToyTransformer()
    base_out = model(x)

    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model, target_modules=("query", "value"), r=8, alpha=16)

    trainable, total, frac = trainable_parameter_summary(model)
    print("LoRA (r=8) on query/value projections")
    print(f"  trainable params : {trainable:,}")
    print(f"  total params     : {total:,}")
    print(f"  trainable share  : {100 * frac:.2f}%")

    # B is initialised to zero, so the adapted model reproduces the base output.
    with torch.no_grad():
        lora_out = model(x)
    max_drift = (lora_out - base_out).abs().max().item()
    print(f"  output drift at init (should be ~0): {max_drift:.2e}")

    # --- 2. Int8 quantisation ---------------------------------------------
    fp32_model = ToyTransformer().eval()
    bytes_fp32 = stored_weight_bytes(fp32_model)
    with torch.no_grad():
        ref = fp32_model(x)

    quantize_model_int8(fp32_model)  # in place
    bytes_int8 = stored_weight_bytes(fp32_model)
    with torch.no_grad():
        q_out = fp32_model(x)

    rel_err = (q_out - ref).norm() / ref.norm()
    print("\nInt8 weight-only quantisation")
    print(f"  stored linear weight bytes : {bytes_fp32:,} -> {bytes_int8:,}")
    print(f"  reduction on quantised tensors : {100 * (1 - bytes_int8 / bytes_fp32):.1f}%")
    print(f"  relative output error : {rel_err:.2e}")

    # --- 3. Compose: LoRA + int8 (QLoRA-style) ----------------------------
    qlora = ToyTransformer()
    for p in qlora.parameters():
        p.requires_grad = False
    apply_lora(qlora, ("query", "value"), r=8, alpha=16)
    quantize_model_int8(qlora)

    n_quant = sum(isinstance(m, QuantizedLinear) for m in qlora.modules())
    lora_params_fp32 = all(
        p.dtype == torch.float32
        for n, p in qlora.named_parameters()
        if n.endswith(".A") or n.endswith(".B")
    )
    print("\nComposed LoRA + int8 (QLoRA-style)")
    print(f"  quantised linear layers : {n_quant}")
    print(f"  LoRA adapters remain fp32 : {lora_params_fp32}")


if __name__ == "__main__":
    main()
