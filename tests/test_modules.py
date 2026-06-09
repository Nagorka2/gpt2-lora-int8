"""Unit tests for the LoRA and int8-quantisation modules.

Run:  pytest -q
These tests use small synthetic layers / a toy model only; no datasets, no
pre-trained weights, no downloads.
"""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lora import LoRALinear, apply_lora, trainable_parameter_summary
from src.quantization import QuantizedLinear, quantize_model_int8
from toy_model import ToyTransformer


# ----------------------------- LoRA ---------------------------------------

def test_lora_zero_init_matches_base():
    """With B = 0 at init, LoRALinear reproduces the base layer exactly."""
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_freezes_base_only_factors_train():
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}
    assert not lora.base.weight.requires_grad
    assert not lora.base.bias.requires_grad


def test_lora_scaling_value():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=4, alpha=16)
    assert lora.scaling == 16 / 4


def test_lora_update_changes_output_after_training_step():
    """A gradient step on B makes the output differ from the base."""
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    target = torch.randn(5, 24)
    opt = torch.optim.SGD([lora.A, lora.B], lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        loss = ((lora(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert not torch.allclose(lora(x), base(x), atol=1e-4)


def test_apply_lora_targets_named_linears_only():
    model = ToyTransformer(n_layer=2)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model, target_modules=("query", "value"), r=8, alpha=16)

    for layer in model.layers:
        assert isinstance(layer.self_attention.query, LoRALinear)
        assert isinstance(layer.self_attention.value, LoRALinear)
        # key / out were not targeted
        assert isinstance(layer.self_attention.key, nn.Linear)
        assert not isinstance(layer.self_attention.key, LoRALinear)


def test_trainable_fraction_is_small():
    model = ToyTransformer(n_layer=3)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model, ("query", "value"), r=8, alpha=16)
    trainable, total, frac = trainable_parameter_summary(model)
    assert 0 < frac < 0.05  # only the low-rank factors train


# ------------------------- Quantisation -----------------------------------

def test_quant_dequant_error_bounded():
    """Per-row symmetric int8 should reconstruct W with bounded error."""
    torch.manual_seed(0)
    base = nn.Linear(64, 128)
    q = QuantizedLinear(base)
    W = base.weight.data
    W_hat = q.weight_q.float() * q.scale.unsqueeze(1)
    # Max error per row is at most half a step = scale / 2.
    per_row_tol = (q.scale / 2 + 1e-6).unsqueeze(1)
    assert torch.all((W - W_hat).abs() <= per_row_tol)


def test_quant_buffers_not_parameters():
    base = nn.Linear(32, 48)
    q = QuantizedLinear(base)
    assert q.weight_q.dtype == torch.int8
    assert len(list(q.parameters())) == 0          # nothing trainable
    names = dict(q.named_buffers())
    assert "weight_q" in names and "scale" in names and "bias" in names


def test_quant_forward_close_to_fp32():
    torch.manual_seed(0)
    base = nn.Linear(64, 64)
    x = torch.randn(8, 64)
    ref = base(x)
    q = QuantizedLinear(base)
    rel = (q(x) - ref).norm() / ref.norm()
    assert rel < 0.05


def test_quantize_model_replaces_all_linears():
    model = ToyTransformer(n_layer=2)
    quantize_model_int8(model)
    assert not any(
        isinstance(m, nn.Linear) and not isinstance(m, QuantizedLinear)
        for m in model.modules()
    )
    assert any(isinstance(m, QuantizedLinear) for m in model.modules())


# ---------------------- Composition (QLoRA-style) -------------------------

def test_lora_then_quantize_keeps_adapters_fp32():
    model = ToyTransformer(n_layer=2)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model, ("query", "value"), r=8, alpha=16)
    quantize_model_int8(model)

    # The frozen base inside each LoRALinear is now quantised...
    for layer in model.layers:
        assert isinstance(layer.self_attention.query.base, QuantizedLinear)
    # ...while the LoRA factors stay fp32 trainable parameters.
    lora_factors = [p for n, p in model.named_parameters()
                    if n.endswith(".A") or n.endswith(".B")]
    assert len(lora_factors) > 0
    assert all(p.dtype == torch.float32 and p.requires_grad for p in lora_factors)


def test_composed_forward_runs():
    model = ToyTransformer(n_layer=2)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model, ("query", "value"), r=8, alpha=16)
    quantize_model_int8(model)
    out = model(torch.randint(0, 256, (2, 16)))
    assert out.shape == (2, 16, 256)
    assert torch.isfinite(out).all()
