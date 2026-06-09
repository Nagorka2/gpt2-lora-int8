"""Post-training int8 weight-only quantisation, implemented from scratch.

We use per-output-channel symmetric int8 quantisation with no zero-point
(``Z = 0``), following the integer-arithmetic scheme of Jacob et al. (CVPR 2018,
arXiv:1712.05877) and the HuggingFace Optimum quantisation concept guide.

For a trained linear weight ``W`` of shape ``[out, in]``, each output row ``i``
gets its own scale

    s_i = max_j |W_ij| / 127

and the quantised weight is ``W_q = round(W / s)`` clamped to ``[-127, 127]`` and
stored as ``int8``. The value ``-128`` is dropped for symmetry. Inference
dequantises on the fly:

    y = (W_q * s) @ x^T + b

Both the dequantisation and the matmul run in fp32, so this scheme reduces
*stored* weight memory (~4x on the quantised tensors) but not activation memory
or arithmetic precision. In our DD2424 project this lowered peak inference GPU
memory by ~20% at a cost of <= 1.6 pp accuracy across all tested configurations.

Per-output-channel granularity gives each row its own scale, which substantially
reduces clipping error on layers whose weight magnitudes vary across rows.

Composes with LoRA: because ``LoRALinear`` holds its frozen base layer as a child
module named ``base``, ``quantize_model_int8`` recurses into it and quantises
only the frozen backbone, leaving the fp32 LoRA factors ``A``/``B`` untouched —
structurally the QLoRA recipe (quantised backbone, full-precision adapters).
"""

from __future__ import annotations

import torch
from torch import nn


class QuantizedLinear(nn.Module):
    """A drop-in replacement for a trained ``nn.Linear`` with int8 weights.

    Stores the weight as ``int8`` with one fp32 scale per output channel
    (symmetric, zero-point = 0). The quantised weight, the per-row scale, and the
    (unchanged) bias are registered as buffers — not parameters — so they move
    with ``.to(device)`` and appear in ``state_dict()`` but are invisible to the
    optimiser.

    Args:
        base_linear: The trained ``nn.Linear`` to quantise. Its weights are read
            once at construction; the original module is not retained.
    """

    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        W = base_linear.weight.data  # [out, in], fp32
        out_features, in_features = W.shape

        # Per-output-channel symmetric quantisation: one scale per row, no
        # zero-point. Range [-127, 127] (we skip -128 for symmetry).
        max_abs = W.abs().amax(dim=1, keepdim=True)   # [out, 1]
        scale = (max_abs / 127.0).clamp(min=1e-8)     # avoid div-by-zero
        W_q = torch.round(W / scale).clamp(-127, 127).to(torch.int8)

        # Buffers, not parameters: no gradients, but tracked by state_dict()/.to().
        self.register_buffer("weight_q", W_q)             # int8  [out, in]
        self.register_buffer("scale", scale.squeeze(1))   # fp32  [out]
        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.data.clone())
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise on the fly: W_fp32 ~= W_q.float() * scale (broadcast per row).
        W_dequant = self.weight_q.float() * self.scale.unsqueeze(1)
        out = x @ W_dequant.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, dtype=int8"


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Recursively replace every ``nn.Linear`` in ``model`` with ``QuantizedLinear``.

    Modifies ``model`` in place (and returns it). When applied to a LoRA-adapted
    model, the recursion descends into each ``LoRALinear``'s frozen ``base``
    child and quantises it, while the LoRA factors ``A``/``B`` (plain parameters,
    not child modules) remain in fp32.

    Args:
        model: The module to quantise in place.

    Returns:
        The same ``model`` instance, with linear layers quantised.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinear(module))
        else:
            quantize_model_int8(module)  # recurse
    return model
