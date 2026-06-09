"""LoRA (Low-Rank Adaptation) for linear layers, implemented from scratch.

LoRA [Hu et al., ICLR 2022, arXiv:2106.09685] freezes a pre-trained weight
matrix ``W0`` and learns a low-rank additive update ``dW = (alpha / r) * B @ A``
with ``r << min(out, in)``. Because the update is linear in the input, the
adapted forward pass is

    y = W0 @ x + (alpha / r) * B @ (A @ x)

which adds only ``r * (in + out)`` trainable parameters per adapted layer while
keeping the original weights frozen.

In our DD2424 project, applying this to the query and value projections of
GPT-2 small (124M params) trained only 0.24% of the parameters at rank 8, yet
matched or exceeded full fine-tuning on the SST and CFIMDB classification tasks.

This module has no dependency on any particular model: ``LoRALinear`` wraps any
``nn.Linear``. Two appliers are provided:

* ``apply_lora`` — generic; wraps ``nn.Linear`` submodules selected by attribute
  name. Works on any model (used by ``demo.py`` and the tests).
* ``apply_lora_to_gpt`` — the convenience wrapper we used in the course project,
  which targets the attention projections of the CS 224N GPT-2 model.
"""

from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear``, freeze it, and add a trainable low-rank update.

    The base layer's parameters are frozen (``requires_grad = False``); only the
    low-rank factors ``A`` and ``B`` are trained. Following Hu et al., ``B`` is
    initialised to zero so that ``dW = 0`` at initialisation and the wrapped
    layer initially reproduces the base layer's output exactly. The output is
    scaled by ``alpha / r`` to decouple the effective step size from the choice
    of rank.

    Args:
        base_linear: The pre-trained ``nn.Linear`` to adapt. Kept as a child
            module (named ``base``) so it participates in ``state_dict()`` and
            ``.to(device)`` but is invisible to the optimiser once frozen.
        r: LoRA rank. Number of columns of ``B`` / rows of ``A``.
        alpha: LoRA scaling numerator. The update is scaled by ``alpha / r``.

    Shapes:
        A: ``[r, in_features]``     B: ``[out_features, r]``
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}.")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.r = r
        self.alpha = alpha
        # alpha / r decouples the learning rate from the rank (Hu et al., 2022).
        self.scaling = alpha / r

        # A ~ N(0, sigma^2) with small sigma; B = 0  =>  dW = 0 at init.
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.normal_(self.A, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        base_out = self.base(x)
        # (x @ A^T) @ B^T : [..., r] -> [..., out_features], without
        # materialising the full dW = B @ A.
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3g}"


def apply_lora(
    model: nn.Module,
    target_modules=("query", "value"),
    r: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """Replace selected ``nn.Linear`` submodules with ``LoRALinear`` wrappers.

    Walks the module tree and, for every direct child that is an ``nn.Linear``
    whose attribute name is in ``target_modules``, swaps it in place for a
    ``LoRALinear``. This is model-agnostic: it works on any module that exposes
    its projections as named ``nn.Linear`` attributes.

    Note: the caller is responsible for freezing the rest of the model (e.g.
    ``for p in model.parameters(): p.requires_grad = False``) before calling
    this if a "LoRA-only" trainable set is desired. ``LoRALinear`` itself always
    freezes the base layer it wraps.

    Args:
        model: The module to adapt (modified in place; also returned).
        target_modules: Attribute names of ``nn.Linear`` layers to adapt.
        r: LoRA rank.
        alpha: LoRA scaling numerator.

    Returns:
        The same ``model`` instance, with matching layers wrapped.
    """
    target = set(target_modules)
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in target and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
    return model


def apply_lora_to_gpt(gpt_model, r: int = 8, alpha: int = 16,
                      target=("query", "value")):
    """Adapt the attention projections of the CS 224N GPT-2 model with LoRA.

    This is the convenience wrapper used in the original course project. It
    assumes the starter model's structure (``gpt_model.gpt_layers``, each layer
    exposing ``self_attention.{query,key,value}`` and ``attention_dense``). It is
    kept here for reference and reproducibility; for arbitrary models prefer the
    generic ``apply_lora`` above.

    Args:
        gpt_model: The CS 224N GPT-2 model instance.
        r: LoRA rank.
        alpha: LoRA scaling numerator.
        target: Which projections to adapt. We used ``("query", "value")``,
            following Hu et al. (2022), §7.1, as the best parameter-budget
            trade-off.

    Returns:
        The same ``gpt_model`` instance, with the selected projections wrapped.
    """
    for layer in gpt_model.gpt_layers:
        attn = layer.self_attention
        if "query" in target:
            attn.query = LoRALinear(attn.query, r=r, alpha=alpha)
        if "key" in target:
            attn.key = LoRALinear(attn.key, r=r, alpha=alpha)
        if "value" in target:
            attn.value = LoRALinear(attn.value, r=r, alpha=alpha)
        if "output" in target:
            layer.attention_dense = LoRALinear(layer.attention_dense, r=r, alpha=alpha)
    return gpt_model


def trainable_parameter_summary(model: nn.Module):
    """Return ``(trainable, total, fraction)`` parameter counts for ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fraction = trainable / total if total else 0.0
    return trainable, total, fraction
