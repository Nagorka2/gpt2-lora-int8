# Contributors

This was a three-person course project for KTH DD2424. The breakdown below is
taken from the individual self-assessment section of our project report.

- **Adrian Nagorka** — LoRA extension and efficiency analysis
- **Nicholas Lawrence** — core GPT-2 implementation and sentiment classification
- **Joel Håkansson** — paraphrase detection, sonnet generation, leaderboard

## Adrian Nagorka — LoRA extension & efficiency analysis

- Implemented the **LoRA** formulation (Hu et al., 2022) from scratch: the
  `LoRALinear` module wrapping frozen `nn.Linear` layers, the `apply_lora_to_gpt`
  helper traversing the module tree, and the unified CLI interface.
  (Report §4.2.)
- Ran the **rank sweep** (r ∈ {8, 16}) across SST, CFIMDB, and Quora and analysed
  the parameter-efficiency vs. accuracy trade-off — identifying the Quora 6.9 pp
  gap as a capacity ceiling rather than a regularisation effect.
- Implemented **post-training int8 weight quantisation** and composed it with
  LoRA, mirroring QLoRA at smaller scale. (Tables 2–7, §5.5.)

## Nicholas Lawrence — core GPT-2 & sentiment classification

- Implemented **causal self-attention**, the GPT-2 transformer block,
  token/positional embeddings, the weight-tied output head, and the **AdamW**
  optimiser from scratch following the CS 224N framework. (Report §1.1.)
- Identified and resolved a critical detail: the attention **padding mask** is
  distinct from the upper-triangular causal mask and must be combined with it,
  otherwise padding tokens corrupt self-attention on shorter sequences.
  (Report §1.1.)

## Joel Håkansson — paraphrase detection, sonnet generation & submissions

- Ran the controlled **full fine-tune vs. LoRA r=8** comparison on Quora (283k
  examples, 5 epochs) and analysed how training-set scale shifts LoRA's role from
  regulariser to capacity bottleneck — in direct contrast to CFIMDB.
  (Report §5.3, Table 7.)
- Led **sonnet generation**, providing a qualitative case study in data-starved
  fine-tuning: a 124M model fine-tuned on only 143 examples produces plausible
  surface features (archaic pronouns, stanzaic format) while failing on deeper
  structure. (Report §5.4.)

## Note on this public repository

This repo publishes only the LoRA and int8-quantisation extensions (Adrian's
work, with shared experimental design). The core GPT-2 implementation led by
Nicholas and the task scripts used by Joel are part of the CS 224N starter
solution and are intentionally **not** published here (see `NOTICE`); their
contributions are credited above and the integration is documented in
`docs/integration.md`.

