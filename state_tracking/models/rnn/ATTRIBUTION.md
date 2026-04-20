# Attribution

The files `op.py` and `layer.py` in this directory are derived from:

- **Project**: [`open-lm-engine/accelerated-model-architectures`](https://github.com/open-lm-engine/accelerated-model-architectures)
- **Path**: `xma/layers/m2rnn/` (files `op.py`, `layer.py`, `utils.py`)
- **Author**: Mayank Mishra
- **Upstream copyright**: `Copyright (c) 2025, Mayank Mishra`

## Why vendor

Upstream requires Python >= 3.12 and torch >= 2.10 and pulls in a triton
kernel chain via `xma.accelerator`, `xma.custom_op`, and `xma.torch_utils`.
Our environment is Python 3.10 / torch 2.5, so we kept only the
self-contained **pure-PyTorch reference path** (the
`forward_backward_torch` method of `_M2RNN`) and rewrote it as a simple
differentiable function. PyTorch's native autograd is sufficient - no
custom backward is needed.

## Edits

- Removed triton / CUDA / CustomOp dispatch and the `KernelBackend` enum.
- Removed the variable-length (`cu_seqlens`, `max_seqlen`) batching path;
  the state-tracking task uses fixed-length sequences.
- Inlined tiny helpers that originally lived in `xma.math` / `xma.utils` /
  `xma.torch_utils` (`divide_if_divisible`, `clip_gradients`, `tanh`).
  `clip_gradients` here is a plain `clamp`; upstream uses a
  straight-through variant, but for fixed-length S_n training the
  difference is negligible.

## Files not derived from upstream

- `model.py` - Transformer-style LM wrapper (embed + RMSNorm + M2RNN +
  SwiGLU + LM head). Written for the state-tracking baseline.
- `__init__.py` - package exports.
- `ATTRIBUTION.md` - this file.
