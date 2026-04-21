# Attribution

The files in this directory that are derived from upstream come from:

- **Project**: [`open-lm-engine/accelerated-model-architectures`](https://github.com/open-lm-engine/accelerated-model-architectures)
- **Author**: Mayank Mishra
- **Upstream copyright**: `Copyright (c) 2025, Mayank Mishra`

## Derived files

| local path                       | upstream path                                                |
| -------------------------------- | ------------------------------------------------------------ |
| `op.py`                          | `xma/layers/m2rnn/op.py` (pure-PyTorch path) + `xma/layers/m2rnn/utils.py` (`_get_num_heads`) |
| `layer.py`                       | `xma/layers/m2rnn/layer.py`                                  |
| `triton/forward.py`              | `xma/layers/m2rnn/triton_implementation/forward.py`          |
| `triton/backward.py`             | `xma/layers/m2rnn/triton_implementation/backward.py`         |
| `triton/kernels.py`              | bundle of `xma/triton_utils/matmul.py` and the primitives actually used in `xma/triton_utils/activations.py` (`clamp`, `tanh`, `tanh_backward`, `sigmoid_backward`) |
| `triton/math_utils.py`           | pure-Python helpers from `xma/math.py` (`ceil_divide`, `get_powers_of_2`, `get_next_power_of_2`) |

The upstream copyright header is preserved in each vendored file.

## Why vendor

Upstream requires Python >= 3.12 and torch >= 2.10 and pulls in a wide
dispatch chain via `xma.accelerator`, `xma.custom_op`, and
`xma.torch_utils`. Our environment is Python 3.10 / torch 2.5 / triton
3.1, so we kept two paths side by side:

- A **pure-PyTorch reference** (derived from the `forward_backward_torch`
  method of `_M2RNN`). PyTorch's native autograd is sufficient — no
  custom backward needed. Useful for CPU and for debugging.
- The **vendored triton kernels**, invoked via our own
  `torch.autograd.Function` (`_M2RNNTritonFunction` in `op.py`). This
  avoids vendoring `xma/custom_op.py`, `xma/accelerator.py`, and
  `xma/inductor.py` — the kernels are already standalone
  `@triton.jit` / `@triton.autotune` functions.

The `backend` knob on `M2RNNLayer` (and the `backend: triton|torch`
field in `state_tracking/config/model/rnn.yaml`) selects between them.

## Edits

### Pure-torch path (`op.py`, `layer.py`)

- Removed CUDA/CustomOp dispatch and the `KernelBackend` enum.
- Removed the variable-length (`cu_seqlens`, `max_seqlen`) batching
  path; the state-tracking task uses fixed-length sequences.
- Inlined tiny helpers that originally lived in `xma.math` /
  `xma.utils` / `xma.torch_utils` (`divide_if_divisible`,
  `clip_gradients`, `tanh`). `clip_gradients` here is a plain `clamp`;
  upstream uses a straight-through variant, but for fixed-length S_n
  training the difference is negligible.

### Triton path (`triton/forward.py`, `triton/backward.py`)

- Removed the `@xma_op(mutates_args=...)` decorator on the top-level
  Python launch functions. `@xma_op` is upstream's wrapper over
  `torch.library.custom_op` for Accelerator/ROCm/TPU dispatch and for
  `torch.compile` interop; we call the functions directly from
  `_M2RNNTritonFunction` in `op.py`.
- Rewrote the upstream relative imports (`....custom_op`, `....math`,
  `....triton_utils`, `..utils`) to point at the local vendored copies
  under `state_tracking/models/rnn/triton/` and at the parent `op.py`.
- The `@triton.jit` and `@triton.autotune` kernel bodies are unchanged.

### Triton primitives (`triton/kernels.py`, `triton/math_utils.py`)

- `kernels.py` bundles `matmul` from `xma/triton_utils/matmul.py` and
  the subset of `xma/triton_utils/activations.py` actually used by the
  forward/backward (`tanh`, `tanh_backward`, `sigmoid_backward`,
  `clamp`). No other activations, no `compute_p_norm`.
- `math_utils.py` keeps only the pure-Python helpers from
  `xma/math.py` — `ceil_divide`, `check_power_of_2`, `get_powers_of_2`,
  `get_next_power_of_2`.

## Files not derived from upstream

- `model.py` — Transformer-style LM wrapper (embed + RMSNorm + M2RNN +
  SwiGLU + LM head). Written for the state-tracking baseline.
- `__init__.py` (both top-level and `triton/`) — package exports.
- `ATTRIBUTION.md` — this file.
