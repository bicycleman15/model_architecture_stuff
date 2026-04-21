# **************************************************
# Copyright (c) 2025, Mayank Mishra
#
# Vendored subpackage: pure-Python launch wrappers and triton kernels for
# M2RNN. Corresponds to `xma/layers/m2rnn/triton_implementation/` in
# open-lm-engine/accelerated-model-architectures. See ATTRIBUTION.md.
# **************************************************

from .backward import _m2rnn_backward_triton
from .forward import _m2rnn_forward_triton

__all__ = ["_m2rnn_forward_triton", "_m2rnn_backward_triton"]
