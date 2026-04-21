# **************************************************
# Copyright (c) 2025, Mayank Mishra
#
# Vendored from open-lm-engine/accelerated-model-architectures:
#   xma/triton_utils/matmul.py
#   xma/triton_utils/activations.py   (only the primitives actually used by
#                                       m2rnn's forward/backward)
#
# Edits: none beyond bundling these triton primitives in a single file so
# the m2rnn kernels can import them via a single relative import.
# **************************************************

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
@triton.jit
def matmul(A, B, C, output_dtype: tl.constexpr):
    if A.shape[0] == 1:
        x = tl.sum(A.T * B, axis=0, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif A.shape[1] == 1:
        x = A * B
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif B.shape[1] == 1:
        x = tl.sum(A * B.T, axis=1, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif C is None:
        if output_dtype == tl.bfloat16:
            x = tl.dot(A, B, out_dtype=tl.float32).to(output_dtype)
        else:
            x = tl.dot(A, B, out_dtype=output_dtype)
    elif C.shape[0] == 1 or C.shape[1] == 1:
        x = tl.dot(A, B, out_dtype=tl.float32)
        x += C
        x = x.to(output_dtype)
    else:
        x = tl.dot(A, B, C.to(tl.float32), out_dtype=tl.float32).to(output_dtype)

    return x


# ---------------------------------------------------------------------------
# activations / misc element-wise
# ---------------------------------------------------------------------------
@triton.jit
def clamp(x, min_value, max_value):
    dtype = x.dtype

    x = max(min_value, x)
    x = min(max_value, x)
    x = x.to(dtype)

    return x


@triton.jit
def tanh(x, output_dtype: tl.constexpr = None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )
    x = x.to(output_dtype)

    return x


@triton.jit
def tanh_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = 1 - y * y
    y = y.to(dtype)

    return y


@triton.jit
def sigmoid_backward(y):
    dtype = y.dtype

    y = y.to(tl.float32)
    y = y * (1 - y)
    y = y.to(dtype)

    return y
