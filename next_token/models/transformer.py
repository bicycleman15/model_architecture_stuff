"""Transformer baseline for the star-graph next-token task.

Thin wrapper around the workspace-root [`models/transformer.py`](../../models/transformer.py)
loaded via `importlib.util` so we don't have to add the workspace root to
`sys.path` (mirrors `state_tracking/models/transformer.py`).

The wrapped class exposes `forward(input_ids, labels=None, **kwargs)` returning:
  * `(loss, stats)` when `labels` is provided (training).
  * `(logits, stats)` when `labels` is None (generation / forced eval).
"""

from __future__ import annotations

import importlib.util
import pathlib

_WS_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WS_TRANSFORMER = _WS_ROOT / "models" / "transformer.py"


def _load_ws_transformer_module():
    spec = importlib.util.spec_from_file_location("_ws_root_transformer", _WS_TRANSFORMER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ws = _load_ws_transformer_module()
Transformer = _ws.Transformer
TransformerConfig = _ws.TransformerConfig


def build_transformer(vocab_size: int, block_size: int, mcfg) -> Transformer:
    config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=mcfg.n_layer,
        dim=mcfg.dim,
        n_head=mcfg.n_head,
        initializer_range=mcfg.get("initializer_range", 0.02),
        norm_eps=mcfg.get("norm_eps", 1e-5),
        use_fused_ops=mcfg.get("use_fused_ops", False),
        use_qk_norm=mcfg.get("use_qk_norm", False),
    )
    return Transformer(config)
