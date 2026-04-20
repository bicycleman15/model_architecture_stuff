"""Transformer baseline: thin wrapper around the workspace-root Transformer.
"""

from __future__ import annotations

import importlib.util
import pathlib

_WS_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WS_TRANSFORMER = _WS_ROOT / "models" / "transformer.py"


def _load_ws_transformer_module():
    spec = importlib.util.spec_from_file_location(
        "_ws_root_transformer", _WS_TRANSFORMER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ws = _load_ws_transformer_module()
_RootTransformer = _ws.Transformer
TransformerConfig = _ws.TransformerConfig


class Transformer(_RootTransformer):
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, **kwargs):
        out = super().forward(input_ids=x, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out


def build_transformer(vocab_size: int, mcfg) -> Transformer:
    conf = TransformerConfig(
        vocab_size=vocab_size,
        block_size=mcfg.block_size,
        n_layer=mcfg.n_layer,
        dim=mcfg.dim,
        n_head=mcfg.n_head,
    )
    return Transformer(conf)
