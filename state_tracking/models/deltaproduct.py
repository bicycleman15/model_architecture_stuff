"""DeltaProduct / DeltaNet wrapper around fla's GatedDeltaProductForCausalLM.

DeltaNet is just DeltaProduct with num_householder=1, so a single class serves both.
"""

from fla import GatedDeltaProductForCausalLM
from fla.models import GatedDeltaProductConfig
from torch import Tensor


class DeltaProduct(GatedDeltaProductForCausalLM):
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:  # [B, T] -> [B, T, V]
        return super().forward(input_ids=x).logits


def build_deltaproduct(vocab_size: int, mcfg) -> DeltaProduct:
    conf = GatedDeltaProductConfig(
        vocab_size=vocab_size,
        hidden_size=mcfg.d_state,
        num_hidden_layers=mcfg.n_layers,
        num_heads=mcfg.n_heads,
        head_dim=mcfg.head_dim,
        num_householder=mcfg.num_householder,
        allow_neg_eigval=mcfg.allow_neg_eigval,
        use_short_conv=False,
        use_gate=False,
        use_forget_gate=False,
        expand_v=1,
        fuse_cross_entropy=False,
        bos_token_id=0,
        eos_token_id=vocab_size - 1,
    )
    return DeltaProduct(conf)
