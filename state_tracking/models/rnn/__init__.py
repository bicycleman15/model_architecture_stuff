"""M2RNN-based RNN baseline for the state-tracking task.

Vendored from open-lm-engine/accelerated-model-architectures (xma.layers.m2rnn)
and wrapped in a transformer-style LM stack. See ATTRIBUTION.md.
"""

from state_tracking.models.rnn.layer import M2RNNLayer
from state_tracking.models.rnn.model import RNNModel, build_rnn
from state_tracking.models.rnn.op import m2rnn

__all__ = ["M2RNNLayer", "RNNModel", "build_rnn", "m2rnn"]
