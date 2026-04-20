"""Model dispatcher for the state-tracking training script.

Selects and constructs a model from a Hydra config (`cfg.model.name`).
"""


def get_model(cfg, vocab_size: int):
    name = cfg.model.name

    if name in ("deltaproduct", "deltanet"):
        from state_tracking.models.deltaproduct import build_deltaproduct
        return build_deltaproduct(vocab_size, cfg.model)

    if name == "transformer":
        from state_tracking.models.transformer import build_transformer
        return build_transformer(vocab_size, cfg.model)

    if name == "rnn":
        from state_tracking.models.rnn import build_rnn
        return build_rnn(vocab_size, cfg.model)

    raise ValueError(f"Unknown model name: {name!r}")
