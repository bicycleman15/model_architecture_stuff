from models.transformer import TransformerConfig, Transformer
from models.hourglass import Config as HourglassConfig, HierarchicalLM


def get_model(cfg):
    """Dispatch model construction based on cfg.model.name from Hydra config."""

    name = cfg.model.name
    vocab_size = cfg.dataset.vocab_size
    norm_eps = cfg.train.norm_eps
    use_fused_ops = cfg.model.get("use_fused_ops", False)
    use_qk_norm = cfg.model.get("use_qk_norm", False)

    if name == "transformer":
        config = TransformerConfig(
            vocab_size=vocab_size,
            block_size=cfg.model.block_size,

            n_layer=cfg.model.n_layer,
            dim=cfg.model.dim,
            n_head=cfg.model.n_head,

            norm_eps=norm_eps,
            
            use_fused_ops=use_fused_ops,
            use_qk_norm=use_qk_norm,
        )
        model = Transformer(config)
        return config, model

    elif name == "hourglass":
        proc_dim = cfg.model.get("processor_dim", None)
        proc_config_raw = cfg.model.get("processor_config", None)

        config = HourglassConfig(
            vocab_size=vocab_size,
            block_size=cfg.model.block_size,
            dim=cfg.model.dim,
            n_head=cfg.model.n_head,

            n_compressor_layers=cfg.model.n_compressor_layers,
            n_processor_layers=cfg.model.n_processor_layers,
            n_decoder_layers=cfg.model.n_decoder_layers,

            chunk_method=cfg.model.chunk_method,
            chunk_size=cfg.model.chunk_size,

            processor_dim=proc_dim,
            processor_config=proc_config_raw,

            norm_eps=norm_eps,

            use_fused_ops=use_fused_ops,
            use_qk_norm=use_qk_norm,
        )

        model = HierarchicalLM(config)
        return config, model

    else:
        raise ValueError(f"Unknown model type: {name}")
