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
        initializer_range = cfg.model.get("initializer_range", 0.02)
        config = TransformerConfig(
            vocab_size=vocab_size,
            block_size=cfg.model.block_size,

            n_layer=cfg.model.n_layer,
            dim=cfg.model.dim,
            n_head=cfg.model.n_head,

            initializer_range=initializer_range,
            norm_eps=norm_eps,
            
            use_fused_ops=use_fused_ops,
            use_qk_norm=use_qk_norm,
        )
        model = Transformer(config)
        return config, model

    elif name == "hourglass":
        proc_dim = cfg.model.get("processor_dim", None)
        proc_config_raw = cfg.model.get("processor_config", None)
        initializer_range = cfg.model.get("initializer_range", 0.02)
        lr_multiplier = cfg.model.get("lr_multiplier", None)

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

            initializer_range=initializer_range,
            lr_multiplier=list(lr_multiplier) if lr_multiplier is not None else None,

            norm_eps=norm_eps,

            use_fused_ops=use_fused_ops,
            use_qk_norm=use_qk_norm,
        )

        model = HierarchicalLM(config)
        model._init_weights(initializer_range)
        if lr_multiplier is not None:
            model.apply_lr_multiplier(list(lr_multiplier))
        return config, model

    elif name == "reinforce_hourglass":
        from models.reinforce_hourglass import ReinforceConfig, ReinforceHierarchicalLM

        proc_dim = cfg.model.processor_dim
        proc_config_raw = cfg.model.processor_config
        initializer_range = cfg.model.initializer_range
        lr_multiplier = cfg.model.lr_multiplier

        config = ReinforceConfig(
            vocab_size=vocab_size,
            block_size=cfg.model.block_size,
            dim=cfg.model.dim,
            n_head=cfg.model.n_head,

            n_compressor_layers=cfg.model.n_compressor_layers,
            n_processor_layers=cfg.model.n_processor_layers,
            n_decoder_layers=cfg.model.n_decoder_layers,

            chunk_method=cfg.model.chunk_method,
            chunk_size=cfg.model.get("chunk_size", 4),

            processor_dim=proc_dim,
            processor_config=proc_config_raw,

            initializer_range=initializer_range,
            lr_multiplier=list(lr_multiplier) if lr_multiplier is not None else None,

            norm_eps=norm_eps,

            use_fused_ops=use_fused_ops,
            use_qk_norm=use_qk_norm,

            reinforce_gamma=cfg.model.reinforce_gamma,
            target_downsample_rate=cfg.model.target_downsample_rate,
            router_logit_scale=cfg.model.router_logit_scale,
            router_softcap=cfg.model.router_softcap,
        )

        model = ReinforceHierarchicalLM(config)
        model._init_weights(initializer_range)
        if lr_multiplier is not None:
            model.apply_lr_multiplier(list(lr_multiplier))
        return config, model

    else:
        raise ValueError(f"Unknown model type: {name}")
