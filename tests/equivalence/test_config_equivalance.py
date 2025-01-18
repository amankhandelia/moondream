from moondream.jax.config import MoondreamConfig as JaxConfig
from moondream.torch.config import MoondreamConfig as TorchConfig


def test_default_configs_match():
    torch_config = TorchConfig()
    jax_config = JaxConfig()

    # Test text config
    assert torch_config.text.dim == jax_config.text.dim
    assert torch_config.text.n_layers == jax_config.text.n_layers
    assert torch_config.text.vocab_size == jax_config.text.vocab_size
    assert torch_config.text.max_context == jax_config.text.max_context
    assert torch_config.text.n_heads == jax_config.text.n_heads
    assert torch_config.text.prefix_attn == jax_config.text.prefix_attn

    # Test vision config
    assert torch_config.vision.enc_dim == jax_config.vision.enc_dim
    assert torch_config.vision.enc_patch_size == jax_config.vision.enc_patch_size
    assert torch_config.vision.enc_n_layers == jax_config.vision.enc_n_layers
    assert torch_config.vision.enc_ff_dim == jax_config.vision.enc_ff_dim
    assert torch_config.vision.enc_n_heads == jax_config.vision.enc_n_heads
    assert torch_config.vision.proj_out_dim == jax_config.vision.proj_out_dim
    assert torch_config.vision.crop_size == jax_config.vision.crop_size
    assert torch_config.vision.in_channels == jax_config.vision.in_channels
    assert torch_config.vision.max_crops == jax_config.vision.max_crops
    assert torch_config.vision.overlap_margin == jax_config.vision.overlap_margin
    assert torch_config.vision.proj_inner_dim == jax_config.vision.proj_inner_dim

    # Test region config
    assert torch_config.region.dim == jax_config.region.dim
    assert torch_config.region.coord_feat_dim == jax_config.region.coord_feat_dim
    assert torch_config.region.coord_out_dim == jax_config.region.coord_out_dim
    assert torch_config.region.size_feat_dim == jax_config.region.size_feat_dim
    assert torch_config.region.size_out_dim == jax_config.region.size_out_dim
    assert torch_config.region.inner_dim == jax_config.region.inner_dim

    # Test tokenizer config
    assert torch_config.tokenizer.bos_id == jax_config.tokenizer.bos_id
    assert torch_config.tokenizer.eos_id == jax_config.tokenizer.eos_id
    assert torch_config.tokenizer.templates == jax_config.tokenizer.templates


def test_from_dict():
    config_dict = {
        "text": {"dim": 1024, "n_layers": 12},
        "vision": {"enc_dim": 768, "crop_size": 224},
        "region": {"dim": 1024, "inner_dim": 4096},
        "tokenizer": {"bos_id": 1, "eos_id": 2},
    }

    torch_config = TorchConfig.from_dict(config_dict)
    jax_config = JaxConfig.from_dict(config_dict)

    assert torch_config.text.dim == jax_config.text.dim == 1024
    assert torch_config.vision.enc_dim == jax_config.vision.enc_dim == 768
    assert torch_config.region.inner_dim == jax_config.region.inner_dim == 4096
    assert torch_config.tokenizer.bos_id == jax_config.tokenizer.bos_id == 1


def test_to_dict():
    torch_config = TorchConfig()
    jax_config = JaxConfig()

    torch_dict = torch_config.to_dict()
    jax_dict = jax_config.to_dict()

    assert torch_dict == jax_dict
