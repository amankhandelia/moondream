import logging
import os
import pickle
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from moondream.jax import layers as jax_layers
from moondream.jax import text as jax_text
from moondream.jax.layers import AttentionWeights, LinearWeights
from moondream.torch import text as torch_text
from moondream.torch.config import TextConfig as TorchTextConfig

CACHE_DIR = "tests/equivalence/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_test_cases():
    """Generate test cases for text model components"""
    return [
        # (batch_size, seq_len, hidden_dim, num_heads, vocab_size)
        (1, 8, 64, 2, 100),  # Minimal case
        (2, 16, 128, 4, 200),  # Medium case
    ]


def cache_torch_results(test_id: str, results: Dict[str, Any]):
    cache_file = os.path.join(CACHE_DIR, f"text_cache_{test_id}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)


def load_cached_results(test_id: str) -> Dict[str, Any]:
    cache_file = os.path.join(CACHE_DIR, f"text_cache_{test_id}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def generate_kv_cache(batch_size, num_heads, seq_len, head_dim, num_layers):
    """Generate kv cache for testing"""
    return torch.zeros(num_layers, 2, batch_size, num_heads, seq_len, head_dim)


def create_test_model(hidden_dim, num_heads, num_layers, vocab_size, seq_len):
    """Create test model with config"""
    config = TorchTextConfig(
        dim=hidden_dim,
        n_heads=num_heads,
        n_layers=num_layers,
        vocab_size=vocab_size,
        max_context=seq_len,
    )
    return torch_text.build_text_model(config, dtype=torch.float32), config


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads,vocab_size", generate_test_cases())
def test_text_encoder_equivalence(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    regenerate_cache: bool = False,
):
    test_id = f"text_b{batch_size}_s{seq_len}_h{hidden_dim}_nh{num_heads}_v{vocab_size}"

    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        # Create configs
        torch_config = TorchTextConfig(
            dim=hidden_dim,
            n_heads=num_heads,
            n_layers=2,
            vocab_size=vocab_size,
            max_context=seq_len,
        )

        # Generate input
        torch_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        torch_model = torch_text.build_text_model(torch_config, dtype=torch.float32)

        # Run torch implementation
        torch_output = torch_text.text_encoder(torch_input_ids, torch_model)

        # Cache results
        cached_results = {
            "input_ids": torch_input_ids.numpy(),
            "wte": torch_model.wte.detach().numpy(),
            "output": torch_output.detach().numpy(),
        }
        cache_torch_results(test_id, cached_results)

    # Run JAX implementation
    jax_input_ids = jnp.array(cached_results["input_ids"])
    jax_weights = jax_text.TextWeights(
        wte=jnp.array(cached_results["wte"]),
        blocks=[],  # Not needed for encoder test
        post_ln=None,
        lm_head=None,
        freqs_cis=None,
    )

    with jax.default_matmul_precision("float32"):
        jax_output = jax_text.text_encoder(jax_input_ids, jax_weights)

    # Compare results
    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads,vocab_size", generate_test_cases())
def test_attn_equivalence(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    regenerate_cache: bool = False,
):
    test_id = f"attn_b{batch_size}_s{seq_len}_h{hidden_dim}_nh{num_heads}_v{vocab_size}"
    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        # Generate input and model
        torch_input = torch.randn(batch_size, seq_len, hidden_dim)
        torch_model, _ = create_test_model(hidden_dim, num_heads, 1, vocab_size, seq_len)
        head_dim = hidden_dim // num_heads
        kv_cache = generate_kv_cache(batch_size, num_heads, seq_len, head_dim, 1)[0]
        attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)

        # Run torch implementation
        torch_output, torch_kv = torch_text.attn(
            torch_input, torch_model.blocks[0].attn, torch_model.freqs_cis, kv_cache, attn_mask, num_heads, pos=0
        )

        cached_results = {
            "input": torch_input.numpy(),
            "freqs_cis": torch_model.freqs_cis.numpy(),
            "kv_cache": kv_cache.numpy(),
            "attn_mask": attn_mask.numpy(),
            "weights": {
                "qkv_weight": torch_model.blocks[0].attn.qkv.weight.detach().numpy(),
                "qkv_bias": torch_model.blocks[0].attn.qkv.bias.detach().numpy(),
                "proj_weight": torch_model.blocks[0].attn.proj.weight.detach().numpy(),
                "proj_bias": torch_model.blocks[0].attn.proj.bias.detach().numpy(),
            },
            "output": torch_output.numpy(),
            "kv_output": torch_kv.numpy(),
        }
        cache_torch_results(test_id, cached_results)

    # Run JAX implementation
    jax_input = jnp.array(cached_results["input"])
    jax_weights = jax_text.TextWeights(
        wte=None,
        blocks=[
            {
                "attn": AttentionWeights(
                    qkv=LinearWeights(
                        weight=jnp.array(cached_results["weights"]["qkv_weight"]),
                        bias=jnp.array(cached_results["weights"]["qkv_bias"]),
                    ),
                    proj=LinearWeights(
                        weight=jnp.array(cached_results["weights"]["proj_weight"]),
                        bias=jnp.array(cached_results["weights"]["proj_bias"]),
                    ),
                )
            }
        ],
        post_ln=None,
        lm_head=None,
        freqs_cis=jnp.array(cached_results["freqs_cis"]),
    )

    with jax.default_matmul_precision("float32"):
        jax_output, jax_kv = jax_text.attn(
            jax_input,
            jax_weights.blocks[0].attn,
            jax_weights.freqs_cis,
            jnp.array(cached_results["kv_cache"]),
            jnp.array(cached_results["attn_mask"]),
            num_heads,
            pos=0,
        )

    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cached_results["kv_output"], jax_kv, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads,vocab_size", generate_test_cases())
def test_text_decoder_equivalence(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    regenerate_cache: bool = False,
):
    test_id = f"decoder_b{batch_size}_s{seq_len}_h{hidden_dim}_nh{num_heads}_v{vocab_size}"
    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        # Generate input and model
        torch_input = torch.randn(batch_size, seq_len, hidden_dim)
        torch_model, config = create_test_model(hidden_dim, num_heads, 2, vocab_size, seq_len)
        head_dim = hidden_dim // num_heads
        kv_cache = generate_kv_cache(batch_size, num_heads, seq_len, head_dim, 2)
        attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)

        # Run torch implementation
        torch_output, torch_kv = torch_text.text_decoder(
            torch_input, torch_model, kv_cache, attn_mask, pos=0, config=config
        )

        # Cache results with model weights
        cached_results = {
            "input": torch_input.numpy(),
            "output": torch_output.numpy(),
            "kv_output": torch_kv.numpy(),
            "model_state": {
                "freqs_cis": torch_model.freqs_cis.numpy(),
                "kv_cache": kv_cache.numpy(),
                "attn_mask": attn_mask.numpy(),
            },
        }
        # Add weights for each layer
        for i, block in enumerate(torch_model.blocks):
            cached_results[f"layer_{i}"] = {
                "ln_weight": block.ln.weight.detach().numpy(),
                "ln_bias": block.ln.bias.detach().numpy(),
                "qkv_weight": block.attn.qkv.weight.detach().numpy(),
                "qkv_bias": block.attn.qkv.bias.detach().numpy(),
                "proj_weight": block.attn.proj.weight.detach().numpy(),
                "proj_bias": block.attn.proj.bias.detach().numpy(),
                "mlp_fc1_weight": block.mlp.fc1.weight.detach().numpy(),
                "mlp_fc1_bias": block.mlp.fc1.bias.detach().numpy(),
                "mlp_fc2_weight": block.mlp.fc2.weight.detach().numpy(),
                "mlp_fc2_bias": block.mlp.fc2.bias.detach().numpy(),
            }
        cache_torch_results(test_id, cached_results)

    # Convert cached weights to JAX format and run comparison
    # ... implement JAX test here following same pattern as above ...


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads,vocab_size", generate_test_cases())
def test_lm_head_equivalence(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    regenerate_cache: bool = False,
):
    test_id = f"lm_head_b{batch_size}_s{seq_len}_h{hidden_dim}_nh{num_heads}_v{vocab_size}"
    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        with torch.no_grad():
            torch_hidden = torch.randn(batch_size, seq_len, hidden_dim)
            torch_model, _ = create_test_model(hidden_dim, num_heads, 1, vocab_size, seq_len)
            torch_output = torch_text.lm_head(torch_hidden, torch_model)

        cached_results = {
            "hidden": torch_hidden.numpy(),
            "weights": {
                "post_ln_weight": torch_model.post_ln.weight.detach().numpy(),
                "post_ln_bias": torch_model.post_ln.bias.detach().numpy(),
                "lm_head_weight": torch_model.lm_head.weight.detach().numpy(),
                "lm_head_bias": torch_model.lm_head.bias.detach().numpy(),
            },
            "output": torch_output.numpy(),
        }
        cache_torch_results(test_id, cached_results)

    jax_hidden = jnp.array(cached_results["hidden"])
    jax_weights = jax_text.TextWeights(
        wte=None,
        blocks=[],
        post_ln=jax_layers.LayerNormWeights(
            weight=jnp.array(cached_results["weights"]["post_ln_weight"]),
            bias=jnp.array(cached_results["weights"]["post_ln_bias"]),
        ),
        lm_head=jax_layers.LinearWeights(
            weight=jnp.array(cached_results["weights"]["lm_head_weight"]),
            bias=jnp.array(cached_results["weights"]["lm_head_bias"]),
        ),
        freqs_cis=None,
    )

    with jax.default_matmul_precision("float32"):
        jax_output = jax_text.lm_head(jax_hidden, jax_weights)

    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads,vocab_size", generate_test_cases())
def test_prefill_equivalence(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    regenerate_cache: bool = False,
):
    test_id = f"prefill_b{batch_size}_s{seq_len}_h{hidden_dim}_nh{num_heads}_v{vocab_size}"
    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        torch_embeds = torch.randn(batch_size, seq_len, hidden_dim)
        torch_model, config = create_test_model(hidden_dim, num_heads, 2, vocab_size, seq_len)
        head_dim = hidden_dim // num_heads
        kv_cache = generate_kv_cache(batch_size, num_heads, seq_len, head_dim, 2)
        attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        pos = 0

        torch_output = torch_text.prefill(torch_embeds, kv_cache, attn_mask, pos, torch_model, config)

        cached_results = {
            "embeds": torch_embeds.numpy(),
            "kv_cache": kv_cache.numpy(),
            "attn_mask": attn_mask.numpy(),
            "output": torch_output.numpy(),
            "model_state": torch_model.state_dict(),
        }
        cache_torch_results(test_id, cached_results)

    # Convert cached model state to JAX weights format
    jax_weights = jax_text.TextWeights(
        wte=None,
        blocks=[],  # Convert blocks from cached state
        post_ln=None,
        lm_head=None,
        freqs_cis=jnp.array(cached_results["model_state"]["freqs_cis"]),
    )

    with jax.default_matmul_precision("float32"):
        jax_output = jax_text.prefill(
            jnp.array(cached_results["embeds"]),
            jnp.array(cached_results["kv_cache"]),
            jnp.array(cached_results["attn_mask"]),
            0,
            jax_weights,
            config,
        )

    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads,vocab_size", generate_test_cases())
def test_decode_one_token_equivalence(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    regenerate_cache: bool = False,
):
    test_id = f"decode_one_b{batch_size}_s{seq_len}_h{hidden_dim}_nh{num_heads}_v{vocab_size}"
    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        token_emb = torch.randn(hidden_dim)
        torch_model, config = create_test_model(hidden_dim, num_heads, 2, vocab_size, seq_len)
        head_dim = hidden_dim // num_heads
        kv_cache = generate_kv_cache(batch_size, num_heads, seq_len, head_dim, 2)
        attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        pos = 0

        torch_logits, torch_hidden, torch_kv = torch_text.decode_one_token(
            token_emb, kv_cache, attn_mask, pos, torch_model, config
        )

        cached_results = {
            "token_emb": token_emb.numpy(),
            "kv_cache": kv_cache.numpy(),
            "attn_mask": attn_mask.numpy(),
            "logits": torch_logits.numpy(),
            "hidden": torch_hidden.numpy(),
            "kv": torch_kv.numpy(),
            "model_state": torch_model.state_dict(),
        }
        cache_torch_results(test_id, cached_results)

    # Convert cached model state to JAX weights format
    jax_weights = jax_text.TextWeights(
        wte=None,
        blocks=[],  # Convert blocks from cached state
        post_ln=None,
        lm_head=None,
        freqs_cis=jnp.array(cached_results["model_state"]["freqs_cis"]),
    )

    with jax.default_matmul_precision("float32"):
        jax_logits, jax_hidden, jax_kv = jax_text.decode_one_token(
            jnp.array(cached_results["token_emb"]),
            jnp.array(cached_results["kv_cache"]),
            jnp.array(cached_results["attn_mask"]),
            0,
            jax_weights,
            config,
        )

    np.testing.assert_allclose(cached_results["logits"], jax_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cached_results["hidden"], jax_hidden, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cached_results["kv"], jax_kv, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    # Run all tests with cache regeneration
    for case in generate_test_cases():
        test_text_encoder_equivalence(*case, regenerate_cache=True)
        test_attn_equivalence(*case, regenerate_cache=True)
        test_text_decoder_equivalence(*case, regenerate_cache=True)
        test_lm_head_equivalence(*case, regenerate_cache=True)
        test_prefill_equivalence(*case, regenerate_cache=True)
        test_decode_one_token_equivalence(*case, regenerate_cache=True)
