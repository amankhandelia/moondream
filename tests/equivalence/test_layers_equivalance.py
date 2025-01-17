import logging
import os
import pickle
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from moondream.jax import layers as jax_layers
from moondream.torch import layers as torch_layers

CACHE_DIR = "tests/equivalence/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_test_cases():
    """Generate test cases for different layer configurations"""
    return [
        # (batch_size, seq_len, hidden_dim, num_heads)
        (1, 8, 64, 2),  # Minimal case
        (2, 16, 128, 4),  # Medium case
    ]


def generate_linear_test_cases():
    return [
        # (batch_size, seq_len, in_dim, out_dim)
        (1, 1, 4, 8),  # simple base case for debugging
        (1, 8, 64, 128),
        (2, 16, 128, 256),
    ]


def generate_layer_norm_test_cases():
    return [
        # (batch_size, seq_len, hidden_dim)
        (1, 8, 64),
        (2, 16, 128),
    ]


def generate_mlp_test_cases():
    return [
        # (batch_size, seq_len, hidden_dim, intermediate_dim)
        (1, 8, 64, 256),
        (2, 16, 128, 512),
    ]


def cache_torch_results(test_id: str, results: Dict[str, Any]):
    cache_file = os.path.join(CACHE_DIR, f"layers_cache_{test_id}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)


def load_cached_results(test_id: str) -> Dict[str, Any]:
    cache_file = os.path.join(CACHE_DIR, f"layers_cache_{test_id}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def create_test_weights(hidden_dim: int, num_heads: int) -> Tuple[Any, Any]:
    # Create PyTorch weights
    torch_weights = torch_layers.AttentionWeights(
        qkv=torch_layers.LinearWeights(
            weight=torch.randn(3 * hidden_dim, hidden_dim),
            bias=torch.randn(3 * hidden_dim),
        ),
        proj=torch_layers.LinearWeights(
            weight=torch.randn(hidden_dim, hidden_dim),
            bias=torch.randn(hidden_dim),
        ),
    )

    # Convert to JAX weights
    jax_weights = jax_layers.AttentionWeights(
        qkv=jax_layers.LinearWeights(
            weight=jnp.array(torch_weights.qkv.weight.numpy()),
            bias=jnp.array(torch_weights.qkv.bias.numpy()),
        ),
        proj=jax_layers.LinearWeights(
            weight=jnp.array(torch_weights.proj.weight.numpy()),
            bias=jnp.array(torch_weights.proj.bias.numpy()),
        ),
    )

    return torch_weights, jax_weights


def create_linear_weights(in_dim: int, out_dim: int) -> Any:
    torch_weights = torch_layers.LinearWeights(
        weight=torch.randn(out_dim, in_dim),
        bias=torch.randn(out_dim),
    )

    return torch_weights


def create_layer_norm_weights(hidden_dim: int) -> Tuple[Any, Any]:
    torch_weights = torch_layers.LayerNormWeights(
        weight=torch.randn(hidden_dim),
        bias=torch.randn(hidden_dim),
    )

    jax_weights = jax_layers.LayerNormWeights(
        weight=jnp.array(torch_weights.weight.numpy()),
        bias=jnp.array(torch_weights.bias.numpy()),
    )

    return torch_weights, jax_weights


def create_mlp_weights(hidden_dim: int, intermediate_dim: int) -> Tuple[Any, Any]:
    torch_weights = torch_layers.MLPWeights(
        fc1=torch_layers.LinearWeights(
            weight=torch.randn(intermediate_dim, hidden_dim),
            bias=torch.randn(intermediate_dim),
        ),
        fc2=torch_layers.LinearWeights(
            weight=torch.randn(hidden_dim, intermediate_dim),
            bias=torch.randn(hidden_dim),
        ),
    )

    jax_weights = jax_layers.MLPWeights(
        fc1=jax_layers.LinearWeights(
            weight=jnp.array(torch_weights.fc1.weight.numpy()),
            bias=jnp.array(torch_weights.fc1.bias.numpy()),
        ),
        fc2=jax_layers.LinearWeights(
            weight=jnp.array(torch_weights.fc2.weight.numpy()),
            bias=jnp.array(torch_weights.fc2.bias.numpy()),
        ),
    )

    return torch_weights, jax_weights


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads", generate_test_cases())
def test_attention_equivalence(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    regenerate_cache: bool = False,
):
    test_id = f"b{batch_size}_s{seq_len}_h{hidden_dim}_nh{num_heads}"

    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        # Generate input and weights
        torch_input = torch.randn(batch_size, seq_len, hidden_dim)
        torch_weights, jax_weights = create_test_weights(hidden_dim, num_heads)

        # Run PyTorch implementation
        torch_output = torch_layers.attn(torch_input, torch_weights, num_heads)

        # Cache results
        cached_results = {
            "input": torch_input.numpy(),
            "weights": {
                "qkv_weight": torch_weights.qkv.weight.numpy(),
                "qkv_bias": torch_weights.qkv.bias.numpy(),
                "proj_weight": torch_weights.proj.weight.numpy(),
                "proj_bias": torch_weights.proj.bias.numpy(),
            },
            "output": torch_output.numpy(),
        }
        cache_torch_results(test_id, cached_results)

    # Run JAX implementation with cached inputs
    jax_input = jnp.array(cached_results["input"])
    jax_weights = jax_layers.AttentionWeights(
        qkv=jax_layers.LinearWeights(
            weight=jnp.array(cached_results["weights"]["qkv_weight"]),
            bias=jnp.array(cached_results["weights"]["qkv_bias"]),
        ),
        proj=jax_layers.LinearWeights(
            weight=jnp.array(cached_results["weights"]["proj_weight"]),
            bias=jnp.array(cached_results["weights"]["proj_bias"]),
        ),
    )

    jax_output = jax_layers.attn(jax_input, jax_weights, num_heads)

    # Compare results
    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size,seq_len,in_dim,out_dim", generate_linear_test_cases())
def test_linear_equivalence(
    batch_size: int,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    regenerate_cache: bool = False,
):
    test_id = f"linear_b{batch_size}_s{seq_len}_i{in_dim}_o{out_dim}"

    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        torch_input = torch.randn(batch_size, seq_len, in_dim)
        torch_weights = create_linear_weights(in_dim, out_dim)

        torch_output = torch_layers.linear(torch_input, torch_weights)

        cached_results = {
            "input": torch_input.numpy(),
            "weights": {
                "weight": torch_weights.weight.numpy(),
                "bias": torch_weights.bias.numpy(),
            },
            "output": torch_output.numpy(),
        }
        cache_torch_results(test_id, cached_results)

    jax_input = jnp.array(cached_results["input"])
    jax_weights = jax_layers.LinearWeights(
        weight=jnp.array(cached_results["weights"]["weight"]),
        bias=jnp.array(cached_results["weights"]["bias"]),
    )

    with jax.default_matmul_precision("float32"):
        jax_output = jax_layers.linear(jax_input, jax_weights)
    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim", generate_layer_norm_test_cases())
def test_layer_norm_equivalence(batch_size: int, seq_len: int, hidden_dim: int, regenerate_cache: bool = False):
    test_id = f"ln_b{batch_size}_s{seq_len}_h{hidden_dim}"

    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        torch_input = torch.randn(batch_size, seq_len, hidden_dim)
        torch_weights, jax_weights = create_layer_norm_weights(hidden_dim)

        torch_output = torch_layers.layer_norm(torch_input, torch_weights)

        cached_results = {
            "input": torch_input.numpy(),
            "weights": {
                "weight": torch_weights.weight.numpy(),
                "bias": torch_weights.bias.numpy(),
            },
            "output": torch_output.numpy(),
        }
        cache_torch_results(test_id, cached_results)

    jax_input = jnp.array(cached_results["input"])
    jax_weights = jax_layers.LayerNormWeights(
        weight=jnp.array(cached_results["weights"]["weight"]),
        bias=jnp.array(cached_results["weights"]["bias"]),
    )

    jax_output = jax_layers.layer_norm(jax_input, jax_weights)
    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,intermediate_dim", generate_mlp_test_cases())
def test_mlp_equivalence(
    batch_size: int, seq_len: int, hidden_dim: int, intermediate_dim: int, regenerate_cache: bool = False
):
    test_id = f"mlp_b{batch_size}_s{seq_len}_h{hidden_dim}_i{intermediate_dim}"

    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        torch_input = torch.randn(batch_size, seq_len, hidden_dim)
        torch_weights, jax_weights = create_mlp_weights(hidden_dim, intermediate_dim)

        torch_output = torch_layers.mlp(torch_input, torch_weights)

        cached_results = {
            "input": torch_input.numpy(),
            "weights": {
                "fc1_weight": torch_weights.fc1.weight.numpy(),
                "fc1_bias": torch_weights.fc1.bias.numpy(),
                "fc2_weight": torch_weights.fc2.weight.numpy(),
                "fc2_bias": torch_weights.fc2.bias.numpy(),
            },
            "output": torch_output.numpy(),
        }
        cache_torch_results(test_id, cached_results)

    jax_input = jnp.array(cached_results["input"])
    jax_weights = jax_layers.MLPWeights(
        fc1=jax_layers.LinearWeights(
            weight=jnp.array(cached_results["weights"]["fc1_weight"]),
            bias=jnp.array(cached_results["weights"]["fc1_bias"]),
        ),
        fc2=jax_layers.LinearWeights(
            weight=jnp.array(cached_results["weights"]["fc2_weight"]),
            bias=jnp.array(cached_results["weights"]["fc2_bias"]),
        ),
    )

    with jax.default_matmul_precision("float32"):
        jax_output = jax_layers.mlp(jax_input, jax_weights)

    np.testing.assert_allclose(cached_results["output"], jax_output, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    # Run all tests with cache regeneration
    for case in generate_linear_test_cases():
        test_linear_equivalence(*case, regenerate_cache=True)
    for case in generate_layer_norm_test_cases():
        test_layer_norm_equivalence(*case, regenerate_cache=True)
    for case in generate_mlp_test_cases():
        test_mlp_equivalence(*case, regenerate_cache=True)
    for case in generate_test_cases():
        test_attention_equivalence(*case, regenerate_cache=True)
