import os
import pickle
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from moondream.jax import rope as jax_rope
from moondream.torch import rope as torch_rope

CACHE_DIR = "tests/equivalence/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def generate_test_cases():
    """Generate small test cases for quick testing"""
    return [
        # (dim, end, num_heads, rot_dim, batch_size, seq_len, hidden_dim)
        (32, 16, 2, 32, 1, 8, 64),  # Minimal case
        (64, 32, 4, 64, 2, 16, 128),  # Medium case
    ]


def cache_torch_results(test_id: str, results: Dict[str, Any]):
    """Cache PyTorch results for faster debugging"""
    cache_file = os.path.join(CACHE_DIR, f"rope_cache_{test_id}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)


def load_cached_results(test_id: str) -> Dict[str, Any]:
    """Load cached PyTorch results"""
    cache_file = os.path.join(CACHE_DIR, f"rope_cache_{test_id}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


@pytest.mark.parametrize("dim,end,num_heads,rot_dim,batch_size,seq_len,hidden_dim", generate_test_cases())
def test_rope_equivalence(
    dim: int,
    end: int,
    num_heads: int,
    rot_dim: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    regenerate_cache: bool = False,
):
    test_id = f"d{dim}_e{end}_h{num_heads}_r{rot_dim}_b{batch_size}_s{seq_len}_hd{hidden_dim}"

    # Try to load cached results
    cached_results = None if regenerate_cache else load_cached_results(test_id)

    if cached_results is None:
        # Generate PyTorch results
        torch_freqs = torch_rope.precompute_freqs_cis(dim=dim, end=end, dtype=torch.float32)

        x = torch.randn(batch_size, num_heads, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len)

        torch_result = torch_rope.apply_rotary_emb(
            x=x, freqs_cis=torch_freqs, position_ids=position_ids, num_heads=num_heads, rot_dim=rot_dim
        )

        # Cache results
        cached_results = {
            "freqs_cis": torch_freqs.numpy(),
            "input": x.numpy(),
            "position_ids": position_ids.numpy(),
            "output": torch_result.numpy(),
        }
        cache_torch_results(test_id, cached_results)

    # Run JAX implementation
    jax_freqs = jax_rope.precompute_freqs_cis(dim=dim, end=end, dtype=jnp.float32)

    jax_result = jax_rope.apply_rotary_emb(
        x=jnp.array(cached_results["input"]),
        freqs_cis=jax_freqs,
        position_ids=jnp.array(cached_results["position_ids"]),
        num_heads=num_heads,
        rot_dim=rot_dim,
    )

    # Compare results
    np.testing.assert_allclose(cached_results["freqs_cis"], jax_freqs, rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(cached_results["output"], jax_result, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    # Run test with cache regeneration
    for case in generate_test_cases():
        test_rope_equivalence(*case, regenerate_cache=True)
