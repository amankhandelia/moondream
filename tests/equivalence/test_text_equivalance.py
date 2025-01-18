import logging
import os
import pickle
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from moondream.jax import text as jax_text
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


if __name__ == "__main__":
    # Run tests with cache regeneration
    for case in generate_test_cases():
        test_text_encoder_equivalence(*case, regenerate_cache=True)
