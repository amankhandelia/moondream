from flax import linen as nn
import jax.numpy as jnp
from typing import Optional, Tuple
from .attention import Attention, precompute_freqs_cis, apply_rotary_emb

class TextModel(nn.Module):
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    max_context: int
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Token embeddings
        self.wte = self.param('wte', 
                            nn.initializers.normal(stddev=0.02),
                            (self.vocab_size, self.dim))
        
        # Precompute rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            dim=self.dim // self.n_heads,
            end=self.max_context
        )
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                dim=self.dim,
                n_heads=self.n_heads,
                mlp_dim=4*self.dim,
                dtype=self.dtype,
                name=f'block_{i}'
            ) for i in range(self.n_layers)
        ]
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(dtype=self.dtype, name='ln_f')
        
        # Language model head
        self.lm_head = nn.Dense(
            features=self.vocab_size,
            dtype=self.dtype,
            name='lm_head'
        )
    
    def __call__(self, 
                 input_ids: jnp.ndarray,
                 kv_cache: Optional[jnp.ndarray] = None,
                 mask: Optional[jnp.ndarray] = None,
                 pos: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get token embeddings
        x = jnp.take(self.wte, input_ids, axis=0)
        
        # Apply transformer blocks
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            x, block_kv_cache = block(
                x,
                freqs_cis=self.freqs_cis,
                mask=mask,
                pos=pos
            )
            new_kv_cache.append(block_kv_cache)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x[:, -1])
        
        return logits, jnp.stack(new_kv_cache)
