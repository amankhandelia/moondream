from flax import linen as nn
import jax.numpy as jnp
from .attention import Attention

class TransformerBlock(nn.Module):
    dim: int
    n_heads: int
    mlp_dim: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray,
                 freqs_cis: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 pos: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Attention block
        attn_out, kv_cache = Attention(
            dim=self.dim,
            n_heads=self.n_heads,
            dtype=self.dtype,
            name='attn'
        )(x, freqs_cis, mask, pos)
        
        # Add & Norm
        x = x + attn_out
        x = nn.LayerNorm(dtype=self.dtype, name='ln1')(x)
        
        # MLP block
        mlp_out = nn.Dense(features=self.mlp_dim, dtype=self.dtype, name='mlp_fc1')(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(features=self.dim, dtype=self.dtype, name='mlp_fc2')(mlp_out)
        
        # Add & Norm
        x = x + mlp_out
        x = nn.LayerNorm(dtype=self.dtype, name='ln2')(x)
        
        return x, kv_cache
