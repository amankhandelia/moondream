import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

class Attention(nn.Module):
    dim: int
    n_heads: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 freqs_cis: Optional[jnp.ndarray] = None,
                 mask: Optional[jnp.ndarray] = None,
                 pos: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        head_dim = self.dim // self.n_heads
        
        # Project input to query, key, value
        qkv = nn.Dense(features=3*self.dim, dtype=self.dtype, name='qkv')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, head_dim)
        
        # Apply rotary embeddings if provided
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis, pos)
            k = apply_rotary_emb(k, freqs_cis, pos)
        
        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
            
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        out = out.reshape(out.shape[0], out.shape[1], self.dim)
        
        # Final projection
        out = nn.Dense(features=self.dim, dtype=self.dtype, name='proj')(out)
        return out, jnp.stack([k, v])

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.polar(jnp.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    x: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    pos: int
) -> jnp.ndarray:
    x_ = x.reshape(*x.shape[:-1], -1, 2)
    x_ = jax.lax.complex(x_[..., 0], x_[..., 1])
    
    # Apply rotation using precomputed frequencies
    freqs_cis = freqs_cis[pos:pos+x.shape[1]]
    x_out = x_ * freqs_cis
    
    # Convert back to real representation
    x_out = jnp.stack([jnp.real(x_out), jnp.imag(x_out)], axis=-1)
    return x_out.reshape(*x.shape)
