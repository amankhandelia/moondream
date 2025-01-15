from flax import linen as nn
import jax.numpy as jnp
from .transformer import TransformerBlock

class VisionModel(nn.Module):
    dim: int
    patch_size: int
    n_layers: int
    n_heads: int
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Patch embedding
        self.patch_emb = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            name='patch_emb'
        )
        
        # Position embeddings
        self.pos_emb = self.param('pos_emb',
                                nn.initializers.normal(stddev=0.02),
                                (1, (224//self.patch_size)**2, self.dim))
        
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
        self.ln = nn.LayerNorm(dtype=self.dtype, name='ln')
        
        # Projection MLP
        self.proj_mlp = nn.Sequential([
            nn.Dense(features=4*self.dim, dtype=self.dtype, name='proj_fc1'),
            nn.gelu,
            nn.Dense(features=self.dim, dtype=self.dtype, name='proj_fc2')
        ])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Patch embedding
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//self.patch_size, self.patch_size,
                     w//self.patch_size, self.patch_size)
        x = x.transpose(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, -1, self.patch_size*self.patch_size*c)
        x = self.patch_emb(x)
        
        # Add position embeddings
        x = x + self.pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x, _ = block(x)
        
        # Final layer norm
        x = self.ln(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Projection MLP
        x = self.proj_mlp(x)
        
        return x
