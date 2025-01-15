from flax import linen as nn
import jax.numpy as jnp
from typing import Optional, Dict
from .vision import VisionModel
from .text import TextModel

class MoondreamModel(nn.Module):
    vision_config: Dict
    text_config: Dict
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Vision model
        self.vision = VisionModel(
            dim=self.vision_config['dim'],
            patch_size=self.vision_config['patch_size'],
            n_layers=self.vision_config['n_layers'],
            n_heads=self.vision_config['n_heads'],
            dtype=self.dtype,
            name='vision'
        )
        
        # Text model
        self.text = TextModel(
            dim=self.text_config['dim'],
            n_layers=self.text_config['n_layers'],
            n_heads=self.text_config['n_heads'],
            vocab_size=self.text_config['vocab_size'],
            max_context=self.text_config['max_context'],
            dtype=self.dtype,
            name='text'
        )
        
        # Region model components
        self.region = nn.ModuleDict({
            'coord_encoder': nn.Dense(
                features=self.text_config['dim'],
                dtype=self.dtype,
                name='coord_encoder'
            ),
            'coord_decoder': nn.Sequential([
                nn.Dense(features=4*self.text_config['dim'], dtype=self.dtype, name='coord_fc1'),
                nn.gelu,
                nn.Dense(features=2, dtype=self.dtype, name='coord_fc2')
            ]),
            'size_encoder': nn.Dense(
                features=self.text_config['dim'],
                dtype=self.dtype,
                name='size_encoder'
            ),
            'size_decoder': nn.Sequential([
                nn.Dense(features=4*self.text_config['dim'], dtype=self.dtype, name='size_fc1'),
                nn.gelu,
                nn.Dense(features=2, dtype=self.dtype, name='size_fc2')
            ])
        })
    
    def encode_image(self, image: jnp.ndarray) -> jnp.ndarray:
        return self.vision(image)
    
    def generate_text(self, 
                     image_emb: jnp.ndarray,
                     prompt: jnp.ndarray,
                     max_tokens: int = 512) -> jnp.ndarray:
        # Combine image embedding with prompt
        x = jnp.concatenate([image_emb, prompt], axis=1)
        
        # Generate text tokens
        tokens = []
        for _ in range(max_tokens):
            logits, _ = self.text(x)
            next_token = jnp.argmax(logits, axis=-1)
            tokens.append(next_token)
            x = jnp.concatenate([x, next_token], axis=1)
        
        return jnp.array(tokens)
