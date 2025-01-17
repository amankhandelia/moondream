from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


def gelu_approx(x):
    return jax.nn.gelu(x, approximate=True)


@dataclass
class LinearWeights:
    weight: jnp.ndarray
    bias: jnp.ndarray


def linear(x: jnp.ndarray, w: LinearWeights) -> jnp.ndarray:
    return x @ w.weight.T + w.bias


@dataclass
class LayerNormWeights:
    weight: jnp.ndarray
    bias: jnp.ndarray


def layer_norm(x: jnp.ndarray, w: LayerNormWeights) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return w.weight * (x - mean) / jnp.sqrt(variance + 1e-5) + w.bias


@dataclass
class MLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights
    act: Literal["gelu_approx"] = "gelu_approx"


def mlp(x: jnp.ndarray, w: MLPWeights) -> jnp.ndarray:
    x = linear(x, w.fc1)
    x = gelu_approx(x)
    x = linear(x, w.fc2)
    return x


@dataclass
class AttentionWeights:
    qkv: LinearWeights
    proj: LinearWeights


def attn(x: jnp.ndarray, w: AttentionWeights, n_heads: int) -> jnp.ndarray:
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads
    
    qkv = linear(x, w.qkv)
    q, k, v = [
        t.reshape(bsz, q_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        for t in jnp.split(qkv, 3, axis=-1)
    ]
    
    scale = 1.0 / jnp.sqrt(head_dim)
    attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    
    out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(bsz, q_len, d_model)
    out = linear(out, w.proj)
    return out
