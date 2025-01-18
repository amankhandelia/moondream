from typing import Any, NamedTuple

import jax.numpy as jnp
from jax.nn import softmax

from .config import TextConfig
from .layers import AttentionWeights, layer_norm, linear, mlp
from .rope import apply_rotary_emb


class TextWeights(NamedTuple):
    wte: jnp.ndarray
    blocks: list
    post_ln: Any
    lm_head: Any
    freqs_cis: jnp.ndarray


def text_encoder(input_ids: jnp.ndarray, w: TextWeights) -> jnp.ndarray:
    return w.wte[input_ids]


def attn(
    x: jnp.ndarray,
    w: AttentionWeights,
    freqs_cis: jnp.ndarray,
    layer_kv_cache: jnp.ndarray,
    attn_mask: jnp.ndarray,
    n_heads: int,
    pos: int,
):
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    qkv = linear(x, w.qkv)
    q, k, v = [t.reshape(bsz, q_len, n_heads, head_dim).transpose(0, 2, 1, 3) for t in jnp.split(qkv, 3, axis=-1)]

    position_ids = jnp.arange(pos, pos + q_len)
    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_heads)

    k_, v_ = k, v
    if layer_kv_cache is not None:
        k = jnp.concatenate([layer_kv_cache[0, :, :, :pos, :], k], axis=2)
        v = jnp.concatenate([layer_kv_cache[1, :, :, :pos, :], v], axis=2)

    scale = 1.0 / jnp.sqrt(head_dim)
    scores = (q @ jnp.swapaxes(k, -2, -1)) * scale

    if attn_mask is not None:
        scores = scores + attn_mask

    attn_weights = softmax(scores, axis=-1)
    out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(bsz, q_len, d_model)
    out = linear(out, w.proj)

    return out, jnp.stack([k_, v_])


def text_decoder(
    inputs_embeds: jnp.ndarray,
    w: TextWeights,
    kv_cache: jnp.ndarray,
    attn_mask: jnp.ndarray,
    pos: int,
    config: TextConfig,
):
    hidden = inputs_embeds
    new_kv_cache = []

    for i, block in enumerate(w.blocks):
        l_in = layer_norm(hidden, block.ln)
        l_attn, new_kv = attn(
            l_in,
            block.attn,
            freqs_cis=w.freqs_cis,
            layer_kv_cache=kv_cache[i],
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            pos=pos,
        )
        l_mlp = mlp(l_in, block.mlp)
        hidden = hidden + l_attn + l_mlp
        new_kv_cache.append(new_kv)

    return hidden, jnp.stack(new_kv_cache)


def lm_head(hidden_BTC: jnp.ndarray, w: TextWeights):
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    logits = linear(hidden_BC, w.lm_head)
    return logits


def prefill(
    inputs_embeds: jnp.ndarray,
    kv_cache: jnp.ndarray,
    attn_mask: jnp.ndarray,
    pos: int,
    w: TextWeights,
    config: TextConfig,
):
    hidden, new_kv = text_decoder(inputs_embeds, w, kv_cache, attn_mask, pos, config)
    kv_cache = kv_cache.at[:, :, :, pos : pos + inputs_embeds.shape[1], :].set(new_kv)
    return hidden


def decode_one_token(
    token_emb: jnp.ndarray,
    kv_cache: jnp.ndarray,
    attn_mask: jnp.ndarray,
    pos: int,
    w: TextWeights,
    config: TextConfig,
):
    hidden, kv_cache_update = text_decoder(token_emb[None], w, kv_cache, attn_mask, pos, config)
    logits = lm_head(hidden, w)
    return logits, hidden, kv_cache_update
