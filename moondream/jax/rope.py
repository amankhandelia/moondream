import jax
import jax.numpy as jnp


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=dtype) / (dim // 2)))
    t = jnp.arange(end, dtype=dtype)[:, None]
    freqs = t * freqs[None, :]
    freqs_complex = jnp.exp(1j * freqs)
    return jnp.stack([freqs_complex.real, freqs_complex.imag], axis=-1)


def apply_rotary_emb(
    x: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    position_ids: jnp.ndarray,
    num_heads: int,
    rot_dim: int = 32,
    interleave: bool = False,
) -> jnp.ndarray:
    assert rot_dim == freqs_cis.shape[-2] * 2
    assert num_heads == x.shape[1]

    x_rot, x_pass = jax.lax.slice_in_dim(x, 0, rot_dim, axis=-1), jax.lax.slice_in_dim(x, rot_dim, None, axis=-1)

    if interleave:
        x_rot = x_rot.astype(jnp.float32)
        xq_r = jax.lax.slice_in_dim(x_rot.reshape(*x_rot.shape[:-1], -1, 2), 0, 1, axis=-1).squeeze(-1)
        xq_i = jax.lax.slice_in_dim(x_rot.reshape(*x_rot.shape[:-1], -1, 2), 1, 2, axis=-1).squeeze(-1)
    else:
        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = jax.lax.slice_in_dim(x_rot, 0, d_q, axis=-1), jax.lax.slice_in_dim(x_rot, d_q, None, axis=-1)

    # Add batch and head dimensions to freqs
    freqs_cos = jnp.expand_dims(jnp.expand_dims(freqs_cis[position_ids, :, 0], axis=0), axis=0)
    freqs_sin = jnp.expand_dims(jnp.expand_dims(freqs_cis[position_ids, :, 1], axis=0), axis=0)

    # Complex multiplication
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xq_out = jnp.stack((xq_out_r, xq_out_i), axis=-1).reshape(*xq_out_r.shape[:-1], -1)

    return jnp.concatenate([xq_out.astype(x.dtype), x_pass], axis=-1)
