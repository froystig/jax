from __future__ import annotations

import ipdb, sys, traceback
def info(t, v, i):
  traceback.print_exception(t, v, i)
  ipdb.pm()
sys.excepthook = info

from typing import Sequence

import jax
import jax.extend as jex
from jax import lax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
from jax import numpy as jnp
from jax import random as jax_api_random
from jax._src import blocked_sampler
from jax._src import typing
from jax._src.pallas.mosaic.primitives import prng_seed
from jax._src.pallas.mosaic.primitives import prng_random_bits
from jax._src.pallas import primitives
from jax._src import prng as jax_prng


def pallas_hash(key, msg):
  block_size = (8, 128)
  def kernel(key_ref, msg_ref, o_ref):
    prng_seed(key_ref[0, 0], msg_ref[0, 0])
    for _ in range(10):
      prng_random_bits(block_size)
    o_ref[...] = prng_random_bits(block_size)
  out = jax.ShapeDtypeStruct(block_size, jnp.int32)
  key = jnp.array([[key]], dtype=jnp.int32)
  msg = jnp.array([[msg]], dtype=jnp.int32)
  result = pl.pallas_call(
      kernel,
      in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM),
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM)],
      out_shape=out)(key, msg)
  return result


def seed(seed: jax.Array) -> jax.Array:
  return jnp.array([seed], dtype=jnp.uint32)

def iota_hash(key: jax.Array, shape) -> jax.Array:
  _, counts = jax._src.prng.iota_2x32_shape(shape)
  bitss = jax.vmap(pallas_hash, in_axes=(None, 0))(key, counts)
  return bitss[..., 0, 0]  # TODO: wasteful

def split(key: jax.Array, shape) -> jax.Array:
  return iota_hash(key, shape)

def fold_in(key: jax.Array, data: jax.Array) -> jax.Array:
  bits = pallas_hash(key, data)
  return bits[0, 0]

def random_bits(key: jax.Array, bit_width: int, shape: Sequence[int]
                ) -> jax.Array:
  if bit_width != 32:
    raise NotImplementedError(f'bit_width {bit_width}')
  return iota_hash(key, shape)

palrng = jex.random.define_prng_impl(key_shape=(1,), seed=seed, split=split,
                                     random_bits=random_bits, fold_in=fold_in,
                                     name='jjf', tag='jjf')

def main(*_):
  def f(key):
    k1, k2 = jax.random.split(key)
    x1 = jax.random.uniform(k1, (3,))
    x2 = jax.random.normal(k2, (4, 5))
    return x1, x2

  key = jax.random.key(42, impl=palrng)
  print(f(key))
  print(jax.jit(f)(key))
  print(jax.make_jaxpr(f)(key))


if __name__ == '__main__':
  import sys
  main(sys.argv[1:])
