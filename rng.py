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
  result = pl.pallas_call(kernel, 
                          in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM),
                                    pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM)] ,
                          out_shape=out)(key, msg)
  return result


def seed(seed: jax.Array) -> jax.Array:
  pass

def split(key: jax.Array, shape) -> jax.Array:
  pass

def fold_in(key: jax.Array, data: jax.Array) -> jax.Array:
  pass

def random_bits(key: jax.Array, bit_width: int, shape: Sequence[int]
                ) -> jax.Array:
  pass

jex.random.define_prng_impl(key_shape=(1,), seed=seed, split=split,
                            random_bits=random_bits, fold_in=fold_in,
                            name='jjf', tag='jjf')

if __name__ == '__main__':
  pass
