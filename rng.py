from typing import Sequence

import jax
import jax.extend as jex

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
