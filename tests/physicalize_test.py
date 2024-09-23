# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Sequence
import unittest
from absl.testing import absltest

import jax
from jax import core
from jax import numpy as jnp
from jax import random
from jax import lax
import jax._src.interpreters.physical as physical
import jax._src.test_util as jtu
from jax._src import ad_util

jax.config.parse_flags_with_absl()


Shape = Sequence[int]


def find_primitive(jaxpr: core.ClosedJaxpr, 
                  primitive: core.Primitive):
  for eqn in jaxpr.eqns:
    if eqn.primitive == primitive:
      return eqn
  raise ValueError(f"Primitive not found: {primitive.name}")


class PhysicalizeTest(unittest.TestCase):

  def assert_jaxpr(self,
                jaxpr: core.ClosedJaxpr,
                primitive: core.Primitive,
                expected_in_shapes: Sequence[Shape],
                expected_in_dtypes: Sequence[Any],
                expected_out_shapes: Sequence[Shape],
                expected_out_dtypes: Sequence[Any],
                ):
    broadcast_op = find_primitive(jaxpr, primitive)
    for invar, expected_shape, expected_dt in zip(
      broadcast_op.invars, expected_in_shapes, expected_in_dtypes):
      in_aval = invar.aval
      self.assertEqual(in_aval.shape, expected_shape)
      self.assertEqual(in_aval.dtype, expected_dt)
    for outvar, expected_shape, expected_dt in zip(
      broadcast_op.outvars, expected_out_shapes, expected_out_dtypes):
      out_aval = outvar.aval
      self.assertEqual(out_aval.shape, expected_shape)
      self.assertEqual(out_aval.dtype, expected_dt)

  def test_broadcast_in_dim(self):
    def fun(k):
      return lax.broadcast_in_dim(k, (2, 2), broadcast_dimensions=(0,))
    k = random.split(random.key(0), 2)
    result = jax.jit(fun)(k)
    self.assertEqual(result.shape, (2, 2))

    traced = jax.jit(fun).trace(k)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    phys_aval = phys_jaxpr.jaxpr.invars[0].aval
    self.assertEqual(phys_aval.shape, (2, 2))
    self.assertEqual(phys_aval.dtype, jnp.uint32)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.broadcast_in_dim_p,
                    expected_in_shapes=[(2, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(2, 2, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_slice(self):
    def fun(k):
      return lax.slice(k, (0, 0), (4, 1))
    k = random.split(random.key(0), (4, 4))
    result = jax.jit(fun)(k)
    self.assertEqual(result.shape, (4, 1))
    traced = jax.jit(fun).trace(k)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.slice_p,
                    expected_in_shapes=[(4, 4, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(4, 1, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_dynamic_slice(self):
    def fun(k, starts):
      return lax.dynamic_slice(k, starts, (2, 3))
    k = random.split(random.key(0), (4, 4))
    result = jax.jit(fun)(k, (0, 1))
    self.assertEqual(result.shape, (2, 3))
    traced = jax.jit(fun).trace(k, (0, 1))
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.dynamic_slice_p,
                    expected_in_shapes=[(4, 4, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(2, 3, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_dynamic_update_slice(self):
    def fun(k, updates, starts):
      return lax.dynamic_update_slice(k, updates, starts)
    k = random.split(random.key(0), (4, 4))
    updates = random.split(random.key(1), (2, 3))
    result = jax.jit(fun)(k, updates, (0, 1))
    self.assertEqual(result.shape, (4, 4))
    traced = jax.jit(fun).trace(k, updates, (0, 1))
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.dynamic_update_slice_p,
                    expected_in_shapes=[(4, 4, 2), (2, 3, 2)],
                    expected_in_dtypes=[jnp.uint32, jnp.uint32],
                    expected_out_shapes=[(4, 4, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_add_jaxvals(self):
    self.skipTest('PRNGKey has no add implementation.')
    def fun(x, y):
      return ad_util.add_jaxvals(x, y)
    x = random.split(random.key(0), (4, 4))
    y = random.split(random.key(1), (4, 4))
    result = jax.jit(fun)(x, y)
    self.assertEqual(result.shape, (4, 4))
    traced = jax.jit(fun).trace(x, y)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.dynamic_update_slice_p,
                    expected_in_shapes=[(4, 4, 2), (4, 4, 2)],
                    expected_in_dtypes=[jnp.uint32, jnp.uint32],
                    expected_out_shapes=[(4, 4, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_reshape(self):
    def fun(k):
      return jnp.reshape(k, (12, 2))
    k = random.split(random.key(0), (6, 4))
    result = jax.jit(fun)(k)
    self.assertEqual(result.shape, (12, 2))
    traced = jax.jit(fun).trace(k)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.reshape_p,
                    expected_in_shapes=[(6, 4, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(12, 2, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())

