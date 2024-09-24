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

import math
from typing import Any, Sequence
import unittest
import operator as op
from absl.testing import absltest

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from jax import lax
from jax._src import ad_util
from jax._src import core
from jax._src import dtypes
import jax._src.interpreters.physical as physical
import jax._src.test_util as jtu
from jax._src.interpreters import xla
from jax._src.interpreters import pxla
from jax._src.lax import lax as lax_internal
from jax._src import tree_util as tree_util_internal
from jax._src.sharding_impls import (
    NamedSharding, PmapSharding, physical_sharding, logical_sharding)

jax.config.parse_flags_with_absl()


Shape = Sequence[int]


class test_extended_type(dtypes.extended):
  pass

@staticmethod
class TestTyRules:
  @staticmethod
  def add(dtype, x, y):
    del dtype
    return x + y
  
  @staticmethod
  def full(shape, fill_value, dtype):
    raise ValueError('%s, %s' % (fill_value, dtype))
    #physical_shape = (*shape, *dtype.element_shape)
    #return lax.full(physical_shape, fill_value, fill_value.dtype)
  
  @staticmethod
  def physical_element_aval(dtype) -> core.ShapedArray:
    return core.ShapedArray(dtype.element_shape, jnp.dtype('uint32'))
  
  @staticmethod
  def global_sharded_result_handler(aval, out_sharding, committed):
    phys_aval = core.physical_aval(aval)
    phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]

    phys_sharding = physical_sharding(aval, out_sharding)
    phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)
    def handler(bufs):
      return TestExtendedArray(phys_handler(bufs))
    return handler
  
  @staticmethod
  def convert_from(my_dtype, other_dtype) -> bool:
    if other_dtype == jnp.uint32 or other_dtype == jnp.float32:
      return True
    return False

  @staticmethod
  def convert_to(other_dtype, my_dtype) -> bool:
    if other_dtype == jnp.uint32 or other_dtype == jnp.float32:
      return True
    return False


class TestTy(dtypes.ExtendedDType):
  _rules = TestTyRules
  type = test_extended_type
  element_shape = (7, 3)
  name = 'TestTy'

  def __str__(self):
    return self.name

TEST_TY = TestTy()

class TestExtendedArray(jax.Array):
  device = property(op.attrgetter('_base_array.device'))
  is_fully_addressable = property(op.attrgetter('_base_array.is_fully_addressable'))  # type: ignore[assignment]
  is_fully_replicated = property(op.attrgetter('_base_array.is_fully_replicated'))  # type: ignore[assignment]

  def __init__(self, data):
    self._base_array = data
  
  def copy_to_host_async(self):
    _ = self._base_array.copy_to_host_async()
  
  @property
  def aval(self):
    return core.ShapedArray(self.shape, self.dtype)

  @property
  def shape(self):
    return self._base_array.shape[:-len(self.dtype.element_shape)]
  
  @property
  def size(self):
    return math.prod(self.shape)

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def dtype(self):
    return TEST_TY
  
  @property
  def sharding(self):
    return logical_sharding(self.aval, self._base_array.sharding)
  
  def addressable_data(self, index: int) -> 'TestExtendedArray':
    return TestExtendedArray(self._base_array.addressable_data(index))

  @property
  def addressable_shards(self) -> list[Any]:
    return [
        type(s)(
            device=s._device,
            sharding=s._sharding,
            global_shape=s._global_shape,
            data=TestExtendedArray(s._data),
        )
        for s in self._base_array.addressable_shards
    ]
  
  @property
  def global_shards(self) -> list[Any]:
    return [
        type(s)(
            device=s._device,
            sharding=s._sharding,
            global_shape=s._global_shape,
            data=TestExtendedArray(s._data),
        )
        for s in self._base_array.global_shards
    ]

  def __str__(self):
    return f"TestExtendedArray({str(self._base_array)})"


def test_extended_flatten(x):
  return (x._base_array,), None

def test_extended_unflatten(impl, children):
  base_array, = children
  return TestExtendedArray(base_array)

tree_util_internal.dispatch_registry.register_node(
    TestExtendedArray, test_extended_flatten, test_extended_unflatten)
core.pytype_aval_mappings[TestExtendedArray] = lambda x: x.aval
xla.pytype_aval_mappings[TestExtendedArray] = lambda x: x.aval
xla.canonicalize_dtype_handlers[TestExtendedArray] = lambda x: x


def test_array_shard_arg_handler(xs: Sequence[TestExtendedArray], shardings, layouts):
  arrs = [x._base_array for x in xs]
  phys_shardings = [physical_sharding(x.aval, sharding)
                    for x, sharding in zip(xs, shardings)]
  return pxla.shard_args(phys_shardings, layouts, arrs)

pxla.shard_arg_handlers[TestExtendedArray] = test_array_shard_arg_handler



def find_primitive(jaxpr: core.Jaxpr, 
                  primitive: core.Primitive):
  for eqn in jaxpr.eqns:
    if eqn.primitive == primitive:
      return eqn
    if 'jaxpr' in eqn.params:
      try:
        return find_primitive(eqn.params['jaxpr'], primitive)
      except ValueError:
        continue
  raise ValueError(f"Primitive not found: {primitive.name}\n\tin JAXPR: {jaxpr}")


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
    def fun(x, y):
      return ad_util.add_jaxvals(x, y)
    x = TestExtendedArray(jnp.ones((8, 7, 3), dtype=jnp.int32))
    y = TestExtendedArray(jnp.zeros((8, 7, 3), dtype=jnp.int32))
    result = jax.jit(fun)(x, y)
    self.assertEqual(result.shape, (8,))
    traced = jax.jit(fun).trace(x, y)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.add_p,
                    expected_in_shapes=[(8, 7, 3), (8, 7, 3)],
                    expected_in_dtypes=[jnp.uint32, jnp.uint32],
                    expected_out_shapes=[(8, 7, 3)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_reshape(self):
    with self.subTest('static'):
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
    # TODO(justinfu): Test dynamic.

  def test_convert_element_type(self):
    def fun(x, y):
      x = lax.convert_element_type(x, core.bint(5))
      x = lax.convert_element_type(x, jnp.int32)
      y = lax.convert_element_type(y, core.bint(8))
      y = lax.convert_element_type(y, jnp.int32)
      return x + y
    result = jax.jit(fun)(2, 3)
    self.assertEqual(result, 5)
    traced = jax.jit(fun).trace(2, 3)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.convert_element_type_p,
                    expected_in_shapes=[()],
                    expected_in_dtypes=[jnp.int32],
                    expected_out_shapes=[()],
                    expected_out_dtypes=[jnp.int32],
                    )

  def test_empty(self):
    def fun():
      return lax_internal.empty(random.key(0).dtype)
    result = jax.jit(fun)()
    self.assertEqual(result.shape, ())
    np.testing.assert_array_equal(jax.random.key_data(result),
                                  jnp.zeros((2,)))

  def test_gather(self):
    def fun(k, starts):
      return lax.gather(k, starts,
                        lax.GatherDimensionNumbers((0, 1), (), (0, 1)),
                        (2, 3))
    k = random.split(random.key(0), (4, 4))
    starts = jnp.array([0, 1])
    result = jax.jit(fun)(k, starts)
    np.testing.assert_array_equal(jax.random.key_data(result),
                                  jax.random.key_data(k[0:2, 1:4]))
    traced = jax.jit(fun).trace(k, starts)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.gather_p,
                    expected_in_shapes=[(4, 4, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(2, 3, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_scatter(self):
    def fun(k, indices, updates):
      return lax.scatter(k,
                         indices,
                         updates,
                        lax.ScatterDimensionNumbers((0, 1), (), (0, 1)))
    k = random.split(random.key(0), (4, 4))
    updates = random.split(random.key(1), (2, 3))
    indices = jnp.array([0, 1])
    result = jax.jit(fun)(k, indices, updates)
    np.testing.assert_array_equal(jax.random.key_data(result),
                                  jax.random.key_data(k).at[0:2, 1:4].set(
                                    jax.random.key_data(updates)))
    traced = jax.jit(fun).trace(k, indices, updates)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.scatter_p,
                    expected_in_shapes=[(4, 4, 2), (2,), (2, 3, 2)],
                    expected_in_dtypes=[jnp.uint32, jnp.int32, jnp.uint32],
                    expected_out_shapes=[(4, 4, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_compare(self):
    def fun(x, y):
      return x == y
    x = random.split(random.key(0), (2, 3))
    y = random.split(random.key(1), (2, 3))
    result = jax.jit(fun)(x, y)
    np.testing.assert_array_equal(result,
                                  jnp.zeros((2, 3)))
    traced = jax.jit(fun).trace(x, y)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.eq_p,
                    expected_in_shapes=[(2, 3, 2), (2, 3, 2)],
                    expected_in_dtypes=[jnp.uint32, jnp.uint32],
                    expected_out_shapes=[(2, 3, 2)],
                    expected_out_dtypes=[jnp.bool_],
                    )

  def test_select(self):
    def fun(which, x, y):
      return lax.select(which, x, y)

    which = jnp.array([True, False, True])
    x = random.split(random.key(0), (3,))
    y = random.split(random.key(1), (3,))
    result = jax.jit(fun)(which, x, y)
    np.testing.assert_array_equal(result,
                                  jnp.stack([x[0], y[1], x[2]]))
    traced = jax.jit(fun).trace(which, x, y)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.select_n_p,
                    expected_in_shapes=[(3, 2), (3, 2), (3, 2)],
                    expected_in_dtypes=[jnp.bool_, jnp.uint32, jnp.uint32],
                    expected_out_shapes=[(3, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_transpose(self):
    def fun(x):
      return jnp.transpose(x, (1, 2, 0))
    x = random.split(random.key(0), (2, 3, 7))
    result = jax.jit(fun)(x)
    self.assertEqual(result.shape, (3, 7, 2))
    traced = jax.jit(fun).trace(x)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.transpose_p,
                    expected_in_shapes=[(2, 3, 7, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(3, 7, 2, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_reshape(self):
    def fun(x):
      return jnp.reshape(x, (6, 7))
    x = random.split(random.key(0), (2, 3, 7))
    result = jax.jit(fun)(x)
    self.assertEqual(result.shape, (6, 7))
    traced = jax.jit(fun).trace(x)
    phys_jaxpr = physical.physicalize_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr, 
                    lax.reshape_p,
                    expected_in_shapes=[(2, 3, 7, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(6, 7, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  # TODO: search for more opaque lowering, core.physical_aval, dtypes.extended
  # transformations - shard_map, jax2tf, pjit, control_flow.loops
  # TODO: add jax2tf tests

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())

