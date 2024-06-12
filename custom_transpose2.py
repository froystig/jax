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

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import custom_api_util
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.tree_util import (tree_flatten, tree_leaves, tree_map,
                                tree_structure, treedef_tuple, tree_unflatten)


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

@custom_api_util.register_custom_decorator_type
class custom_transpose:
  fun: Callable
  _fun_transpose: Callable

  def __init__(self, fun: Callable):
    functools.update_wrapper(self, fun)
    self.fun = fun

  __getattr__ = custom_api_util.forward_attr

  def def_transpose(self, transpose: Callable):
    self._fun_transpose = transpose
    return transpose

  @traceback_util.api_boundary
  def __call__(self, res_arg, lin_arg):
    assert self._fun_transpose is not None, 'todo error'

    _, res_tree = tree_flatten(res_arg)
    _, lin_tree = tree_flatten(lin_arg)
    args_flat, in_tree = tree_flatten((res_arg, lin_arg))
    assert in_tree == treedef_tuple((res_tree, lin_tree))

    # TODO(danfm,frostig,mattjj): check that out_trees match

    flat_fun, out_tree = api_util.flatten_fun_nokwargs(
        lu.wrap_init(self.fun), in_tree)
    in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(self.fun, in_tree, out_tree, False, "custom_transpose")
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
        flat_fun, in_avals, debug)
    call_jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())

    flat_rule = _flatten_rule(
      lu.wrap_init(self._fun_transpose), res_tree, lin_tree, out_tree())

    out_flat = custom_transpose_p.bind(*consts, *args_flat,
                                       call=call_jaxpr,
                                       rule=flat_rule,
                                       num_consts=len(consts),
                                       res_tree=res_tree,
                                       lin_tree=lin_tree,
                                       out_tree=out_tree())
    return tree_unflatten(out_tree(), out_flat)

@lu.transformation
def _flatten_rule(res_tree, lin_tree, out_tree, *args):
  res_flat, cts_out_flat = util.split_list(args, [res_tree.num_leaves])
  assert len(cts_out_flat) == out_tree.num_leaves
  res = tree_unflatten(res_tree, res_flat)
  cts_out = tree_unflatten(out_tree, cts_out_flat)
  cts_in = yield (res, cts_out), {}
  # TODO(danfm): Handle Nones in cts_in
  cts_in_flat, cts_in_tree = tree_flatten(cts_in)
  assert cts_in_tree == lin_tree
  yield cts_in_flat

def custom_transpose_impl(*args, call, **_):
  return core.jaxpr_as_fun(call)(*args)

def custom_transpose_abstract_eval(*in_avals, call: core.ClosedJaxpr, **_):
  del in_avals
  return call.out_avals

def custom_transpose_transpose(cts, *args, call, rule, num_consts,
                               res_tree, lin_tree, out_tree):
  del call
  # call :: consts, res, lin -> out
  # rule :: res, out -> lin
  _, res_flat, lin_flat = util.split_list(
      args, [num_consts, res_tree.num_leaves])

  # res = tree_unflatten(res_tree, res_flat)
  # lin_by_rule = rule(res, tree_unflatten(out_tree, cts))
  # lin_by_rule_flat, lin_tree_by_rule = tree_flatten(lin_by_rule)
  # assert lin_tree == lin_tree_by_rule, 'todo error'

  lin_by_rule_flat = rule.call_wrapped(*res_flat, *cts)
  return [None] * res_tree.num_leaves + lin_by_rule_flat

def custom_transpose_batching(
    spmd_axis_name, axis_size, axis_name, main_type, args, in_dims, *,
    call, rule, num_consts, res_tree, lin_tree, out_tree):
  args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
          else x for x, d in zip(args, in_dims)]
  in_batched = [d is not batching.not_mapped for d in in_dims]
  batched_call, out_batched = batching.batch_jaxpr(
      call, axis_size, in_batched, False, axis_name, spmd_axis_name, main_type)
  batched_rule = batching.batch(rule, axis_name, axis_size, in_dims,
                                out_batched, main_type, spmd_axis_name)
  batched_outs = custom_transpose_p.bind(
      *args,
      call=batched_call,
      rule=batched_rule,
      num_consts=num_consts,
      res_tree=res_tree,
      lin_tree=lin_tree,
      out_tree=out_tree,
  )
  return batched_outs, [0] * out_tree.num_leaves

custom_transpose_p = core.Primitive('custom_transpose_call')
custom_transpose_p.multiple_results = True
custom_transpose_p.def_impl(custom_transpose_impl)
custom_transpose_p.def_abstract_eval(custom_transpose_abstract_eval)
ad.primitive_transposes[custom_transpose_p] = custom_transpose_transpose
batching.spmd_axis_primitive_batchers[custom_transpose_p] = \
    custom_transpose_batching
batching.axis_primitive_batchers[custom_transpose_p] = \
    functools.partial(custom_transpose_batching, None)

def test():
  import jax
  import jax.numpy as jnp

  def transpose_unary(f, x_example):
    def transposed(y):
      x, = jax.linear_transpose(f, x_example)(y)
      return x
    return transposed

  T = lambda f: transpose_unary(f, 0.)

  @custom_transpose
  def f(_, z):
    return 2. * z

  @f.def_transpose
  def ft(_, z):
    return 3. * z

  f = functools.partial(f, ())
  print(jax.make_jaxpr(jax.vmap(f))(jnp.linspace(0, 1, 5)))
  assert 0

  print(f(1.))              # 2.
  print(T(f)(1.))           # 3.
  print(T(T(f))(1.))        # 3.
  print(T(T(T(f)))(1.))     # 3.
  print(T(T(T(T(f))))(1.))  # 3. ...
  print(jax.make_jaxpr(f)(1.))
  print(jax.make_jaxpr(T(f))(1.))

  @custom_transpose
  def f(c, z):
    return c * z

  @f.def_transpose
  def ft(c, z):
    return f(c + 1., z)

  g = functools.partial(f, 1.)
  print(g(1.))              # 1.
  print(T(g)(1.))           # 2.
  print(T(T(g))(1.))        # 3.
  print(T(T(T(g)))(1.))     # 4.
  print(T(T(T(T(g))))(1.))  # 5. ...
  print(jax.make_jaxpr(g)(1.))
  print(jax.make_jaxpr(T(g))(1.))
  print(jax.make_jaxpr(T(T(g)))(1.))

if __name__ == '__main__':
  # import ipdb, sys, traceback
  # def info(t, v, i):
  #   traceback.print_exception(t, v, i)
  #   ipdb.pm()
  # sys.excepthook = info
  test()
