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
from collections.abc import Callable, Iterable, Iterator, Sequence
import functools
from functools import partial
from typing import Any

from jax._src import core
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src import linear_util as lu

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

physicalize_rules = {}


def default_physicalize_rule(primitive, in_avals, out_avals, *args, **params):
  del in_avals, out_avals
  return primitive.bind(*args, **params)


def physicalize_jaxpr(jaxpr: core.ClosedJaxpr) -> core.ClosedJaxpr:
  """Replaces all extended dtypes with physical types in a jaxpr."""
  fun = partial(physicalize_jaxpr_interp, jaxpr.jaxpr, jaxpr.consts)
  wrapped_fn = lu.wrap_init(fun)
  physical_in_avals = [core.physical_aval(aval) for aval in jaxpr.in_avals]
  new_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      wrapped_fn, physical_in_avals)
  return core.ClosedJaxpr(new_jaxpr, consts)


def physicalize_jaxpr_interp(jaxpr: core.Jaxpr,
                             consts: Sequence[core.Value],
                             *physical_in_args: core.Value) -> list[Any]:
  env: dict[core.Var, Any] = {}

  last_used = core.last_used(jaxpr)

  def read_env(var: core.Atom):
    if isinstance(var, core.Literal):
      return var.val
    return env[var]

  def write_env(var: core.Var, val: Any):
    env[var] = val

  map(write_env, jaxpr.constvars, consts)
  assert len(jaxpr.invars) == len(physical_in_args), (
      f"Length mismatch: {jaxpr.invars} != {physical_in_args}")
  map(write_env, jaxpr.invars, physical_in_args)

  for eqn in jaxpr.eqns:
    physical_invals = map(read_env, eqn.invars)
    physicalize_rule = physicalize_rules.get(
        eqn.primitive, functools.partial(default_physicalize_rule, eqn.primitive)
    )
    name_stack = (
        source_info_util.current_name_stack() + eqn.source_info.name_stack
    )
    with source_info_util.user_context(
        eqn.source_info.traceback, name_stack=name_stack
    ):
      has_extended = any(dtypes.issubdtype(invar.aval.dtype, dtypes.extended) for invar in eqn.invars)
      if has_extended:
        physical_outvals = physicalize_rule(
            [x.aval for x in eqn.invars], [x.aval for x in eqn.outvars],
            *physical_invals, **eqn.params
        )
      else:
        physical_outvals = eqn.primitive.bind(*physical_invals, **eqn.params)
    if eqn.primitive.multiple_results:
      assert len(physical_outvals) == len(eqn.outvars)
      map(write_env, eqn.outvars, physical_outvals)
    else:
      write_env(eqn.outvars[0], physical_outvals)
    core.clean_up_dead_vars(eqn, env, last_used)

  return map(read_env, jaxpr.outvars)

