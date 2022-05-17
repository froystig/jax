# Copyright 2022 Google LLC
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

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import core

from jax._src import util

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


# TODO(frostig,sharadmv): factor to primitives submodule with ability
# to "get all primitives"
@dataclass(frozen=True, repr=False)
class Primitive:
  _prim: core.Primitive

  @property
  def name(self): return self._prim.name

  def __repr__(self):
    return f"Primitive(name='{self.name}')"


@dataclass(frozen=True)
class Type:
  shape: Tuple[int]
  named_shape: Dict[str, int]
  dtype: jnp.dtype


def type_from_aval(aval: core.AbstractValue):
  assert isinstance(aval, core.ShapedArray), aval
  return Type(aval.shape, aval.named_shape, aval.dtype)


@dataclass(frozen=True)
class Var:
  _var: core.Var = field(repr=False)
  type: Type = field(hash=False, compare=False)


@dataclass(frozen=True)
class Equation:
  primitive: Primitive
  params: Dict[str, Any]
  invars: List[Var]
  outvars: List[Var]

  def pretty_print(self):
    return str(core.pp_eqn(
        core.JaxprEqn([v._var for v in self.invars],
                      [v._var for v in self.outvars],
                      self.primitive._prim, self.params, core.no_effects, None),
        core.JaxprPpContext(), core.JaxprPpSettings())).rstrip()


def from_core_eqn(eqn: core.JaxprEqn):
  invars  = [Var(v, type_from_aval(v.aval)) for v in eqn.invars]
  outvars = [Var(v, type_from_aval(v.aval)) for v in eqn.outvars]
  return Equation(Primitive(eqn.primitive), dict(eqn.params), invars, outvars)


@dataclass(frozen=True, repr=False)
class Jaxpr:
  _closed_jaxpr: core.ClosedJaxpr

  def consts(self) -> Sequence[Tuple[Var, Any]]:
    return [(Var(v, type_from_aval(v.aval)), const)
            for v, const in zip(self._closed_jaxpr.jaxpr.constvars,
                                self._closed_jaxpr.consts)]

  def inputs(self) -> List[Var]:
    return [Var(v, type_from_aval(v.aval))
            for v in self._closed_jaxpr.jaxpr.invars]

  def outputs(self) -> List[Var]:
    return [Var(v, type_from_aval(v.aval))
            for v in self._closed_jaxpr.jaxpr.outvars]

  def equations(self) -> List[Equation]:
    return map(from_core_eqn, self._closed_jaxpr.jaxpr.eqns)

  def __repr__(self):
    return repr(self._closed_jaxpr)


def make_jaxpr(f, **kwargs):
  core_jaxpr_maker = jax.make_jaxpr(f, **kwargs)
  @functools.wraps(f)
  def jaxpr_maker(*args, **kwargs):
    return Jaxpr(core_jaxpr_maker(*args, **kwargs))
  return jaxpr_maker
