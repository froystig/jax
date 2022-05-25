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
from __future__ import annotations

import functools
from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import core

from jax._src import source_info_util
from jax._src import util
from jax._src.extend import primitives

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@dataclass(frozen=True)
class Type:
  shape: Tuple[int]
  named_shape: Dict[str, int]
  dtype: jnp.dtype


def type_from_aval(aval: core.AbstractValue):
  assert isinstance(aval, core.ShapedArray), aval
  return Type(aval.shape, aval.named_shape, aval.dtype)

def from_core_atom(atom: core.Atom) -> Atom:
  if isinstance(atom, core.Literal):
    return Literal(type_from_aval(atom.aval), atom.val)
  elif isinstance(atom, core.Var):
    return Var(atom, type_from_aval(atom.aval))
  assert False

@dataclass(frozen=True)
class Var:
  _var: core.Var = field(repr=False)
  type: Type = field(hash=False, compare=False)

  def _to_core(self) -> core.Var:
    return self._var

@dataclass(frozen=True)
class Literal:
  type: Type
  value: Any

  def _to_core(self) -> core.Literal:
    return core.Literal(self.value, self.type)

Atom = Union[Var, Literal]

@dataclass(frozen=True)
class Equation:
  primitive: primitives.Primitive
  params: Dict[str, Any]
  inputs: List[Atom]
  outputs: List[Var]

  def pretty_print(self):
    return str(core.pp_eqn(
        core.JaxprEqn([v._to_core() for v in self.inputs],
                      [v._to_core() for v in self.outputs],
                      self.primitive._prim, self.params, core.no_effects,
                      source_info_util.new_source_info()),
        core.JaxprPpContext(), core.JaxprPpSettings())).rstrip()


def from_core_eqn(eqn: core.JaxprEqn) -> Equation:
  invars = map(from_core_atom, eqn.invars)
  outvars  = map(from_core_atom, eqn.outvars)
  return Equation(primitives.Primitive(eqn.primitive), dict(eqn.params), invars,
                  outvars)


@dataclass(frozen=True, repr=False)
class Jaxpr:
  _closed_jaxpr: core.ClosedJaxpr

  @property
  def consts(self) -> Sequence[Tuple[Var, Any]]:
    return [(Var(v, type_from_aval(v.aval)), const)
            for v, const in zip(self._closed_jaxpr.jaxpr.constvars,
                                self._closed_jaxpr.consts)]

  @property
  def inputs(self) -> List[Var]:
    return [Var(v, type_from_aval(v.aval))
            for v in self._closed_jaxpr.jaxpr.invars]

  @property
  def outputs(self) -> List[Atom]:
    return [from_core_atom(v)
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
