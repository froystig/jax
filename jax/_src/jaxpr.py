# Copyright 2023 The JAX Authors.
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

import collections
import itertools as it
from functools import total_ordering
from typing import (
    Any, Callable, ClassVar, DefaultDict, Dict, FrozenSet, Generator, Generic,
    Hashable, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set,
    Tuple, Type, TypeVar, Union)
import warnings

import jax
from jax._src import effects
import jax._src.pretty_printer as pp


Effect = effects.Effect
Effects = effects.Effects
EffectTypeSet = effects.EffectTypeSet
no_effects: Effects = effects.no_effects


class Jaxpr:
  __slots__ = ['__weakref__', '_consts', '_constvars', '_invars',
               '_outvars', '_eqns', '_effects', '_debug_info']

  _consts: List[Any]
  _constvars: List[Var]
  _invars: List[Var]
  _outvars: List[Atom]
  _eqns: List[JaxprEqn]
  _effects: Effects
  _debug_info: Optional[JaxprDebugInfo]

  consts = property(lambda self: self._consts)
  constvars = property(lambda self: self._constvars)
  invars = property(lambda self: self._invars)
  outvars = property(lambda self: self._outvars)
  eqns = property(lambda self: self._eqns)
  effects = property(lambda self: self._effects)
  debug_info = property(lambda self: self._debug_info)

  @staticmethod
  def _from_closed_jaxpr(closed_jaxpr: jax._src.core.ClosedJaxpr) -> Jaxpr:
    return Jaxpr._from_open_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)

  @staticmethod
  def _from_open_jaxpr(
      open_jaxpr: jax._src.core.Jaxpr, consts: Sequence[Any]) -> Jaxpr:
    return Jaxpr(
        consts, open_jaxpr.constvars, open_jaxpr.invars, open_jaxpr.outvars,
        open_jaxpr.eqns, open_jaxpr.effects, open_jaxpr.debug_info)

  def __init__(self, consts: Sequence[Any], constvars: Sequence[Var],
               invars: Sequence[Var], outvars: Sequence[Atom],
               eqns: Sequence[JaxprEqn], effects: Effects = no_effects,
               debug_info: Optional[JaxprDebugInfo] = None):
    """
    Args:
      consts: list of arrays to bind to constvar inputs.
      constvars: list of variables introduced for constants. Array constants are
        replaced with such variables while scalar constants are kept inline.
      invars: list of input variables. Together, `constvars` and `invars` are
        the inputs to the Jaxpr.
      outvars: list of output atoms.
      eqns: list of equations.
      effects: set of effects. The effects on a jaxpr are a superset of the
        union of the effects for each equation.
      debug_info: optional JaxprDebugInfo.
    """
    self._consts = list(consts)
    self._constvars = list(constvars)
    self._invars = list(invars)
    self._outvars = list(outvars)
    self._eqns = list(eqns)
    self._effects = effects
    self._debug_info = debug_info
    assert (not debug_info or len(debug_info.arg_names) == len(invars) and
            len(debug_info.result_paths) == len(outvars))

  @property
  def jaxpr(self):
    warnings.warn('deprecated `jaxpr` attribute is a self-reference',
                  DeprecationWarning)
    return self

  @property
  def in_avals(self):
    return [v.aval for v in self.invars]

  @property
  def out_avals(self):
    return [v.aval for v in self.outvars]

  @property
  def literals(self):
    return self.consts  # backwards compatible alias

  def map_jaxpr(self, f):
    return Jaxpr._from_open_jaxpr(f(self), self.consts)

  def replace(self, *, consts=None, constvars=None, invars=None, outvars=None,
              eqns=None, effects=None, debug_info=None):
    consts = self.consts if consts is None else consts
    constvars = self.constvars if constvars is None else constvars
    invars =  self.invars if invars is None else invars
    outvars = self.outvars if outvars is None else outvars
    eqns = self.eqns if eqns is None else eqns
    effects = self.effects if effects is None else effects
    return Jaxpr(
        consts=consts, constvars=constvars, invars=invars, outvars=outvars,
        eqns=eqns, effects=effects, debug_info=debug_info)

  def pretty_print(self, *,
                   source_info: bool = False,
                   print_shapes: bool = True,
                   name_stack: bool = False,
                   custom_pp_eqn_rules: bool = True,
                   print_effects: bool = False,
                   **kwargs):
    settings = jax._src.core.JaxprPpSettings(source_info=source_info,
                               print_shapes=print_shapes,
                               print_effects=print_effects,
                               name_stack=name_stack,
                               custom_pp_eqn_rules=custom_pp_eqn_rules)
    doc = jax._src.core.pp_jaxpr(
        self, jax._src.core.JaxprPpContext(), settings).format(**kwargs)
    return doc.format(**kwargs)

  def __str__(self):
    ctx = jax._src.core.JaxprPpContext()
    settings = jax._src.core.JaxprPpSettings()
    return str(jax._src.core.pp_jaxpr(self, ctx, settings))
  
  __repr__ = __str__

  def _repr_pretty_(self, p, cycle):
    return p.text(self.pretty_print(use_color=True))
