from __future__ import annotations
import collections
from dataclasses import dataclass

from typing import Any, Dict, List, Set

from jax import core
from jax.interpreters import batching

# TODO(frostig,sharadmv): factor to primitives submodule with ability
# to "get all primitives"
@dataclass(frozen=True, repr=False)
class Primitive:
  _prim: core.Primitive

  def bind(self, *args, **params) -> List[Any]:
    from jax._src.extend import jaxpr
    params = jaxpr.convert_params_to_core(params, self._prim.call_primitive)
    subfuns, params = self._prim.get_bind_params(params)
    ans = self._prim.bind(*subfuns, *args, **params)
    if not self._prim.multiple_results:
      ans = [ans]
    return ans

  @property
  def name(self): return self._prim.name

  def __repr__(self):
    return f"Primitive(name='{self.name}')"

_all_primitives: Dict[str, Set[Primitive]] = collections.defaultdict(set)

def register_primitive(name: str, primitive: Primitive):
  _all_primitives[name].add(primitive)

def primitives_by_name(name: str) -> List[Primitive]:
  return _all_primitives[name]

def register_core_primitive(prim: core.Primitive):
  register_primitive(prim.name, Primitive(prim))

for prim in core.all_primitives:
  register_core_primitive(prim)
