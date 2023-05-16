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

from typing import (
    Any, Callable, ClassVar, DefaultDict, Dict, FrozenSet, Generator, Generic,
    Hashable, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set,
    Tuple, Type, TypeVar, Union)
import jax


Effect = effects.Effect
Effects = effects.Effects
EffectTypeSet = effects.EffectTypeSet
no_effects: Effects = effects.no_effects


class JaxprDebugInfo(NamedTuple):
  traced_for: str     # e.g. 'jit', 'scan', etc
  func_src_info: str  # e.g. f'{fun.__name__} at {filename}:{lineno}'
  arg_names: Tuple[Optional[str], ...]     # e.g. ('args[0]', ... )
  result_paths: Tuple[Optional[str], ...]  # e.g. ('[0]', '[1]', ...)


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
    return Jaxpr.from_open_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts)

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
    consts = self.consts is consts is None else consts
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
    settings = JaxprPpSettings(source_info=source_info,
                               print_shapes=print_shapes,
                               print_effects=print_effects,
                               name_stack=name_stack,
                               custom_pp_eqn_rules=custom_pp_eqn_rules)
    doc = pp_jaxpr(self, JaxprPpContext(), settings).format(**kwargs)
    return doc.format(**kwargs)

  def __str__(self):
    return str(pp_jaxpr(self, JaxprPpContext(), JaxprPpSettings()))
  
  __repr__ = __str__

  def _repr_pretty_(self, p, cycle):
    return p.text(self.pretty_print(use_color=True))

def join_effects(*effects: Effects) -> Effects:
  return set.union(*effects) if effects else no_effects


@curry
def jaxpr_as_fun(closed_jaxpr: ClosedJaxpr, *args):
  return eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


class JaxprEqn(NamedTuple):
  invars: List[Atom]
  outvars: List[Var]
  primitive: Primitive
  params: Dict[str, Any]
  effects: Effects
  source_info: source_info_util.SourceInfo

  def __repr__(self):
    return str(pp_eqn(self, JaxprPpContext(), JaxprPpSettings())).rstrip()

  def replace(
      self,
      invars: Optional[List[Atom]] = None,
      outvars: Optional[List[Var]] = None,
      primitive: Optional[Primitive] = None,
      params: Optional[Dict[str, Any]] = None,
      effects: Optional[Effects] = None,
      source_info: Optional[source_info_util.SourceInfo] = None,
  ):
    # It is slightly faster to rebuild the tuple directly than to call _replace.
    return JaxprEqn(
      self.invars if invars is None else invars,
      self.outvars if outvars is None else outvars,
      self.primitive if primitive is None else primitive,
      self.params if params is None else params,
      self.effects if effects is None else effects,
      self.source_info if source_info is None else source_info,
    )


# TODO(mattjj): call typecheck rules here, so we don't form bad eqns
def new_jaxpr_eqn(invars, outvars, primitive, params, effects, source_info=None):
  source_info = source_info or source_info_util.new_source_info()
  if config.jax_enable_checks:
    assert all(isinstance(x, (Var, Literal)) for x in  invars)
    assert all(isinstance(v,  Var)           for v in outvars)
  return JaxprEqn(invars, outvars, primitive, params, effects, source_info)

@total_ordering
class Var:
  __slots__ = ["count", "suffix", "aval"]

  count: int
  suffix: str
  aval: AbstractValue

  def __init__(self, count: int, suffix: str, aval: AbstractValue):
    self.count = count
    self.suffix = suffix
    self.aval = raise_to_shaped(aval)

  def __lt__(self, other):
    if not isinstance(other, Var):
      return NotImplemented
    else:
      return (self.count, self.suffix) < (other.count, other.suffix)

  def __repr__(self):
    return _encode_digits_alphabetic(self.count) + self.suffix

def _encode_digits_alphabetic(n):
  if n == -1:
    return '*'
  s = ''
  while len(s) == 0 or n:
    n, i = n // 26, n % 26
    s = chr(97 + i % 26) + s
  return s

def _jaxpr_vars(jaxpr):
  return it.chain(
      jaxpr.invars, jaxpr.constvars,
      (v for eqn in jaxpr.eqns for v in eqn.outvars))

def gensym(jaxprs: Optional[Sequence[Jaxpr]] = None,
           suffix: str = '') -> Callable[[AbstractValue], Var]:
  """Produce distinct variables, printed with the optional suffix.

  If `jaxprs` is provided, the variables produced will be distinct from those in
  any of the given jaxprs.
  """
  if jaxprs is None:
    start = 0
  else:
    all_vars = it.chain.from_iterable(_jaxpr_vars(j) for j in jaxprs)
    start = 1 + max((v.count for v in all_vars), default=-1)
  counter = it.count(start=start)
  return lambda aval: Var(next(counter), suffix, aval)

# In a jaxpr, `dropvar` can appear in place of a bound variable to indicate that
# the assignment is dropped, i.e. that an expression's output value will never
# be read. In that sense, `dropvar` is not a variable, but it is convenient to
# treat it as a special case of one. Its `aval` is similarly inexact.
class DropVar(Var):
  def __init__(self, aval: AbstractValue):
    super().__init__(-1, '', aval)
  def __repr__(self): return '_'

class Literal:
  __slots__ = ["val", "aval", "hash"]

  val: Any
  aval: AbstractValue
  hash: Optional[int]

  def __init__(self, val, aval):
    self.val = val
    self.aval = aval
    try:
      self.hash = hash(val)
    except TypeError:
      if type(val) in literalable_types:
        try:
          self.hash = hash((val.item(), val.dtype))
        except (TypeError, AttributeError, ValueError):
          self.hash = None

  __hash__ = None  # type: ignore

  def __repr__(self):
    if hasattr(self, 'hash'):
      return f'{self.val}'
    else:
      return f'Literal(val={self.val})'

literalable_types: Set[type] = set()

Atom = Union[Var, Literal]

class Primitive:
  name: str
  # set for multi-output primitives.
  multiple_results: bool = False
  # set for call primitives processed in final style.
  call_primitive: bool = False
  # set for map primitives processed in final style.
  map_primitive: bool = False

  def __init__(self, name: str):
    self.name = name

  def __repr__(self):
    return f'{self.name}'

  def bind(self, *args, **params):
    assert (not config.jax_enable_checks or
            all(isinstance(arg, Tracer) or valid_jaxtype(arg) for arg in args)), args
    return self.bind_with_trace(find_top_trace(args), args, params)

  def bind_with_trace(self, trace, args, params):
    out = trace.process_primitive(self, map(trace.full_raise, args), params)
    return map(full_lower, out) if self.multiple_results else full_lower(out)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_abstract_eval(self, abstract_eval):
    self.abstract_eval = _effect_free_abstract_eval(abstract_eval)  # type: ignore[assignment]
    return abstract_eval

  def def_effectful_abstract_eval(self, effectful_abstract_eval):
    self.abstract_eval = effectful_abstract_eval  # type: ignore[assignment]
    return effectful_abstract_eval

  def def_custom_bind(self, bind):
    self.bind = bind
    return bind

  def impl(self, *args, **params):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))

  def abstract_eval(self, *args, **params):
    raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                              .format(self.name))

  def get_bind_params(self, params):
    return [], params


def _effect_free_abstract_eval(abstract_eval):
  def abstract_eval_(*args, **kwargs):
    return abstract_eval(*args, **kwargs), no_effects
  return abstract_eval_
  

# -- Pretty-printing --

class JaxprPpSettings(NamedTuple):
  print_shapes: bool = True
  source_info: bool = False
  name_stack: bool = False
  custom_pp_eqn_rules: bool = True
  print_effects: bool = False

# A JaxprPpContext allows us to globally uniquify variable names within nested
# Jaxprs.
class JaxprPpContext:
  var_ids: DefaultDict[Var, int]

  def __init__(self):
    self.var_ids = collections.defaultdict(it.count().__next__, {})


def pp_var(v: Var, context: JaxprPpContext) -> str:
  if isinstance(v, (Literal, DropVar)): return str(v)
  return f"{_encode_digits_alphabetic(context.var_ids[v])}{v.suffix}"

def pp_aval(a: AbstractValue, context: JaxprPpContext) -> str:
  if isinstance(a, DShapedArray):
    shape = [pp_var(d, context) if type(d) is Var else str(d) for d in a.shape]
    dtype = _short_dtype_name(a.dtype)
    return f'{dtype}[{",".join(shape)}]'
  else:
    return a.str_short(short_dtypes=True)

def pp_vars(vs: Sequence[Any], context: JaxprPpContext,
            *, separator="", print_shapes: bool = False) -> pp.Doc:
  if print_shapes:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()), [
        pp.text(pp_var(v, context)) +
        pp.type_annotation(pp.text(":" + pp_aval(v.aval, context)))
        for v in vs
      ])
    ))
  else:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()),
              [pp.text(pp_var(v, context)) for v in vs])
    ))

def pp_kv_pair(k:str, v: Any, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if type(v) is tuple and all(isinstance(j, (Jaxpr, ClosedJaxpr)) for j in v):
    pp_v = pp_jaxprs(v, context, settings)
  elif isinstance(v, Jaxpr):
    pp_v = pp_jaxpr(v, context, settings)
  elif isinstance(v, ClosedJaxpr):
    pp_v = pp_jaxpr(v.jaxpr, context, settings)
  else:
    pp_v = pp.text(str(v))
  return pp.text(f'{k}=') + pp_v

def pp_kv_pairs(kv_pairs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if not kv_pairs:
    return pp.nil()
  return pp.group(
    pp.nest(2, pp.concat([
      pp.text("["),  pp.brk(""),
      pp.join(pp.brk(), [pp_kv_pair(k, v, context, settings) for k, v in kv_pairs])
    ]))
    + pp.brk("") + pp.text("]")
  )

def pp_eqn(eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings
           ) -> pp.Doc:
  rule = (_pp_eqn if not settings.custom_pp_eqn_rules else
          pp_eqn_rules.get(eqn.primitive, _pp_eqn))
  return rule(eqn, context, settings)

def _pp_eqn(eqn, context, settings) -> pp.Doc:
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  name_stack_annotation = f'[{eqn.source_info.name_stack}]' if settings.name_stack else None
  lhs = pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name, annotation=name_stack_annotation),
         pp_kv_pairs(sorted(eqn.params.items()), context, settings),
         pp.text(" ") + pp_vars(eqn.invars, context)]
  return pp.concat([lhs, pp.text(" = ", annotation=annotation), *rhs])
CustomPpEqnRule = Callable[[JaxprEqn, JaxprPpContext, JaxprPpSettings], pp.Doc]
pp_eqn_rules: Dict[Primitive, CustomPpEqnRule]  = {}

def pp_eqns(eqns, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  return pp.join(
    pp.brk("; "),
    [pp_eqn(e, context, settings) for e in eqns])

def _compact_eqn_should_include(k: str, v: Any) -> bool:
  if k == 'branches': return False
  if isinstance(v, (Jaxpr, ClosedJaxpr)): return False
  if (isinstance(v, tuple) and
      any(isinstance(e, (Jaxpr, ClosedJaxpr)) for e in v)):
    return False
  return True

def str_eqn_compact(primitive_name: str, params: Dict) -> str:
  "Compact equation to string conversion used in HLO metadata."
  kvs = " ".join(f"{k}={v}" for k, v in params.items()
                 if _compact_eqn_should_include(k, v))
  return f"{primitive_name}[{kvs}]" if len(kvs) > 0 else primitive_name

def pp_jaxpr_skeleton(jaxpr, eqns_fn, context: JaxprPpContext,
                      settings: JaxprPpSettings) -> pp.Doc:
  constvars = pp_vars(jaxpr.constvars, context, print_shapes=settings.print_shapes)
  invars = pp_vars(jaxpr.invars, context, print_shapes=settings.print_shapes)
  eqns = eqns_fn()
  outvars = pp.concat([
    pp.text("("), pp_vars(jaxpr.outvars, context, separator=","),
    pp.text(")" if len(jaxpr.outvars) != 1 else ",)")])
  if settings.print_effects:
    # TODO(sharadmv): render an entire signature here
    eff_text = [pp.text(" : { ")]
    for i, eff in enumerate(jaxpr.effects):
      if i > 0:
        eff_text.append(pp.text(", "))
      if isinstance(eff, effects.JaxprInputEffect):
        index = eff.input_index
        all_vars = [*jaxpr.constvars, *jaxpr.invars]
        eff_text.append(pp_effect(eff.replace(input_index=all_vars[index]),
                                  context))
      else:
        eff_text.append(pp_effect(eff, context))
    eff_text.append(pp.text(" }"))
  else:
    eff_text = []
  return pp.group(pp.nest(2, pp.concat([
    pp.text("{ "), pp.keyword(pp.text("lambda ")),
    constvars, pp.text("; "), invars,
    pp.text(". "), pp.keyword(pp.text("let")),
    pp.nest(2, pp.brk() + eqns), pp.brk(),
    pp.keyword(pp.text("in ")), outvars,
    pp.concat(eff_text)
  ])) + pp.text(" }"))


def pp_jaxpr(jaxpr, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  eqns_fn = lambda: pp_eqns(jaxpr.eqns, context, settings)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)

def pp_jaxprs(jaxprs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  jaxprs = [j.jaxpr if isinstance(j, ClosedJaxpr) else j for j in jaxprs]
  return pp.group(pp.nest(2, pp.concat([
      pp.text('('), pp.brk(""),
      pp.join(pp.brk(), map(lambda x: pp_jaxpr(x, context, settings), jaxprs))]
    )) + pp.brk("") + pp.text(')')
  )


def pp_jaxpr_eqn_range(jaxpr: Jaxpr, lo: int, hi: int, context: JaxprPpContext,
                       settings: JaxprPpSettings) -> pp.Doc:
  lo = max(lo, 0)
  hi = max(lo, min(hi, len(jaxpr.eqns)))
  eqns = jaxpr.eqns[lo:hi]
  def eqns_fn():
    pps = []
    if len(eqns) == 0 and len(jaxpr.eqns) != 0:
      pps.append(pp.text('...'))
    else:
      if lo != 0:
        pps.append(pp.text('...'))
      pps.extend(map((lambda e: pp_eqn(e, context, settings)), eqns))
      if hi != len(jaxpr.eqns):
        pps.append(pp.text('...'))
    return pp.join(pp.brk("; "), pps)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)

def pp_effect(effect: Effect, context: JaxprPpContext) -> pp.Doc:
  if hasattr(effect, "_pretty_print"):
    return effect._pretty_print(context)
  return pp.text(str(effect))
