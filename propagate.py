# # Propagate interpreter
# The propagate interpreter converts a `Jaxpr` to a directed graph where
# vars are nodes and primitives are edges. It initializes invars and outvars with
# `Cell`s (an interface defined below), where a `Cell` encapsulates a value (or a set
# of values) that a node in the graph can take on, and the Cell is computed from
# neighboring `Cell`s, using a set of propagation rules for each primitive.Each rule
# indicates whether the propagation has been completed for the given edge.
# If so, the propagate interpreter continues on to that primitive's neighbors
# in the graph. Propagation continues until there are `Cell`s for every node, or
# when no further progress can be made. Finally, `Cell` values for all nodes in the
# graph are returned.

import abc
import collections
import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar 

import dataclasses
from jax.config import config
from jax import extend as jex
from jax import tree_util
from jax import util as jax_util

config.update('jax_traceback_filtering', 'off')


# ## `propagate` implementation

# ### `Cell` definition

State = Any
T = TypeVar('T')


class Cell(metaclass=abc.ABCMeta):

  def __init__(self, aval):
    self.aval = aval

  @abc.abstractmethod
  def __lt__(self, other: Any) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def top(self) -> bool:
    pass

  @abc.abstractmethod
  def bottom(self) -> bool:
    pass

  @abc.abstractmethod
  def join(self: T, other: T) -> T:
    pass

  @property
  def shape(self) -> Tuple[int]:
    return self.aval.shape

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def is_unknown(self):
    # Convenient alias
    return self.bottom()

  @abc.abstractclassmethod
  def new(cls, value):
    """Creates a new instance of a Cell from a value."""
    raise NotImplementedError

  @abc.abstractclassmethod
  def unknown(cls, aval: jex.Type):
    """Creates an unknown Cell from an abstract value."""
    raise NotImplementedError

  @abc.abstractmethod
  def tree_flatten(self):
    pass

  @abc.abstractclassmethod
  def tree_unflatten(cls, data, xs):
    pass

# ### `Environment` - helper for interpreter

class Environment:
  """Keeps track of variables and their values during propagation."""

  def __init__(self, cell_type, jaxpr):
    self.cell_type = cell_type
    self.env: Dict[jex.Var, Cell] = {}
    self.states: Dict[jex.Equation, Cell] = {}
    self.jaxpr: jex.Jaxpr = jaxpr

  def read(self, var: jex.Atom) -> Cell:
    if isinstance(var, jex.Literal):
      return self.cell_type.new(var.value)
    else:
      return self.env.get(var, self.cell_type.unknown(var.type))

  def write(self, var: jex.Atom, cell: Cell) -> Cell:
    if isinstance(var, jex.Literal):
      return cell
    cur_cell = self.read(var)
    self.env[var] = cur_cell.join(cell)
    return self.env[var]

  def __getitem__(self, var: jex.Atom) -> Cell:
    return self.read(var)

  def __setitem__(self, key, val):
    raise ValueError('Environments do not support __setitem__. Please use the '
                     '`write` method instead.')

  def __contains__(self, var: jex.Atom):
    if isinstance(var, jex.Literal):
      return True
    return var in self.env

  def read_state(self, eqn: jex.Equation) -> State:
    return self.states.get(eqn, None)

  def write_state(self, eqn: jex.Equation, state: State) -> None:
    self.states[eqn] = state

# ### Propagation logic

def construct_graph_representation(eqns: List[jex.Equation]):
  """Constructs a graph representation of a Jaxpr."""
  neighbors = collections.defaultdict(set)
  for eqn in eqns:
    for var in it.chain(eqn.inputs, eqn.outputs):
      if isinstance(var, jex.Literal):
        continue
      neighbors[var].add(eqn)

  def get_neighbors(var):
    if isinstance(var, jex.Literal):
      return set()
    return neighbors[var]

  return get_neighbors


def update_queue_state(queue, cur_eqn: jex.Equation, get_neighbor_eqns, incells, outcells,
                       new_incells, new_outcells):
  """Updates the queue from the result of a propagation."""
  all_vars = cur_eqn.inputs + cur_eqn.outputs
  old_cells = incells + outcells
  new_cells = new_incells + new_outcells

  for var, old_cell, new_cell in zip(all_vars, old_cells, new_cells):
    # If old_cell is less than new_cell, we know the propagation has made
    # progress.
    if old_cell < new_cell:
      # Extend left as a heuristic because in graphs corresponding to
      # chains of unary functions, we immediately want to pop off these
      # neighbors in the next iteration
      neighbors = get_neighbor_eqns(var) - set(queue) - {cur_eqn}
      queue.extendleft(neighbors)


PropagationRule = Callable[[List[Any], List[Cell]], Tuple[List[Cell],
                                                          List[Cell], State]]



def propagate(cell_type: Type[Cell],
              rules: Dict[jex.Primitive, PropagationRule],
              jaxpr: jex.Jaxpr,
              incells: List[Cell],
              outcells: List[Cell]):
  """Propagates cells in a Jaxpr using a set of rules.
  Args:
    cell_type: used to instantiate literals into cells
    rules: maps JAX primitives to propagation rule functions
    jaxpr: used to construct the propagation graph
    incells: used to populate the Jaxpr's invars
    outcells: used to populate the Jaxpr's outcells
  Returns:
    The Jaxpr environment after propagation has terminated
  """
  env = Environment(cell_type, jaxpr)
  if not jaxpr.consts:
    constvars, constvals = [], []
  else:
    constvars, constvals = zip(*jaxpr.consts)
  for constvar, constval in zip(constvars, constvals):
    env.write(constvar, cell_type.new(constval))
  jax_util.safe_map(env.write, jaxpr.inputs, incells)
  jax_util.safe_map(env.write, jaxpr.outputs, outcells)

  eqns = jaxpr.equations()

  get_neighbor_eqns = construct_graph_representation(eqns)
  # Initialize propagation queue with equations neighboring constvars, invars,
  # and outvars.
  out_eqns = set()
  for eqn in eqns:
    for var in it.chain(eqn.inputs, eqn.outputs):
      env.write(var, cell_type.unknown(var.type))

  for var in it.chain(jaxpr.outputs, jaxpr.inputs, jaxpr.consts):
    out_eqns.update(get_neighbor_eqns(var))
  queue = collections.deque(out_eqns)
  while queue:
    eqn: jex.Equation = queue.popleft()

    incells = jax_util.safe_map(env.read, eqn.inputs)
    outcells = jax_util.safe_map(env.read, eqn.outputs)

    rule = rules[eqn.primitive]
    new_incells, new_outcells, eqn_state = rule(incells, outcells, **eqn.params)
    env.write_state(eqn, eqn_state)

    new_incells = jax_util.safe_map(env.write, eqn.inputs, new_incells)
    new_outcells = jax_util.safe_map(env.write, eqn.outputs, new_outcells)

    update_queue_state(queue, eqn, get_neighbor_eqns, incells, outcells,
                       new_incells, new_outcells)
  return env

# ## Examples

import jax
import jax.numpy as jnp

# ### `eval` as `propagate`

@tree_util.register_pytree_node_class
class ValueCell(Cell):
  
  def __init__(self, value: Optional[Any], type: jex.Type):
    self.value = value
    self.type = type

  def top(self):
    return self.value is not None

  def bottom(self):
    return self.value is None

  def __lt__(self, other: 'ValueCell'):
    return self.bottom() and other.top()

  def join(self, other: 'ValueCell') -> 'ValueCell':
    if self < other:
      return other
    if other < self:
      return self
    return self

  def tree_flatten(self):
    return (self.value,), self.type

  @classmethod
  def tree_unflatten(cls, type, values):
    return ValueCell(values[0], type)

  @classmethod
  def new(cls, value):
    # TODO(sharadmv,frostig): get type from value?
    aval = jax.core.get_aval(value)
    type = jax._src.extend.jaxpr.type_from_aval(aval)
    return ValueCell(value, type)

  @classmethod
  def unknown(cls, type: jex.Type):
    return ValueCell(None, type)


def default_prim_rule(
    prim: jex.Primitive, incells: List[Cell], outcells: List[Cell],
    **params: Any):
  if all(incell.top() for incell in incells):
    values = [incell.value for incell in incells]
    outputs = prim.bind(*values, **params)
    outcells = [ValueCell.new(output) for output in outputs]
    return incells, outcells, None
  return incells, outcells, None

class EvalRules(dict):

  def __getitem__(self, prim: jex.Primitive):
    return functools.partial(default_prim_rule, prim)

def f(x):
  return jnp.square(x)
jaxpr = jex.make_jaxpr(f)(2.)
print(jaxpr)

incells = [ValueCell.new(2.)]
outcells = [ValueCell.unknown(jaxpr.outputs[0].type)]
out_env = propagate(ValueCell, EvalRules(), jaxpr, incells, outcells)
print([out_env.read(var).value for var in jaxpr.outputs])

# ### `inverse` as `propagate`

_inverse_rules = {}

def default_prim_inverse_rule(
    prim: jex.Primitive, incells: List[Cell], outcells: List[Cell],
    **params: Any):
  if all(incell.top() for incell in incells):
    values = [incell.value for incell in incells]
    outputs = prim.bind(*values, **params)
    outcells = [ValueCell.new(output) for output in outputs]
    return incells, outcells, None
  if prim not in _inverse_rules:
    raise NotImplementedError(prim)
  return _inverse_rules[prim](incells, outcells, **params)

class InverseRules(dict):

  def __getitem__(self, prim: jex.Primitive):
    return functools.partial(default_prim_inverse_rule, prim)

def _add_inverse(incells, outcells):
  x, y = incells
  z, = outcells
  if not z.top():
    return incells, outcells, None
  if x.top():
    return [x.value, ValueCell.new(z.value - x.value)], outcells, None
  if y.top():
    return [ValueCell.new(z.value - y.value), y.value], outcells, None
  return incells, outcells, None
add_p, = jex.primitives_by_name("add")
_inverse_rules[add_p] = _add_inverse

def _integer_pow_inverse(incells, outcells, *, y):
  x, = incells
  z, = outcells
  if z.top():
    return [ValueCell.new(jnp.power(z.value, 1 / y))], outcells, None
  return incells, outcells, None
integer_pow_p, = jex.primitives_by_name("integer_pow")
_inverse_rules[integer_pow_p] = _integer_pow_inverse

def inverse(f, *trace_args, **trace_kwargs):
  def wrapped(*args, **kwargs):
    if not trace_args and not trace_kwargs:
      in_tree = tree_util.tree_structure((args, kwargs))
      jaxpr, out_shape = jex.make_jaxpr(f, return_shape=True)(*args, **kwargs)
    else:
      in_tree = tree_util.tree_structure((trace_args, trace_kwargs))
      jaxpr, out_shape = jex.make_jaxpr(f, return_shape=True)(*trace_args, **trace_kwargs)
    flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
    incells = [ValueCell.unknown(invar.type) for invar in jaxpr.inputs]
    outcells = [ValueCell.new(arg) for arg in flat_args]
    env = propagate(ValueCell, InverseRules(), jaxpr, incells, outcells)
    invals = [env.read(var).value for var in jaxpr.inputs]
    in_args, in_kwargs = tree_util.tree_unflatten(in_tree, invals)
    if not in_kwargs:
      if len(in_args) == 1:
        return in_args[0]
      return in_args
    return in_args, in_kwargs
  return wrapped


@inverse
def f(x):
  return jnp.square(x) + 2.

# Let's try to compute f^-1(6.)

print(f(6.))

# ### Call primitive rules?

def _xla_call_inverse_rule(incells, outcells, *, call_jaxpr, **params):
  def _inverted_func(incells_, outcells_):
    env = propagate(ValueCell, _inverse_rules, call_jaxpr,
                    incells_, outcells_)
    incells_ = [env.read(var) for var in call_jaxpr.inputs]
    outcells_ = [env.read(var) for var in call_jaxpr.outputs]
    return incells_, outcells_
  inverted_call_jaxpr, out_shape = jex.make_jaxpr(_inverted_func,
      return_shape=True)(incells, outcells)
  out_tree = tree_util.tree_structure(out_shape)
  flat_inputs, treedef = tree_util.tree_flatten((incells, outcells))
  out_flat = xla_call_p.bind(*flat_inputs, call_jaxpr=inverted_call_jaxpr,
      **params)
  incells, outcells = tree_util.tree_unflatten(out_tree, out_flat)
  return incells, outcells, None
xla_call_p, = jex.primitives_by_name("xla_call")
_inverse_rules[xla_call_p] = _xla_call_inverse_rule


@inverse
@jax.jit
def f(x):
  return jnp.square(x) + 2.

# Let's try to compute f^-1(6.)

print(f(6.))
print(jex.make_jaxpr(f)(6.))
