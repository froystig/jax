# # Jaxpr API

from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
from jax._src import extend as jex
from jax import util
from jax.config import config

config.update('jax_traceback_filtering', 'off')

unsafe_map, map = map, util.safe_map

def f(x):
  y = jnp.exp(x)
  y = jnp.exp(y)
  return y


jaxpr = jex.jaxpr.make_jaxpr(f)(2.)

print(type(jaxpr))
print(jaxpr)

# ## Jaxpr interpreters

# ### `eval_jaxpr` (incomplete)

def eval_jaxpr(jaxpr: jex.jaxpr.Jaxpr, *args: Any):

  env: Dict[jex.jaxpr.Var, Any] = {}

  def read(atom: jex.jaxpr.Atom) -> Any:
    if isinstance(atom, jex.jaxpr.Literal):
      return atom.value
    return env[atom]

  def write(var: jex.jaxpr.Var, val: Any):
    env[var] = val

  for constvar, constval in jaxpr.consts:
    write(constvar, constval)
  map(write, jaxpr.inputs, args)

  for eqn in jaxpr.equations():
    invals = map(read, eqn.inputs)
    ans = eqn.primitive.bind(*invals, **eqn.params)
    map(write, eqn.outputs, ans)

  return map(read, jaxpr.outputs)

print(eval_jaxpr(jaxpr, 4.))

# ### `inverse_jaxpr`

inverse_rules: Dict[jex.primitives.Primitive, Any] = {}

def _exp_inverse(x):
  return [jnp.log(x)]
exp_p, = jex.primitives.primitives_by_name('exp')
inverse_rules[exp_p] = _exp_inverse

def inverse_jaxpr(jaxpr: jex.jaxpr.Jaxpr, *args: Any):

  env: Dict[jex.jaxpr.Var, Any] = {}

  def read(atom: jex.jaxpr.Atom) -> Any:
    if isinstance(atom, jex.jaxpr.Literal):
      return atom.value
    return env[atom]

  def write(var: jex.jaxpr.Var, val: Any):
    env[var] = val

  for constvar, constval in jaxpr.consts:
    write(constvar, constval)
  map(write, jaxpr.outputs, args)

  for eqn in jaxpr.equations()[::-1]:
    outvals = map(read, eqn.outputs)
    if eqn.primitive not in inverse_rules:
      raise NotImplementedError(eqn.primitive)
    rule = inverse_rules[eqn.primitive]
    ans = rule(*outvals, **eqn.params)
    map(write, eqn.inputs, ans)

  return map(read, jaxpr.inputs)

print(eval_jaxpr(jaxpr, 1.))
print(inverse_jaxpr(jaxpr, 1.))
