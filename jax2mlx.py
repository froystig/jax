from __future__ import annotations

from functools import partial

import mlx
import mlx.core as mx

import jax
import jax.numpy as jnp

import jax.extend as jex

from jax._src import util
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


_mlx_rules = {}

def get_mlx_rule(p: jex.core.Primitive):
  return _mlx_rules.get(p)

_mlx_rules[jex.core.primitives.add_p] = mx.add
_mlx_rules[jex.core.primitives.cos_p] = mx.cos


def mlx_eval_jaxpr(closed_jaxpr, *args):
  jaxpr = closed_jaxpr.jaxpr
  consts = closed_jaxpr.consts

  def read(v: jex.core.Atom):
    return v.val if isinstance(v, jex.core.Literal) else env[v]

  def write(v: jex.core.Var, val):
    env[v] = val

  env = {}
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    rule = get_mlx_rule(eqn.primitive)
    ans = rule(*map(read, eqn.invars))
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  return map(read, jaxpr.outvars)


def main(*args):
  ins_jax = [jnp.array(float(x)) for x in args]
  ins_mlx = [mx.array(float(x)) for x in args]
  jaxpr = jax.make_jaxpr(lambda x, y: x + jax.lax.cos(y))(*ins_jax)
  jax_eval = jex.core.jaxpr_as_fun(jaxpr)
  jax_ceval = jax.jit(jax_eval)
  mlx_eval = partial(mlx_eval_jaxpr, jaxpr)
  mlx_ceval = mx.compile(mlx_eval)
  print('jax eag:', jax_eval(*ins_jax))
  print('jax jit:', jax_ceval(*ins_jax))
  print('mlx eag:', mlx_eval(*ins_mlx))
  print('mlx jit:', mlx_ceval(*ins_mlx))


if __name__ == '__main__':
  import sys
  main(*sys.argv[1:])
