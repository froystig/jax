from __future__ import annotations

from functools import partial, wraps

import jax
import jax.numpy as jnp
import jax.extend as jex
import mlx
import mlx.core as mx

from jax._src import api_util
from jax._src import linear_util
from jax._src import util
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


_dtype_mappings = {
  mx.float32: jnp.dtype('float32'),
}

_mlx_rules = {}

def get_mlx_rule(p: jex.core.Primitive):
  return _mlx_rules.get(p)

_mlx_rules[jex.core.primitives.add_p] = mx.add
_mlx_rules[jex.core.primitives.cos_p] = mx.cos


def abstractify_mlx_value(x):
  return jax.ShapeDtypeStruct(x.shape, _dtype_mappings.get(x.dtype, x.dtype))

def jax2mlx(f):
  @wraps(f)
  def mlxed(*args, **kwargs):
    args_flat, in_tree = jax.tree.flatten((args, kwargs))
    flat_fun, out_tree = api_util.flatten_fun(linear_util.wrap_init(f), in_tree)
    tys_flat = map(abstractify_mlx_value, args_flat)
    jaxpr = jax.make_jaxpr(flat_fun.call_wrapped)(*tys_flat)
    outs_flat = mlx_eval_jaxpr(jaxpr, *args_flat)
    return jax.tree.unflatten(out_tree(), outs_flat)
  return mlxed


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


def test_f(x, y):
  return x + jax.lax.cos(y)

def main(*args):
  ins_jax = [jnp.array(float(x)) for x in args]
  ins_mlx = [mx.array(float(x)) for x in args]
  f = test_f

  jax_eval = f
  jax_ceval = jax.jit(jax_eval)
  print('jax eag:', jax_eval(*ins_jax))
  print('jax jit:', jax_ceval(*ins_jax))

  mlx_eval = jax2mlx(f)
  mlx_ceval = mx.compile(mlx_eval)
  print('mlx eag:', mlx_eval(*ins_mlx))
  print('mlx jit:', mlx_ceval(*ins_mlx))


if __name__ == '__main__':
  import sys
  main(*sys.argv[1:])
