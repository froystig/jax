from __future__ import annotations

import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from jax._src import core
from jax._src.util import safe_map, safe_zip

import slab as sl

map, zip = safe_map, safe_zip

jax.config.update('jax_dynamic_shapes', True)


def eval_djaxpr(jaxpr: core.Jaxpr, slab: sl.Slab, *args: jax.Array | sl.SlabView):
  if jaxpr.constvars: raise NotImplementedError

  env = {}

  def read(a):
    return env[a] if type(a) is core.Var else a.val

  def write(v, val):
    env[v] = val

  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    invals = map(read, eqn.invars)
    slab, outvals = rules[eqn.primitive](slab, *invals, **eqn.params)
    map(write, eqn.outvars, outvals)
  return slab, map(read, jaxpr.outvars)

rules = {}

def matmul_rule(slab, lhs, rhs, *, dimension_numbers, **_):
  slab, out = sl.matmul(slab, lhs, rhs)
  return slab, [out]
rules[lax.dot_general_p] = matmul_rule

def tanh_rule(slab, x, **_):
  slab, out = sl.tanh(slab, x)
  return slab, [out]
rules[lax.tanh_p] = tanh_rule

# -------

def test(slab, xs, views):
  a, b = xs
  av, bv = views

  def f(a, b):
    c = jnp.dot(a, b)
    return jnp.tanh(c)

  jaxpr = jax.make_jaxpr(f, abstracted_axes=(('m', 'k'), ('k', 'n')))(a, b).jaxpr
  print_seg('djaxpr')
  print(jaxpr)
  def lower(slab, m, k, n, a_addr, b_addr):
    av = sl.SlabView(a_addr, (m, k), a.dtype)
    bv = sl.SlabView(b_addr, (k, n), b.dtype)
    slab, [cv] = eval_djaxpr(jaxpr, slab, m, k, n, av, bv)
    return slab, cv
  sl.print_seg('djax output')
  slab, cv = lower(slab, a.shape[0], a.shape[1], b.shape[1], av.addr, bv.addr)
  c = sl.slab_download(slab, cv)
  print(c)
  print_seg('djax -> jax lowering')
  big_jaxpr = jax.make_jaxpr(lower)(slab, a.shape[0], a.shape[1], b.shape[1], av.addr, bv.addr)
  print(big_jaxpr)
  print(len(str(big_jaxpr).split('\n')))

def parse_arr(i, s):
  shape = eval(s)
  return np.random.RandomState(i).normal(size=shape).astype(np.float32)

def print_seg(msg):
  print()
  print(f'-- {msg}')
  print()

def main(args):
  xs = map(parse_arr, range(len(args)), args)
  assert all(len(x.shape) == 2 for x in xs)

  slab = sl.slab_make(1024)

  vals = []
  for x in xs:
    slab, v = sl.slab_upload(slab, x)
    vals.append(v)
    print_seg('slab after write')
    print_seg('slab_read result')
    print(sl.slab_download(slab, v))

  test(slab, xs, vals)


if __name__ == '__main__':
  import ipdb, sys, traceback
  def info(t, v, i):
    traceback.print_exception(t, v, i)
    ipdb.pm()
  sys.excepthook = info

  main(sys.argv[1:])
