from __future__ import annotations

import collections
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from jax._src import core
from jax._src import util

import slab as sl

map, zip = util.safe_map, util.safe_zip

def make_djaxpr(*args, **kwargs):
  def djaxpr_maker(*args_, **kwargs_):
    with jax._src.config.dynamic_shapes(True):
      return jax.make_jaxpr(*args, **kwargs)(*args_, **kwargs_)
  return djaxpr_maker

def _check_axis_size_conflicts(all_axes, sizes):
  if len(all_axes) != len(set(all_axes)):
    d = collections.defaultdict(list)
    for name, sz in zip(all_axes, sizes):
      d[name].append(sz)
    msg = '; '.join([f'{name}: {" != ".join(map(str, sizes))}'
                      for name, sizes in d.items() if len(sizes) > 1])
    raise ValueError(f'abstracted axes resolve to conflicting sizes. {msg}')

def djit(f, abstracted_axes, **djit_kwargs):
  # TODO(frostig,mattjj): un/flatten f
  def f_wrapped(slab, *args):  # TODO(frostig,mattjj): kw support
    djaxpr = make_djaxpr(
        f, abstracted_axes=abstracted_axes, **djit_kwargs)(*args).jaxpr
    slab, views = sl.chain(slab, sl.slab_upload, *args, unary=True)
    shapes = [x.shape for x in args]
    all_axes, sizes = util.unzip2(
        set((name, sz) for axes, shape in zip(abstracted_axes, shapes)
            for name, sz in zip(axes, shape)))
    _check_axis_size_conflicts(all_axes, sizes)
    dim_index = {n: i for i, n in enumerate(all_axes)}

    def interp(slab, dims, addrs):
      # TODO(frostig,mattjj): reconstructing slab views seems less than ideal
      views = []
      for addr, axes, arg in zip(addrs, abstracted_axes, args):
        resolved_shape = tuple(dims[dim_index[name]] for name in axes)
        views.append(sl.SlabView(addr, resolved_shape, arg.dtype))
      slab, outs = eval_djaxpr(djaxpr, slab, *dims, *views)
      return slab, outs

    slab, out_views = interp(slab, sizes, [v.addr for v in views])
    return slab, tuple(sl.slab_download(slab, v) for v in out_views)

  return f_wrapped

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

def print_seg(msg):
  print()
  print(f'-- {msg}')
  print()

def check_djit(slab, f, abstracted_axes, *args):
  refs, _ = jax.tree.flatten(f(*args))
  f_djit = djit(f, abstracted_axes=abstracted_axes)
  slab, outs = f_djit(slab, *args)
  for out, ref in zip(outs, refs):
    assert jnp.allclose(out, ref, atol=1e-4), jnp.max(jnp.abs(out - ref))

def test(slab, xs):
  a, b = xs

  def f(a, b):
    c = jnp.dot(a, b)
    return jnp.tanh(c)

  abstracted_axes = (('m', 'k'), ('k', 'n'))

  print_seg('djaxpr')
  djaxpr = make_djaxpr(f, abstracted_axes=abstracted_axes)(a, b).jaxpr
  print(djaxpr)

  print_seg('djax output')
  f_djit = djit(f, abstracted_axes=abstracted_axes)
  slab = sl.slab_make(1024)
  slab, [c] = f_djit(slab, a, b)
  print(c)

  print_seg('djax -> jax lowering')
  big_jaxpr = jax.make_jaxpr(f_djit)(slab, a, b)
  print(big_jaxpr)
  print(len(str(big_jaxpr).split('\n')))

  # TODO(frostig,mattjj,apaszke): bug
  #check_djit(slab, f, abstracted_axes, a, b)

def parse_arr(i, s):
  shape = eval(s)
  return np.random.RandomState(i).normal(size=shape).astype(np.float32)

def main(args):
  xs = map(parse_arr, range(len(args)), args)
  assert all(len(x.shape) == 2 for x in xs)
  slab = sl.slab_make(1024)
  test(slab, xs)


if __name__ == '__main__':
  import ipdb, sys, traceback
  def info(t, v, i):
    traceback.print_exception(t, v, i)
    ipdb.pm()
  sys.excepthook = info

  main(sys.argv[1:])
