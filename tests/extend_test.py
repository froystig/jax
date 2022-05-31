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

import functools
from typing import Any, Callable, Dict

from absl.testing import absltest

import jax
import jax.extend as jex
from jax import lax
from jax import util
from jax.config import config

from jax._src import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()

unsafe_map, map = map, util.safe_map


def inverse(rules: Dict[jex.Primitive, Any], f: Callable, *args: Any):
  jaxpr: jex.Jaxpr = jex.make_jaxpr(f)(*args)
  return inverse_jaxpr(rules, jaxpr, *args)

def inverse_jaxpr(
    rules: Dict[jex.Primitive, Any], jaxpr: jex.Jaxpr, *args: Any):
  env: Dict[jex.Var, Any] = {}

  def read(atom: jex.Atom) -> Any:
    if isinstance(atom, jex.Literal):
      return atom.value
    return env[atom]

  def write(var: jex.Var, val: Any):
    env[var] = val

  for constvar, constval in jaxpr.consts:
    write(constvar, constval)
  map(write, jaxpr.outputs, args)

  for eqn in jaxpr.equations()[::-1]:
    outvals = map(read, eqn.outputs)
    rule = rules[eqn.primitive]
    ans = rule(*outvals, **eqn.params)
    map(write, eqn.inputs, ans)

  return map(read, jaxpr.inputs)

def inverse_exp(x):
  return [lax.log(x)]

def inverse_xla_call(rules, *args, call_jaxpr, **params):
  inverted_jaxpr = jex.make_jaxpr(
      functools.partial(inverse_jaxpr, rules, call_jaxpr))(*args)
  xla_call_p, = jex.primitives_by_name('xla_call')
  return xla_call_p.bind(*args, call_jaxpr=inverted_jaxpr, **params)


class JaxprInterpreterTest(jtu.JaxTestCase):

  def test_inverse_interpreter_basic(self):
    # End-to-end test that one can write a natural 'inverse'
    # transformation as a jaxpr interpreter.

    rules: Dict[jex.Primitive, Any] = {}
    exp_p, = jex.primitives_by_name('exp')
    rules[exp_p] = inverse_exp

    def f(x):
      y = lax.exp(x)
      z = lax.exp(y)
      return z

    z = 15.15
    out, = inverse(rules, f, z)
    self.assertEqual(out, lax.log(lax.log(z)))

    log_p, = jex.primitives_by_name('log')
    inv_jaxpr = jex.make_jaxpr(functools.partial(inverse, rules, f))(z)
    eq1, eq2 = inv_jaxpr.equations()
    self.assertEqual(eq1.primitive, log_p)
    self.assertEqual(eq2.primitive, log_p)

  def test_inverse_interpreter_with_call(self):
    # Also handle higher-order primitives (e.g. xla_call).

    rules: Dict[jex.Primitive, Any] = {}
    exp_p, = jex.primitives_by_name('exp')
    rules[exp_p] = inverse_exp
    xla_call_p, = jex.primitives_by_name('xla_call')
    rules[xla_call_p] = functools.partial(inverse_xla_call, rules)

    def f(x):
      y = jax.jit(lax.exp)(x)
      z = lax.exp(y)
      return z

    z = 15.15
    out, = inverse(rules, f, z)
    self.assertEqual(out, lax.log(lax.log(z)))

    log_p, = jex.primitives_by_name('log')
    inv_jaxpr = jex.make_jaxpr(functools.partial(inverse, rules, f))(z)
    eq1, eq2 = inv_jaxpr.equations()
    self.assertEqual(eq1.primitive, log_p)
    self.assertEqual(eq2.primitive, xla_call_p)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
