# Copyright 2021 The JAX Authors.
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

# Lowering and execution path that converts jaxprs into MLIR.
from __future__ import annotations

import collections
from collections.abc import Callable, Iterable, Iterator, Sequence
import contextlib
import dataclasses
import functools
from functools import partial
import heapq
import io
import itertools
import operator
import re
import types
import typing
from typing import Any, NamedTuple, Protocol, Union, cast as type_cast
import warnings

from jax._src import ad_util
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects as effects_lib
from jax._src import frozen_dict
from jax._src import hashable_array
from jax._src import jaxpr_util
from jax._src import linear_util as lu
from jax._src import path
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.layout import AutoLayout, Layout
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import dialects, ir, passmanager
# TODO(phawkins): remove try/except after jaxlib 0.7 is the minimum version.
try:
  from jaxlib.mlir._mlir_libs import _jax_mlir_ext
except ImportError:
  from jaxlib.mlir._mlir_libs import register_jax_dialects as _jax_mlir_ext  # pytype: disable=import-error
from jax._src.lib.mlir.dialects import func as func_dialect, hlo
from jax._src.mesh import AxisType
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding import Sharding as JSharding
from jax._src.sharding_impls import ( AUTO, NamedSharding,
                                     SdyArray, SdyArrayList,
                                     modify_sdy_sharding_wrt_axis_types)
from jax._src.state.types import AbstractRef
from jax._src.typing import ArrayLike
from jax._src.util import foreach
import numpy as np

# mypy: ignore-errors

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

T = typing.TypeVar("T")

Value = Any  # = ir.Value

# mypy implicitly sets this variable to true when type checking.
MYPY = False

lowerable_effects: effects_lib.EffectTypeSet = effects_lib.lowerable_effects


# IR Helpers

IrValues = Union[ir.Value, tuple[ir.Value, ...]]


def _is_not_block_argument(x: IrValues) -> bool:
  return not isinstance(x, ir.BlockArgument)


def dense_int_elements(xs) -> ir.DenseIntElementsAttr:
  return ir.DenseIntElementsAttr.get(np.asarray(xs, np.int64))

dense_int_array = ir.DenseI64ArrayAttr.get

def dense_bool_elements(xs: Sequence[bool]) -> ir.DenseElementsAttr:
  a = np.packbits(np.array(xs, np.bool_), bitorder='little')
  # TODO(b/209005197): Work around for MLIR crash for non-splat single element
  # buffers.
  if len(xs) == 1:
    a = np.array(0 if a.item() == 0 else 0xff, np.uint8)
  return ir.DenseElementsAttr.get(
      a, type=ir.IntegerType.get_signless(1), shape=[len(xs)])

def dense_bool_array(xs: Sequence[bool]) -> ir.DenseBoolArrayAttr:
  return ir.DenseBoolArrayAttr.get(xs)

def i32_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
def i64_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)

def shape_tensor(sizes: Sequence[int | ir.RankedTensorType]
                 ) -> ir.RankedTensorType:
  int1d = aval_to_ir_type(core.ShapedArray((1,), np.int32))
  i32_type = aval_to_ir_type(core.ShapedArray((), np.int32))
  def lower_dim(d):
    if type(d) is int:
      return ir_constant(np.array([d], np.int32))
    else:
      if d.type != i32_type:
        d = hlo.convert(i32_type, d)
      return hlo.reshape(int1d, d)
  ds = map(lower_dim, sizes)
  if not ds:
    return type_cast(ir.RankedTensorType, ir_constant(np.array([], np.int32)))
  elif len(ds) == 1:
    return ds[0]
  else:
    return hlo.concatenate(ds, i64_attr(0))


def delegate_lowering(ctx, lowering_fun, *args, **ctx_override_kwargs):
  """Side-effects on `ctx`"""
  ctx_new = ctx.replace(**ctx_override_kwargs)
  out = lowering_fun(ctx_new, *args)
  ctx.set_tokens_out(ctx_new.tokens_out)
  return out


# IR Types

IrTypes = Union[ir.Type, tuple[ir.Type, ...]]

def _is_ir_values(x: IrValues) -> bool:
  """Returns true if `x` is an ir.Value or tuple of ir.Values"""
  if isinstance(x, ir.Value):
    return True
  return (isinstance(x, tuple) and len(x) != 1
          and all(isinstance(v, ir.Value) for v in x))


# Non-canonicalized dtype to IR type mapping.
_dtype_to_ir_type : dict[np.dtype, Callable[[], ir.Type]] = {
  np.dtype(dtypes.float0): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
  np.dtype(dtypes.int4): partial(ir.IntegerType.get_signless, 4),
  np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
  np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
  np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
  np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
  np.dtype(dtypes.uint4): partial(ir.IntegerType.get_unsigned, 4),
  np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
  np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
  np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
  np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
  np.dtype(dtypes.float8_e4m3b11fnuz): ir.Float8E4M3B11FNUZType.get,
  np.dtype(dtypes.float8_e4m3fn): ir.Float8E4M3FNType.get,
  np.dtype(dtypes.float8_e4m3fnuz): ir.Float8E4M3FNUZType.get,
  np.dtype(dtypes.float8_e5m2): ir.Float8E5M2Type.get,
  np.dtype(dtypes.float8_e5m2fnuz): ir.Float8E5M2FNUZType.get,
  np.dtype(dtypes.bfloat16): ir.BF16Type.get,
  np.dtype(np.float16): ir.F16Type.get,
  np.dtype(np.float32): ir.F32Type.get,
  np.dtype(np.float64): ir.F64Type.get,
  np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
  np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
  np.dtype(dtypes.int2): partial(ir.IntegerType.get_signless, 2),
  np.dtype(dtypes.uint2): partial(ir.IntegerType.get_unsigned, 2),
  np.dtype(dtypes.float8_e3m4): ir.Float8E3M4Type.get,
  np.dtype(dtypes.float8_e4m3): ir.Float8E4M3Type.get,
  np.dtype(dtypes.float8_e8m0fnu): ir.Float8E8M0FNUType.get,
  np.dtype(dtypes.float4_e2m1fn): ir.Float4E2M1FNType.get,
}

def dtype_to_ir_type(dtype: core.bint | np.dtype | np.generic) -> ir.Type:
  if isinstance(dtype, core.bint):
    # TODO Support different-size underlying dtypes to take advantage of the
    # bound for packing?
    dtype = np.dtype(np.int32)
  assert isinstance(dtype, (np.dtype, np.generic)), type(dtype)
  dtype = np.dtype(dtype)
  try:
    ir_type_factory = _dtype_to_ir_type[dtype]
  except KeyError as err:
    raise TypeError(
        f"No dtype_to_ir_type handler for dtype: {dtype}") from err
  return ir_type_factory()

def _array_ir_types(aval: core.ShapedArray | core.DShapedArray) -> ir.Type:
  aval = core.physical_aval(aval)  # type: ignore
  if not core.is_constant_shape(aval.shape):
    return _dynamic_array_ir_types(aval)  # type: ignore
  return ir.RankedTensorType.get(aval.shape, dtype_to_ir_type(aval.dtype))  # type: ignore

def _dynamic_array_ir_types(aval: core.ShapedArray) -> ir.Type:
  dyn_size = ir.ShapedType.get_dynamic_size()
  shape = [d if type(d) is int else dyn_size for d in aval.shape]
  return ir.RankedTensorType.get(shape, dtype_to_ir_type(aval.dtype))

ir_type_handlers: dict[type[core.AbstractValue], Callable[[Any], IrTypes]] = {}

def aval_to_ir_type(aval: core.AbstractValue) -> IrTypes:
  """Converts a JAX aval to zero or more MLIR IR types.

  In general, a JAX value may be represented by multiple IR values, so this
  function may return a tuple of types."""
  try:
    return ir_type_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No ir_type_handler for aval type: {type(aval)}") from err

ir_type_handlers[core.ShapedArray] = _array_ir_types
ir_type_handlers[core.AbstractToken] = lambda _: hlo.TokenType.get()
ir_type_handlers[core.DShapedArray] = _dynamic_array_ir_types

# This is a backwards compatibility shim for external users of jax.mlir apis.
def aval_to_ir_types(aval: core.AbstractValue) -> tuple[ir.Type, ...]:
  typ = aval_to_ir_type(aval)
  return (typ,) if isinstance(typ, ir.Type) else typ

# Constants

class ConstantHandler(Protocol):
  def __call__(self, val: Any) -> IrValues:
    """Builds an IR representation for a constant `val`.

    A JAX value is represented by zero or more IR values."""

_constant_handlers : dict[type, ConstantHandler] = {}

def register_constant_handler(type_: type, handler_fun: ConstantHandler):
  _constant_handlers[type_] = handler_fun

def get_constant_handler(type_: type) -> ConstantHandler:
  return _constant_handlers[type_]

def ir_constant(val: Any,
                const_lowering: dict[int, IrValues] = {},
                *,
                canonicalize_dtype: bool = False) -> IrValues:
  """Translate a Python `val` to an IR constant, canonicalizing its dtype.

  See https://docs.jax.dev/en/latest/internals/constants.html.
  Args:
    val: a Python value to be translated to a constant.
    const_lowering: an optional dictionary with known lowering for some
      constants, indexed by `id`. This is used, e.g., when we pass constants
      as MLIR function arguments.
    canonicalize_dtype: whether to canonicalize the dtype

  Returns:
    A representation of the constant as an IR value or sequence of IR values.
  """
  if np.shape(val) and (c_val := const_lowering.get(id(val))) is not None:
    return c_val
  if canonicalize_dtype:
    val = xla.canonicalize_dtype(val)
  for t in type(val).__mro__:
    handler = _constant_handlers.get(t)
    if handler:
      out = handler(val)
      assert _is_ir_values(out), (type(val), out)
      return out
  if hasattr(val, '__jax_array__'):
    return ir_constant(val.__jax_array__())
  raise TypeError(f"No constant handler for type: {type(val)}")

def _numpy_array_constant(x: np.ndarray | np.generic) -> IrValues:
  element_type = dtype_to_ir_type(x.dtype)
  shape = x.shape
  if x.dtype == np.bool_:
    x = np.packbits(x, bitorder='little')  # type: ignore
  x = np.ascontiguousarray(x)
  attr = ir.DenseElementsAttr.get(x, type=element_type, shape=shape)  # type: ignore
  return hlo.constant(attr)


def _masked_array_constant_handler(*args, **kwargs):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")

register_constant_handler(np.ma.MaskedArray, _masked_array_constant_handler)

def _ndarray_constant_handler(val: np.ndarray | np.generic) -> IrValues:
  """Constant handler for ndarray literals, handling zero-size strides.

  In most cases this function calls _numpy_array_constant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses np.broadcast_to which in turn
  uses size-zero strides).

  Args:
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  if val.dtype == dtypes.float0:
    return _numpy_array_constant(np.zeros(val.shape, dtype=np.bool_))
  elif 0 in val.strides and val.size > 0:
    zero_stride_axes, = np.where(np.equal(0, val.strides))
    other_axes, = np.where(np.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)  # type: ignore
                              for ax in range(val.ndim))]
    out = hlo.broadcast_in_dim(
        ir.RankedTensorType.get(
            val.shape, dtype_to_ir_type(collapsed_val.dtype)),  # type: ignore
        _numpy_array_constant(collapsed_val),
        dense_int_array(other_axes))  # type: ignore
    return out
  else:
    return _numpy_array_constant(val)

register_constant_handler(np.ndarray, _ndarray_constant_handler)

for _scalar_type in [np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64,
                     np.float16, np.float32, np.float64,
                     np.complex64, np.complex128,
                     np.bool_, np.longlong, dtypes.bfloat16]:
  register_constant_handler(_scalar_type, _ndarray_constant_handler)  # type: ignore

def _python_scalar_handler(dtype, val):
  return _numpy_array_constant(np.array(val, dtype))

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))

def _token_constant_handler(val):
  return hlo.create_token()
register_constant_handler(core.Token, _token_constant_handler)

# Attributes

AttributeHandler = Callable[[Any], ir.Attribute]
_attribute_handlers: dict[type[Any], AttributeHandler] = {}

def register_attribute_handler(type_: type[Any], handler_fun: AttributeHandler):
  _attribute_handlers[type_] = handler_fun

def get_attribute_handler(type_: type[Any]) -> AttributeHandler:
  return _attribute_handlers[type_]

def _numpy_scalar_attribute(val: Any) -> ir.Attribute:
  mlir_type = dtype_to_ir_type(val.dtype)
  if isinstance(mlir_type, ir.IntegerType):
    return ir.IntegerAttr.get(mlir_type, val)
  elif isinstance(mlir_type, ir.FloatType):
    return ir.FloatAttr.get(mlir_type, val)
  else:
    raise TypeError(f"Unsupported scalar attribute type: {type(val)}")

def _numpy_array_attribute(x: np.ndarray | np.generic) -> ir.Attribute:
  shape = x.shape
  if x.dtype == np.bool_:
    x = np.packbits(x, bitorder='little')  # type: ignore
  x = np.ascontiguousarray(x)
  element_type = dtype_to_ir_type(x.dtype)
  return ir.DenseElementsAttr.get(x, type=element_type, shape=shape)  # type: ignore

def _numpy_array_attribute_handler(val: np.ndarray | np.generic) -> ir.Attribute:
  if 0 in val.strides and val.size > 0:
    raise ValueError(
        "NumPy arrays with zero strides are not supported as MLIR attributes")
  if val.dtype == dtypes.float0:
    val = np.zeros(val.shape, dtype=np.bool_)
  if dtypes.is_python_scalar(val) or np.isscalar(val):
    return _numpy_scalar_attribute(val)
  else:
    return _numpy_array_attribute(val)

register_attribute_handler(np.ndarray, _numpy_array_attribute_handler)
register_attribute_handler(hashable_array.HashableArray,
                           lambda x: _numpy_array_attribute_handler(x.val))

for _scalar_type in [np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64,
                     np.float16, np.float32, np.float64,
                     np.complex64, np.complex128,
                     np.bool_, np.longlong, dtypes.bfloat16]:
  register_attribute_handler(_scalar_type, _numpy_array_attribute_handler)  # type: ignore

def _dtype_attribute_handler(dtype: np.dtype | np.generic) -> ir.Attribute:
  return ir.TypeAttr.get(dtype_to_ir_type(dtype))

register_attribute_handler(np.dtype, _dtype_attribute_handler)
register_attribute_handler(np.generic, _dtype_attribute_handler)

def _python_scalar_attribute_handler(dtype, val):
  return _numpy_scalar_attribute(np.array(val, dtype))

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_attribute_handler(
      ptype, partial(_python_scalar_attribute_handler, dtype))

register_attribute_handler(str, ir.StringAttr.get)
register_attribute_handler(bytes, ir.StringAttr.get)

def _dict_attribute_handler(val: dict[str, Any]) -> ir.Attribute:
  return ir.DictAttr.get({k: ir_attribute(v) for k, v in val.items()})

register_attribute_handler(dict, _dict_attribute_handler)

def _sequence_attribute_handler(val: Sequence[Any]) -> ir.Attribute:
  return ir.ArrayAttr.get([ir_attribute(v) for v in val])

register_attribute_handler(list, _sequence_attribute_handler)
register_attribute_handler(tuple, _sequence_attribute_handler)
register_attribute_handler(ir.Attribute, lambda x: x)
register_attribute_handler(ir.Type, lambda x: x)

def ir_attribute(val: Any) -> ir.Attribute:
  """Convert a Python value to an MLIR attribute."""
  for t in type(val).__mro__:
    handler = _attribute_handlers.get(t)
    if handler:
      out = handler(val)
      assert isinstance(out, ir.Attribute), (type(val), out)
      return out
  if hasattr(val, '__jax_array__'):
    return ir_attribute(val.__jax_array__())
  raise TypeError(f"No attribute handler defined for type: {type(val)}")

# Source locations

def get_canonical_source_file(file_name: str, caches: TracebackCaches) -> str:
  canonical_file_name = caches.canonical_name_cache.get(file_name, None)
  if canonical_file_name is not None:
    return canonical_file_name

  pattern = config.hlo_source_file_canonicalization_regex.value
  if pattern:
    file_name = re.sub(pattern, '', file_name)
  caches.canonical_name_cache[file_name] = file_name
  return file_name

def _is_user_file(ctx: ModuleContext, file_name: str) -> bool:
  is_user = ctx.traceback_caches.is_user_file_cache.get(file_name, None)
  if is_user is not None:
    return is_user
  out = source_info_util.is_user_filename(file_name)
  ctx.traceback_caches.is_user_file_cache[file_name] = out
  return out

def _traceback_to_location(ctx: ModuleContext, tb: xc.Traceback) -> ir.Location:
  """Converts a full traceback to a callsite() MLIR location."""
  loc = ctx.traceback_caches.traceback_cache.get(tb, None)
  if loc is not None:
    return loc

  frame_locs = []
  frames_limit = config.traceback_in_locations_limit.value
  frames_limit = frames_limit if frames_limit >= 0 else 1000

  codes, lastis = tb.raw_frames()
  for i, code in enumerate(codes):
    if not _is_user_file(ctx, code.co_filename):
      continue

    lasti = lastis[i]
    code_lasti = code, lasti
    loc = ctx.traceback_caches.location_cache.get(code_lasti, None)
    if loc is None:
      frame = source_info_util.raw_frame_to_frame(code, lasti)
      file_loc = ir.Location.file(
          get_canonical_source_file(frame.file_name, ctx.traceback_caches),
          frame.start_line,
          frame.start_column,
          frame.end_line,
          frame.end_column,
      )
      loc = ir.Location.name(frame.function_name, childLoc=file_loc)
      ctx.traceback_caches.location_cache[code_lasti] = loc
    frame_locs.append(loc)
    if len(frame_locs) >= frames_limit:
      break

  n = len(frame_locs)
  if n == 0:
    loc = ir.Location.unknown()
  elif n == 1:
    loc = frame_locs[0]
  else:
    loc = ir.Location.callsite(frame_locs[0], frame_locs[1:])
  ctx.traceback_caches.traceback_cache[tb] = loc
  return loc

def source_info_to_location(
    ctx: ModuleContext, primitive: core.Primitive | None,
    name_stack: source_info_util.NameStack,
    traceback: xc.Traceback | None) -> ir.Location:
  if config.include_full_tracebacks_in_locations.value:
    if traceback is None:
      loc = ir.Location.unknown()
    else:
      loc = _traceback_to_location(ctx, traceback)
  else:
    frame = source_info_util.user_frame(traceback)
    if frame is None:
      loc = ir.Location.unknown()
    else:
      loc = ir.Location.file(get_canonical_source_file(frame.file_name,
                                                       ctx.traceback_caches),
                             frame.start_line, frame.start_column)
  if primitive is None:
    if name_stack.stack:
      loc = ir.Location.name(str(name_stack), childLoc=loc)
  else:
    eqn_str = (
        f"{name_stack}/{primitive.name}" if name_stack.stack else primitive.name
    )
    loc = ir.Location.name(eqn_str, childLoc=loc)
    loc = ir.Location.name(f"{primitive.name}:", childLoc=loc)
  return loc

upstream_dialects = ir.DialectRegistry()
if _jax_mlir_ext:
  _jax_mlir_ext.register_dialects(upstream_dialects)

# Dumping MLIR modules
_ir_dump_counter = itertools.count()

def dump_module_to_file(module: ir.Module, stage_name: str) -> str | None:
  """Dumps the `module` IR to a file.

  Dumps the module if JAX_DUMP_IR_TO is defined.

  Args:
    module: The module to dump
    stage_name: A name to distinguish different stages of a module, will be
      appended to the `module.name`.

  Returns:
    The name of the file containing the dump if JAX_DUMP_IR_TO is defined and
    the module was dumped, `None` otherwise.
  """
  if not (out_dir := path.make_jax_dump_dir(config.jax_dump_ir_to.value)):
    return None
  modes = config.jax_dump_ir_modes.value.split(',')
  if 'stablehlo' not in modes:
    return None
  id = next(_ir_dump_counter)
  sym_name = module.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  name = f"jax_ir{id:04d}_{_make_string_safe_for_filename(module_name)}_{stage_name}.mlir"

  full_path = out_dir / name
  full_path.write_text(module_to_string(module))
  return name

def dump_module_message(module: ir.Module, stage_name: str) -> str:
  dumped_to = dump_module_to_file(module, stage_name)
  if dumped_to:
    return f"The module was dumped to {dumped_to}."
  else:
    return "Define JAX_DUMP_IR_TO to dump the module."

def _make_string_safe_for_filename(s: str) -> str:
  return re.sub(r'[^\w.)( -]', '', s)

def module_to_string(module: ir.Module, enable_debug_info=None) -> str:
  output = io.StringIO()
  if enable_debug_info is None:
    enable_debug_flag = str(config.jax_include_debug_info_in_dumps.value).lower()
    enable_debug_info = enable_debug_flag not in ('false', '0')
  module.operation.print(file=output, enable_debug_info=enable_debug_info)
  return output.getvalue()

def module_to_bytecode(module: ir.Module) -> bytes:
  output = io.BytesIO()
  module.operation.write_bytecode(file=output)
  return output.getvalue()

# Translation rules

# Create one global thread pool that can be shared between multiple ir.Contexts
# and enabling multi-threading
global_thread_pool = ir.ThreadPool()


class JaxIrContext(ir.Context):
  def __init__(self, *args, **kwargs):
    # Note: we're very intentionally *not* calling the __init__() of our
    # immediate superclass ir.Context, whose __init__() has the unfortunate side
    # effect of loading all the dialects linked into the binary into the
    # context. We want to ensure that only the dialects we need are loaded.
    super(ir.Context, self).__init__(*args, **kwargs)

def make_ir_context() -> ir.Context:
  """Creates an MLIR context suitable for JAX IR."""
  context = JaxIrContext()
  context.append_dialect_registry(upstream_dialects)
  context.load_all_available_dialects()

  context.set_thread_pool(global_thread_pool)
  dialects.sdy.register_dialect(context)
  dialects.mhlo.register_mhlo_dialect(context)
  dialects.chlo.register_dialect(context)
  dialects.hlo.register_dialect(context)
  return context


AxisContext = Union[
    sharding_impls.SPMDAxisContext,
    sharding_impls.ReplicaAxisContext,
    sharding_impls.ShardingContext,
]

class ShapePolyLoweringState:
  # The names of the dimension variables, sorted by name. This is the order in
  # which they are passed to the IR functions that need them. This is only
  # used for native serialization with polymorphic shapes when
  # --jax_dynamic_shapes is off.
  # TODO: for multi-platform lowering we prepend to the regular dimension
  # variables a fake dimension variable "platform_index_". This is a
  # temporary abuse, taking advantage that for platform index we need the
  # same lowering strategy as for dimension variables: add it as argument to
  # inner functions, and pass the values along at the call sites.
  dim_vars: tuple[str, ...]
  # Whether the module uses dimension variables, either in its inputs or
  # from an inner call to Exported modules that uses dimension variables.
  # This includes the case when the called Exported module uses a platform
  # index argument.
  uses_dim_vars: bool

  # If the first dimension variable is a platform index argument
  has_platform_index_argument: bool

  def __init__(self,
               dim_vars: tuple[str, ...],
               lowering_platforms: tuple[str, ...] | None):
    if lowering_platforms is not None and len(lowering_platforms) > 1:
      dim_vars = ("_platform_index",) + tuple(dim_vars)
      self.has_platform_index_argument = True
    else:
      self.has_platform_index_argument = False
    self.uses_dim_vars = (len(dim_vars) > 0)
    self.dim_vars = dim_vars

@dataclasses.dataclass(frozen=True)
class LoweringParameters:
  # A mapping between primitives and user-defined LoweringRules.
  # When lowering a primitive, give priority to the rule in this map over
  # existing Jax rules.
  override_lowering_rules: tuple[tuple[core.Primitive, LoweringRule]] | None = None

  # Signals that the entire computation being lowered operates on global
  # constants. This will result in adding jax.global_constant attributes
  # to the arguments of all functions that are created, e.g., floor_divide.
  # This is used only in export and jax2tf in presence of shape polymorphism
  # or multi-platform lowering.
  global_constant_computation: bool = False

  # Signals that we are lowering for exporting.
  for_export: bool = False
  # See usage in https://docs.jax.dev/en/latest/export/export.html#ensuring-forward-and-backward-compatibility
  # We have this here to ensure it is reflected in the cache keys
  export_ignore_forward_compatibility: bool = False
  # During lowering hoist the core.Literal constants as args for the main MLIR
  # function and all the intermediate functions that need them.
  # See https://docs.jax.dev/en/latest/internals/constants.html
  # TODO(necula): perhaps we can use `for_export` instead of this additional
  # field.
  hoist_constants_as_args: bool = config.use_simplified_jaxpr_constants.value


@dataclasses.dataclass
class TracebackCaches:
  traceback_cache: dict[xc.Traceback, ir.Location]
  location_cache: dict[tuple[types.CodeType, int], ir.Location]
  canonical_name_cache: dict[str, str]
  is_user_file_cache: dict[str, bool]

  def __init__(self):
    self.traceback_cache = {}
    self.location_cache = {}
    self.canonical_name_cache = {}
    self.is_user_file_cache = {}


@dataclasses.dataclass(frozen=True)
class LoweringCacheKey:
  primitive: core.Primitive
  eqn_ctx: core.JaxprEqnContext
  avals_in: tuple[core.AbstractValue, ...]
  effects: effects_lib.Effects
  params: frozen_dict.FrozenDict[str, Any]
  platforms: tuple[str, ...]

@dataclasses.dataclass(frozen=True)
class LoweringCacheValue:
  func: func_dialect.FuncOp
  output_types: Sequence[IrTypes]
  const_args: list[ArrayLike]  # The hoisted constants expected by `func`
  inline: bool  # Inline calls to this lowered function?

@dataclasses.dataclass
class ModuleContext:
  """Module-wide context information for MLIR lowering."""
  context: ir.Context
  module: ir.Module
  ip: ir.InsertionPoint
  symbol_table: ir.SymbolTable
  # The lowering platforms for the module. Can be more than one only when
  # exporting.
  platforms: Sequence[str]
  # See ModuleContext.get_backend() for backend and platforms usage.
  backend: xb.XlaBackend | None
  axis_context: AxisContext
  keepalives: list[Any]
  channel_iterator: Iterator[int]
  host_callbacks: list[Any]
  # Keep state for the lowering of shape polymorphism
  shape_poly_state: ShapePolyLoweringState
  all_default_mem_kind: bool

  # Cached primitive lowerings.
  lowering_cache: dict[LoweringCacheKey, LoweringCacheValue]
  cached_primitive_lowerings: dict[Any, func_dialect.FuncOp]

  # Cached traceback information.
  traceback_caches: TracebackCaches

  lowering_parameters: LoweringParameters

  @property
  def axis_env(self) -> sharding_impls.AxisEnv:
    return self.axis_context.axis_env

  def __init__(
      self,
      *,
      platforms: Sequence[str],
      backend: xb.XlaBackend | None,
      axis_context: AxisContext,
      keepalives: list[Any],
      channel_iterator: Iterator[int],
      host_callbacks: list[Any],
      lowering_parameters: LoweringParameters,
      context: ir.Context | None = None,
      module: ir.Module | None = None,
      ip: ir.InsertionPoint | None = None,
      symbol_table: ir.SymbolTable | None = None,
      lowering_cache: None | dict[LoweringCacheKey, Any] = None,
      cached_primitive_lowerings: None | dict[Any, func_dialect.FuncOp] = None,
      traceback_caches: None | TracebackCaches = None,
      shape_poly_state = None,
      all_default_mem_kind: bool = True):

    self.context = context or make_ir_context()
    self.module = module or ir.Module.create(loc=ir.Location.unknown(self.context))
    self.ip = ip or ir.InsertionPoint(self.module.body)
    self.symbol_table = symbol_table or ir.SymbolTable(self.module.operation)
    self.backend = backend
    self.platforms = platforms
    self.axis_context = axis_context
    self.lowering_cache = ({} if lowering_cache is None else lowering_cache)
    self.cached_primitive_lowerings = ({} if cached_primitive_lowerings is None
                                       else cached_primitive_lowerings)
    self.traceback_caches = (TracebackCaches() if traceback_caches is None
                             else traceback_caches)
    self.channel_iterator = channel_iterator
    self.keepalives = keepalives
    self.host_callbacks = host_callbacks
    self.shape_poly_state = (
      shape_poly_state or ShapePolyLoweringState((), tuple(platforms)))
    self.all_default_mem_kind = all_default_mem_kind
    self.lowering_parameters = lowering_parameters

  def get_backend(self) -> xb.XlaBackend:
    if len(self.platforms) > 1:
      raise NotImplementedError(
        "accessing .backend in multi-lowering setting. This can occur when "
        "lowering a primitive that has not been adapted to multi-platform "
        "lowering")
    if self.backend is not None:
      if xb.canonicalize_platform(self.backend.platform) != self.platforms[0]:
        raise ValueError(
          "the platform for the specified backend "
          f"{xb.canonicalize_platform(self.backend.platform)} is different "
          f"from the lowering platform {self.platforms[0]}")
      return self.backend
    return xb.get_backend(self.platforms[0])

  def new_channel(self) -> int:
    channel = next(self.channel_iterator)
    # `xla::HostCallback` requires a 16-bit channel ID.
    if channel >= (1 << 16):
      raise RuntimeError(
          "Host callback lowering created too many channels. PjRt does not"
          " support more than 65535 channels")
    return channel

  # Adds an IFRT host callback object to the context. A reference to these
  # callbacks will be provided to IFRT during compilation so it can do things
  # like serialize them and keep them alive.
  def add_host_callback(self, host_callback: Any) -> None:
    self.host_callbacks.append(host_callback)

  # Keeps a value alive as long as the Python executable is alive.
  # TODO(phawkins): this feature is problematic, because you almost certainly
  # want to keep alive values as long as the underlying runtime executable is
  # still alive/executing. The Python executable object may have a shorter
  # lifetime, so it's highly likely any caller of this method is buggy.
  def add_keepalive(self, keepalive: Any) -> None:
    self.keepalives.append(keepalive)

  def replace(self, **kw): return dataclasses.replace(self, **kw)


@dataclasses.dataclass
class LoweringRuleContext:
  """Per-rule context information for MLIR lowering."""
  module_context: ModuleContext
  # Even though we assigned name_stack entries to each jaxpr equation during
  # tracing, we need to propagate name stacks during lowering as well because
  # lowering may effectively inline multiple jaxprs into a single HLO function.
  # For example, the body of a while loop needs the name stack of the enclosing
  # while instruction to be prepended when forming its HLO name.
  name_stack: source_info_util.NameStack
  traceback: xc.Traceback | None
  primitive: core.Primitive | None
  avals_in: Sequence[core.AbstractValue]
  avals_out: Any  # Usually Sequence[core.AbstractValue], but sometimes None.
  tokens_in: TokenSet
  tokens_out: TokenSet | None  # Mutable store for output containers
  # The values tobe used for the Literal constants, by id of the const.
  # This is used to implement passing along the constants that have been
  # hoisted as main function arguments down to where they are used.
  # See https://docs.jax.dev/en/latest/internals/constants.html
  const_lowering: dict[int, IrValues]
  axis_size_env: dict[core.Var, ir.Value] | None = None  # Dynamic axis sizes
  # The values for the dimension variables in same order as
  # module_context.shape_poly_state.dim_vars
  dim_var_values: Sequence[ir.Value] = ()
  jaxpr_eqn_ctx: core.JaxprEqnContext | None = None
  # Override module_context.platforms if not None. Used during multi-platform
  # lowering, when in a scope with a subset of the module_context.platforms.
  platforms: Sequence[str] | None = None

  def set_tokens_out(self, tokens_out: TokenSet):
    assert self.tokens_out is None, 'Should only set `tokens_out` once.'
    self.tokens_out = tokens_out

  def replace(self, **kw): return dataclasses.replace(self, **kw)  # pytype: disable=wrong-arg-types  # dataclasses-replace-types

  def is_forward_compat(self) -> bool:
    """Returns true if the lowering parameters are in forward compatibility mode.
    """
    lowering_parameters = self.module_context.lowering_parameters

    check_platforms: Sequence[str] = (
        self.platforms or self.module_context.platforms
    )
    force_forward_compat = any(
        p in xb.FORCE_FORWARD_COMPAT_LOWERING_PLATFORMS for p in check_platforms
    )

    return (
        lowering_parameters.for_export or force_forward_compat
    ) and not lowering_parameters.export_ignore_forward_compatibility


if not MYPY:
  class LoweringRule(Protocol):
    def __call__(self, ctx: LoweringRuleContext,
                 *args: ir.Value | Sequence[ir.Value],
                 **kw) -> Sequence[ir.Value | Sequence[ir.Value]]:
      """Converts a JAX primitive invocation into MLIR."""
else:
  LoweringRule = Any

@dataclasses.dataclass(frozen=True)
class LoweringRuleEntry:
  rule: LoweringRule
  inline: bool

_lowerings: dict[core.Primitive, LoweringRuleEntry] = {}
_platform_specific_lowerings: dict[str, dict[core.Primitive, LoweringRuleEntry]]
_platform_specific_lowerings = collections.defaultdict(dict)

def register_lowering(prim: core.Primitive, rule: LoweringRule,
                      platform: str | None = None, inline: bool = True,
                      cacheable: bool = True) -> None:
  """Registers a lowering rule for a primitive.

  Args:
    prim: The primitive to register the rule for.
    rule: The lowering rule to register.
    platform: The platform to register the rule for. If None, this is a common
      rule applicable to all platforms. Platform-specific rules take precedence
      over common rules.
    inline: Whether to emit the lowering inline. If False, the lowering will be
      emitted in a separate function, called by similar instances of the
      lowering.
    uncacheable: Whether this primitive's lowering can be cached. This is a
      temporary flag that will be removed after primitives that have problems
      with caching are fixed.
  """
  assert not isinstance(rule, LoweringRuleEntry)
  if not cacheable:
    _uncacheable_primitives.add(prim)
  if platform is None:
    _lowerings[prim] = LoweringRuleEntry(rule, inline)
  else:
    if not xb.is_known_platform(platform):
      known_platforms = sorted(xb.known_platforms())
      raise NotImplementedError(
          f"Registering an MLIR lowering rule for primitive {prim}"
          f" for an unknown platform {platform}. Known platforms are:"
          f" {', '.join(known_platforms)}.")
    # For backward compatibility reasons, we allow rules to be registered
    # under "gpu" even though the platforms are now called "cuda" and "rocm".
    # TODO(phawkins): fix up users to specify either "cuda" or "rocm" and remove
    # this expansion.
    for p in xb.expand_platform_alias(platform):
      _platform_specific_lowerings[p][prim] = LoweringRuleEntry(rule, inline)


def flatten_ir_values(xs: Iterable[IrValues]) -> list[ir.Value]:
  """Concatenates/flattens a list of ir.Values or ir.Value sequences."""
  out = []
  for x in xs:
    if isinstance(x, ir.Value):
      out.append(x)
    else:
      out.extend(x)
  return out

def flatten_ir_types(xs: Iterable[IrTypes]) -> list[ir.Type]:
  """Concatenates/flattens a list of ir.Types or ir.Type sequences."""
  out = []
  for x in xs:
    if isinstance(x, ir.Type):
      out.append(x)
    else:
      out.extend(x)
  return out

_unflatten_done = object()

def unflatten_ir_types(xs: Iterable[ir.Type], ns: Sequence[int]) -> list[IrTypes]:
  """Splits `xs` into subsequences of lengths `ns`.

  Unlike `split_list`, the `sum(ns)` must be equal to `len(xs)`, and if n == 1
  then types are not wrapped in a singleton list."""
  xs_iter = iter(xs)
  unflattened: list[IrTypes]
  unflattened = [next(xs_iter) if n == 1 else tuple(next(xs_iter)
                 for _ in range(n)) for n in ns]
  assert next(xs_iter, _unflatten_done) is _unflatten_done
  return unflattened

def len_ir_types(x: IrTypes) -> int:
  return 1 if isinstance(x, ir.Type) else len(x)

def unflatten_ir_values_like_types(xs: Iterable[ir.Value],
                                   ys: Sequence[IrTypes]) -> list[IrValues]:
  """Splits `xs` into subsequences of lengths `ns`.

  Unlike `split_list`, the `sum(ns)` must be equal to `len(xs)`, and if n == 1
  then values are not wrapped in a singleton list."""
  xs_iter = iter(xs)
  unflattened: list[IrValues]
  unflattened = [next(xs_iter) if isinstance(y, ir.Type) else
                 tuple(next(xs_iter) for _ in range(len(y)))
                 for y in ys]
  assert next(xs_iter, _unflatten_done) is _unflatten_done
  return unflattened


_module_name_regex = re.compile(r"[^\w.-]")

def sanitize_name(name: str) -> str:
  """Ensure a name is usable as module or function name."""
  return _module_name_regex.sub("_", name)


def sharded_aval(aval: core.AbstractValue,
                 sharding: JSharding | AUTO | None) -> core.AbstractValue:
  """Returns the new aval sharded based on sharding proto."""
  if sharding is None:
    return aval
  if isinstance(sharding, AUTO):
    return aval
  if isinstance(aval, core.AbstractToken):
    return aval
  if not isinstance(aval, (core.ShapedArray, core.DShapedArray)):
    raise NotImplementedError
  return aval.update(sharding.shard_shape(aval.shape), sharding=None)  # type: ignore


def eval_dynamic_shape(ctx: LoweringRuleContext,
                       shape: core.Shape) -> tuple[int | Value, ...]:
  if config.dynamic_shapes.value:
    assert ctx.axis_size_env is not None
    return tuple(ctx.axis_size_env.get(d, d) for d in shape)  # type: ignore
  else:
    ctx = ctx.replace(
        primitive="eval_dynamic_shape",
        avals_in=[core.dim_value_aval()] * len(ctx.module_context.shape_poly_state.dim_vars),
        tokens_out=None)

    res = lower_fun(
        partial(core.evaluate_shape, shape, ctx.module_context.shape_poly_state.dim_vars),
        multiple_results=True)(ctx, *ctx.dim_var_values)
    return tuple(operator.index(d) if core.is_constant_dim(d) else d_ir
                 for d, d_ir in zip(shape, flatten_ir_values(res)))

# TODO: replace usage of eval_dynamic_shape_as_vals with eval_dynamic_shape_as_ivals
def eval_dynamic_shape_as_vals(ctx: LoweringRuleContext,
                               shape: core.Shape) -> tuple[Value, ...]:
  """Evaluates the dynamic shapes as int32 values."""
  def convert_dim(d: int | Value):
    if type(d) is int:
      return ir_constant(np.array(d, dtype=np.int32))
    else:
      i32_type = aval_to_ir_type(core.ShapedArray((), np.int32))
      if d.type != i32_type:  # type: ignore
        return hlo.convert(i32_type, d)
      else:
        return d
  return tuple(convert_dim(v) for v in eval_dynamic_shape(ctx, shape))


def eval_dynamic_shape_as_ivals(
    ctx: LoweringRuleContext, shape: core.Shape
    ) -> tuple[int | Value, ...]:
  """Evaluates the dynamic shapes as int or ir.int32 values."""
  def convert_dim(d: int | Value) -> int | ir.Value:
    if type(d) is int:
      return d
    else:
      i32_type = aval_to_ir_type(core.ShapedArray((), np.int32))
      if d.type != i32_type:  # type: ignore
        return hlo.convert(i32_type, d)
      else:
        return d
  return tuple(convert_dim(v) for v in eval_dynamic_shape(ctx, shape))

def eval_dynamic_shape_as_tensor(ctx: LoweringRuleContext,
                                 shape: core.Shape) -> Value:
  """Evaluates the dynamic shapes as one 1d int32 tensor."""
  return shape_tensor(eval_dynamic_shape(ctx, shape))

class LoweringResult(NamedTuple):
  module: ir.Module
  keepalive: Any | None
  host_callbacks: list[Any]
  shape_poly_state: ShapePolyLoweringState


_platforms_with_donation = ["cpu", "cuda", "rocm", "tpu", "neuron"]


def add_manual_axes(axis_ctx: sharding_impls.SPMDAxisContext, sharding, ndim):
  mesh = axis_ctx.mesh.abstract_mesh
  sharding_mesh = sharding.mesh.abstract_mesh
  if (isinstance(sharding, sharding_impls.NamedSharding) and
      sharding_mesh.shape == mesh.shape):
    out_mesh, spec = sharding_mesh, sharding.spec
  else:
    out_mesh, spec = mesh, sharding_impls.parse_flatten_op_sharding(
        sharding._to_xla_hlo_sharding(ndim), mesh)[0]

  out_mesh = out_mesh.update_axis_types(
      {a: AxisType.Manual for a in axis_ctx.manual_axes})
  out = sharding_impls.NamedSharding(out_mesh, spec,
                                     memory_kind=sharding.memory_kind)
  manual_axes = out.mesh.manual_axes
  if any(p in manual_axes for s in out.spec
         if s is not None and s is not PartitionSpec.UNCONSTRAINED
         for p in (s if isinstance(s, tuple) else (s,))):
    raise ValueError(
        f'pspec {out.spec} contains a manual axes {manual_axes} of mesh'
        f' which is not allowed. If you are using a'
        ' with_sharding_constraint under a shard_map, only use the'
        ' mesh axis in PartitionSpec which are not manual.')
  return out


def _to_physical_op_sharding(
    ctx: ModuleContext,
    aval: core.AbstractValue, sharding: JSharding | AUTO | None,
) -> xc.OpSharding | SdyArray | None:
  if sharding is None:
    return None
  if all_unconstrained(sharding, aval):
    return None
  if isinstance(sharding, AUTO):
    if config.use_shardy_partitioner.value:
      return sharding._to_sdy_sharding(aval.ndim)  # type: ignore
    return None
  assert isinstance(sharding, JSharding)
  if isinstance(aval, AbstractRef):
    return _to_physical_op_sharding(ctx, aval.inner_aval, sharding)
  assert isinstance(aval, (core.ShapedArray, core.DShapedArray))
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    sharding = sharding_impls.physical_sharding(aval, sharding)
    aval = core.physical_aval(aval)
  axis_ctx = ctx.axis_context
  if (isinstance(axis_ctx, sharding_impls.SPMDAxisContext) and
      axis_ctx.manual_axes):
    sharding = add_manual_axes(axis_ctx, sharding, aval.ndim)
  if config.use_shardy_partitioner.value:
    return sharding._to_sdy_sharding(aval.ndim)  # type: ignore
  return sharding._to_xla_hlo_sharding(aval.ndim).to_proto()  # type: ignore


def _to_xla_layout(layout: Layout | None | AutoLayout,
                   aval: core.AbstractValue) -> str | None:
  if layout is None:
    return None
  if isinstance(layout, AutoLayout):
    return "auto"
  if aval is core.abstract_token:
    return None
  return str(layout._to_xla_layout(aval.dtype))  # type: ignore


def _get_mem_kind(s: JSharding | AUTO | None) -> str | None:
  if s is None:
    return None
  if isinstance(s, AUTO):
    return None
  assert isinstance(s, JSharding)
  return s.memory_kind


def contains_unconstrained(s):
  return (isinstance(s, NamedSharding)
          and PartitionSpec.UNCONSTRAINED in s.spec)


def all_unconstrained(s, aval):
  if isinstance(s, NamedSharding):
    if aval.ndim != len(s.spec):
      return False
    return all(p is PartitionSpec.UNCONSTRAINED for p in s.spec)
  return False

class UnconstrainedVariants(NamedTuple):
  contains_unconstrained: bool
  all_unconstrained: bool

def _get_unconstrained_variants(s, aval) -> UnconstrainedVariants:
  us = contains_unconstrained(s)
  return UnconstrainedVariants(
      contains_unconstrained=us, all_unconstrained=all_unconstrained(s, aval))


def check_jaxpr_constants(closed_jaxpr: core.ClosedJaxpr):
  """Check if a JAXPR contains an excessive amount of constants, if so, report where they were captured"""
  if (threshold := config.captured_constants_warn_bytes.value) == -1:
    return

  # need the unaesthetic getter here as some of the consts in the test suite are arbitrary objects
  total_iter, nbytes_iter = itertools.tee(
      map(lambda c: getattr(c, "nbytes", 0), closed_jaxpr.consts)
  )

  if (total_bytes := sum(total_iter)) < threshold:
    return

  message = (
      "A large amount of constants were captured during lowering"
      f" ({util.pprint_bytes(total_bytes)} total). If this is intentional,"
      " disable this warning by setting JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1. "
  )

  if not (num_frames := config.captured_constants_report_frames.value):
    message += (
        "To obtain a report of where these constants were encountered, "
        "set JAX_CAPTURED_CONSTANTS_REPORT_FRAMES=-1."
    )
    warnings.warn(message)
    return

  message += (
      "The subsequent report may be disabled by setting JAX_CAPTURED_CONSTANTS_REPORT_FRAMES=0.\n\n"
      f"Largest {min(num_frames, len(closed_jaxpr.consts))} allocation(s):\n"
  )
  try:
    nbytes_var_const = zip(nbytes_iter, closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts)
    for nbytes, var, const in heapq.nlargest(5, nbytes_var_const, key=operator.itemgetter(0)):
      message += f"  Constant {type(const)}, {var.aval.str_short()}, {util.pprint_bytes(nbytes)} captured at:\n"

      for eqn in jaxpr_util.eqns_using_var(closed_jaxpr.jaxpr, var):
        call_frame_source_info = source_info_util.summarize(eqn.source_info, num_frames)
        message += "  " * 2 + call_frame_source_info.replace("\n", "\n" + "  " * 2) + "\n\n"

    warnings.warn(message)
  except Exception as exc:
    warnings.warn(message + f" Exception raised while generating report: {exc}")


def lower_jaxpr_to_module(
    module_name: str,
    jaxpr: core.ClosedJaxpr,
    *,
    num_const_args: int,
    in_avals: Sequence[core.AbstractValue],
    ordered_effects: list[core.Effect],
    # See ModuleContext.get_backend() for backend and platforms usage.
    platforms: Sequence[str],
    backend: xb.XlaBackend | None,
    axis_context: AxisContext,
    donated_args: Sequence[bool],
    replicated_args: Sequence[bool] | None = None,
    arg_shardings: Sequence[JSharding | AUTO | None] | None = None,
    result_shardings: Sequence[JSharding | AUTO | None] | None = None,
    in_layouts: Sequence[Layout | None | AutoLayout] | None = None,
    out_layouts: Sequence[Layout | None | AutoLayout] | None = None,
    arg_names: Sequence[str] | None = None,
    result_names: Sequence[str] | None = None,
    num_replicas: int = 1,
    num_partitions: int = 1,
    all_default_mem_kind: bool = True,
    input_output_aliases: None | tuple[int | None, ...] = None,
    propagated_out_mem_kinds: tuple[None | str, ...] | None = None,
    lowering_parameters: LoweringParameters,
) -> LoweringResult:
  """Lowers a top-level jaxpr to an MLIR module.

  Handles the quirks of the argument/return value passing conventions of the
  runtime.
  The inputs already account for the constant arguments.
  See https://docs.jax.dev/en/latest/internals/constants.html
  """
  util.test_event("lower_jaxpr_to_module")
  platforms = tuple(map(xb.canonicalize_platform, platforms))

  sharded_in_avals = (in_avals if arg_shardings is None else
                      map(sharded_aval, in_avals, arg_shardings))
  sharded_out_avals = (jaxpr.out_avals if result_shardings is None else
                       map(sharded_aval, jaxpr.out_avals, result_shardings))
  if all_default_mem_kind:
    arg_memory_kinds = None
    result_memory_kinds = None
  else:
    arg_memory_kinds = (map(_get_mem_kind, arg_shardings)
                        if arg_shardings is not None else None)
    result_memory_kinds = (map(_get_mem_kind, result_shardings)
                          if result_shardings is not None else None)

  # TODO(yashkatariya): Simplify the donation logic.
  xla_donated_args = None
  platforms_with_donation = [p for p in platforms
                             if p in _platforms_with_donation]
  if platforms_with_donation:
    if len(platforms_with_donation) != len(platforms) and (
        xla_donated_args or any(donated_args)):
      raise NotImplementedError(
        "In multi-platform lowering either all or no lowering platforms "
        f"should support donation. Lowering for {platforms} of which "
        f"only {platforms_with_donation} support donation")
    input_output_aliases, donated_args, xla_donated_args = _set_up_aliases(
        input_output_aliases, sharded_in_avals, sharded_out_avals, donated_args,
        arg_memory_kinds, result_memory_kinds, in_layouts, out_layouts,
        result_shardings if num_partitions > 1 else None)
    if (num_partitions > 1 and
        (result_shardings is None or
         any(s is None or isinstance(s, AUTO) or contains_unconstrained(s)
             for s in result_shardings))):
      if xla_donated_args is None:
        xla_donated_args = [False] * len(donated_args)
      for input_id in range(len(donated_args)):
        if donated_args[input_id]:
          xla_donated_args[input_id] = True
          donated_args[input_id] = False
  if any(donated_args):
    unused_donations = [str(a) for a, d in zip(sharded_in_avals, donated_args) if d]
    msg = "See an explanation at https://docs.jax.dev/en/latest/faq.html#buffer-donation."
    if not platforms_with_donation:
      msg = f"Donation is not implemented for {platforms}.\n{msg}"
    if unused_donations:
      warnings.warn("Some donated buffers were not usable:"
                    f" {', '.join(unused_donations)}.\n{msg}")
  # Delete donated_args by default here, since it's not needed beyond this point
  del donated_args

  unlowerable_effects = lowerable_effects.filter_not_in(jaxpr.effects)
  if unlowerable_effects:
    raise ValueError(f'Cannot lower jaxpr with effects: {jaxpr.effects}')

  # HLO channels need to start at 1
  channel_iter = itertools.count(1)
  # Create a keepalives list that will be mutated during the lowering.
  keepalives: list[Any] = []
  host_callbacks: list[Any] = []

  dim_vars: Sequence[str]
  if not config.dynamic_shapes.value:
    # Find the dimension variables
    all_dim_poly = [d for aval in sharded_in_avals if hasattr(aval, "shape")
                    for d in aval.shape if not core.is_constant_dim(d)]
    dim_vars = tuple(sorted(functools.reduce(lambda acc, new: acc.union(new._get_vars()),
                                             all_dim_poly, set())))
  else:
    dim_vars = ()

  ctx = ModuleContext(backend=backend,
                      platforms=platforms, axis_context=axis_context,
                      keepalives=keepalives,
                      channel_iterator=channel_iter,
                      host_callbacks=host_callbacks,
                      lowering_parameters=lowering_parameters,
                      shape_poly_state=ShapePolyLoweringState(dim_vars, platforms),
                      all_default_mem_kind=all_default_mem_kind)
  with ctx.context, ir.Location.unknown(ctx.context):
    # Remove module name characters that XLA would alter. This ensures that
    # XLA computation preserves the module name.
    attrs = ctx.module.operation.attributes
    attrs["sym_name"] = ir.StringAttr.get(
        sanitize_name(module_name).rstrip("_"))
    attrs["mhlo.num_replicas"] = i32_attr(num_replicas)
    attrs["mhlo.num_partitions"] = i32_attr(num_partitions)
    lower_jaxpr_to_fun(
        ctx, module_name, jaxpr, ordered_effects,
        num_const_args=num_const_args,
        main_function=True,
        replicated_args=replicated_args,
        in_avals=in_avals,
        arg_shardings=arg_shardings,
        result_shardings=result_shardings,
        input_output_aliases=input_output_aliases,
        xla_donated_args=xla_donated_args,
        arg_names=arg_names,
        result_names=result_names,
        arg_memory_kinds=arg_memory_kinds,
        result_memory_kinds=result_memory_kinds,
        arg_layouts=in_layouts,
        result_layouts=out_layouts,
        propagated_out_mem_kinds=propagated_out_mem_kinds)

  try:
    if not ctx.module.operation.verify():
      raise ValueError(
          "Cannot lower jaxpr with verifier errors. " +
          dump_module_message(ctx.module, "verification"))
  except ir.MLIRError as e:
    msg_lines = ["Cannot lower jaxpr with verifier errors:"]
    def emit_diagnostic_info(d):
      msg_lines.append(f"\t{d.message}")
      msg_lines.append(f"\t\tat {d.location}")
      for n in d.notes:
        emit_diagnostic_info(n)
    for d in e.error_diagnostics:
      emit_diagnostic_info(d)
    raise ValueError("\n".join(msg_lines) + "\n" +
                     dump_module_message(ctx.module, "verification")) from e

  with ctx.context:
    # Cached lowering rule evaluation leaves dead functions. Remove them.
    pipeline = passmanager.PassManager.parse(
        'builtin.module(symbol-dce)')
    pipeline.run(ctx.module.operation)

    if config.use_shardy_partitioner.value:
      pipeline = passmanager.PassManager.parse(
          'builtin.module(sdy-lift-inlined-meshes)')
      pipeline.run(ctx.module.operation)

  util.test_event("mlir.collect_lowered_jaxprs", jaxpr, ctx.module)
  return LoweringResult(ctx.module, ctx.keepalives, ctx.host_callbacks,
                        ctx.shape_poly_state)

def _set_up_aliases(input_output_aliases, avals_in, avals_out,
                    donated_args, arg_memory_kinds, result_memory_kinds,
                    in_layouts, out_layouts, result_shardings):
  if input_output_aliases is None:
    input_output_aliases = [None] * len(avals_in)
  else:
    input_output_aliases = list(input_output_aliases)
  # To match-up in-avals to out-avals we only care about the number of
  # bytes, so we strip off unrelated aval metadata (eg. the named shape)
  strip_metadata = lambda a: (a if a is core.abstract_token else
                              core.ShapedArray(a.shape, a.dtype))
  avals_in = map(strip_metadata, avals_in)
  avals_out = map(strip_metadata, avals_out)

  # Both arg and result memory kinds need to be specified to donate based on
  # the memory kind. For jit's where out_shardings is not specified, we don't
  # know the memory kind so don't condition the logic based on the memory kind.
  # TODO(yashkatariya): Note that this logic should be in C++ where we make
  # donation decisions are made after SPMD propagation passes and memory
  # placement passes so that we have all the information.
  if (arg_memory_kinds is None or result_memory_kinds is None or
      any(a is None for a in arg_memory_kinds) or
      any(r is None for r in result_memory_kinds)):
    arg_memory_kinds = [None] * len(avals_in)
    result_memory_kinds = [None] * len(avals_out)

  donations = collections.defaultdict(collections.deque)
  for i, (aval, am, donated, aliased) in enumerate(
      zip(avals_in, arg_memory_kinds, donated_args, input_output_aliases)):
    if donated and aliased is None:
      donations[(aval, am)].append(i)

  xla_donated_args = None
  out_donated_args = list(donated_args)
  in_out_layout_not_none = in_layouts is not None and out_layouts is not None
  for i, (aval, rm) in enumerate(zip(avals_out, result_memory_kinds)):
    # Only donate if memory kinds match. Relax this when the compiler can
    # donate across memories.
    key = (aval, rm)
    if donations.get(key, ()):
      input_id = donations[key].popleft()
      out_donated_args[input_id] = False
      if (in_out_layout_not_none and
          isinstance(in_layouts[input_id], AutoLayout) and
          not isinstance(out_layouts[i], AutoLayout)):
        raise ValueError(
            f"Input layout being donated was {in_layouts[input_id]} while"
            f" output layout was {out_layouts[i]}. Did you mean to set the"
            " **output layout** to **Layout.AUTO**?\nThis will"
            " allow for the input and output layout to be chosen by XLA and"
            " not the layout of the output which might not be optimal.")
      if (in_out_layout_not_none and
          not isinstance(in_layouts[input_id], AutoLayout) and
          isinstance(out_layouts[i], AutoLayout)):
        raise ValueError(
            f"Input layout being donated was {in_layouts[input_id]} while"
            f" output layout was {out_layouts[i]}. Did you mean to set the"
            " **input layout** to **Layout.AUTO**?\nThis will allow"
            " for the input and output layout to be chosen by XLA and not the"
            " layout of the input which might not be optimal.")
      if (in_layouts is None or out_layouts is None or
          in_layouts[input_id] == out_layouts[i]) and (
              result_shardings is None or not (
              (s := result_shardings[i]) is None or
              isinstance(s, AUTO) or contains_unconstrained(s))):
        input_output_aliases[input_id] = i
      else:
        # Fallback to xla donation if layouts don't match.
        if xla_donated_args is None:
          xla_donated_args = [False] * len(avals_in)
        xla_donated_args[input_id] = True

  return input_output_aliases, out_donated_args, xla_donated_args

Token = ir.Value
token_type = hlo.TokenType.get
create_token = hlo.create_token

class TokenSet:
  """An immutable container of tokens to be used to lower effectful jaxprs. When lowering
  effectful jaxprs, we need to thread HLO tokens to sequence them. Each effect
  will need its own token that will be threaded in and out of the effectful
  primitives. A `TokenSet` encapsulates a set of HLO tokens that will be
  used by the lowering rules.
  """
  _tokens: collections.OrderedDict[core.Effect, Token]

  def __init__(self, *args, **kwargs):
    self._tokens = collections.OrderedDict(*args, **kwargs)

  def __len__(self):
    return len(self._tokens)

  def get(self, effect: core.Effect) -> Token:
    return self._tokens[effect]

  @classmethod
  def create(cls, effects: Sequence[core.Effect]) -> TokenSet:
    """Creates a `TokenSet` corresponding to a list of `core.Effect`s."""
    tokens = [create_token() for _ in effects]
    return TokenSet(zip(effects, tokens))

  def items(self) -> Sequence[tuple[core.Effect, Token]]:
    return tuple(self._tokens.items())

  def effects(self) -> set[core.Effect]:
    return set(self._tokens.keys())

  def subset(self, effects: Sequence[core.Effect]) -> TokenSet:
    """Return a subset of the `TokenSet` restricted to a set of `core.Effect`s."""
    return TokenSet((eff, self._tokens[eff]) for eff in effects)

  def update_tokens(self, tokens: TokenSet) -> TokenSet:
    """Returns a new `TokenSet` with tokens replaced with ones from the input `TokenSet`."""
    new_tokens = []
    for eff in self.effects():
      if eff in tokens._tokens:
        new_tokens.append((eff, tokens._tokens[eff]))
      else:
        new_tokens.append((eff, self._tokens[eff]))
    return TokenSet(new_tokens)

def lower_jaxpr_to_fun(
    ctx: ModuleContext,
    name: str,
    jaxpr: core.ClosedJaxpr,
    effects: Sequence[core.Effect],
    *,
    num_const_args: int,
    main_function: bool = False,
    replicated_args: Sequence[bool] | None = None,
    in_avals: Sequence[core.AbstractValue],
    arg_shardings: Sequence[JSharding | AUTO | None] | None = None,
    result_shardings: Sequence[JSharding | AUTO | None] | None = None,
    use_sharding_annotations: bool = True,
    input_output_aliases: Sequence[int | None] | None = None,
    xla_donated_args: Sequence[bool] | None = None,
    arg_names: Sequence[str | None] | None = None,
    result_names: Sequence[str] | None = None,
    arg_memory_kinds: Sequence[str | None] | None = None,
    result_memory_kinds: Sequence[str | None] | None = None,
    arg_layouts: Sequence[Layout | None | AutoLayout] | None = None,
    result_layouts: Sequence[Layout | None | AutoLayout] | None = None,
    propagated_out_mem_kinds: tuple[None | str, ...] | None = None,
) -> func_dialect.FuncOp:
  """Lowers jaxpr and its callees to an IR function.

  Assumes that an MLIR context, location, and insertion point are set.

  Note: this function does *not* take a name stack. Name stacks do not cross
  the boundaries of HLO functions.

  Args:
    ctx: the lowering context.
    name: the function name. The name will be uniquified by the symbol table,
      so it is ok to use the same name multiple times.
    jaxpr: the jaxpr to lower.
    effects: a sequence of `core.Effect`s corresponding to an ordering of tokens
      that will be created in or used by the lowered function.
    num_const_args: how many constant arguments is this function going to have.
      See https://docs.jax.dev/en/latest/internals/constants.html
    main_function: if true, this is the main function in the module. This has
      several effects:
      * the function's visibility is set to "public".
      * the function's symbol name will be "main"
      * the function's name will be used as the root name stack entry.
    replicated_args: if present, annotates arguments as replicated.
    arg_shardings: sharding annotations for each argument (optional).
    result_shardings: sharding annotations for each result (optional).
    use_sharding_annotations: if True, use "mhlo.sharding" annotations on
      parameters and return values to express sharding. If False, use
      hlo.custom_call operators with sharding annotations.
      TODO(b/228598865): remove this option when "mhlo.sharding" annotations are
      propagated on non-entry functions during MLIR->HLO conversion.
    input_output_aliases: optional sequence that maps argument numbers to the
      corresponding output that should alias them.
    xla_donated_args: optional sequence of args to set donation annotations.
  Returns:
    MLIR func op
  """
  util.test_event("lower_jaxpr_to_fun", name)
  check_jaxpr_constants(jaxpr)

  # The first dimension variable may be the platform index
  num_dim_vars = len(ctx.shape_poly_state.dim_vars)
  dim_var_avals = [core.ShapedArray((), dtypes.canonicalize_dtype(np.int64))] * num_dim_vars
  dim_var_types = map(aval_to_ir_type, dim_var_avals)

  nr_args = num_const_args + len(jaxpr.in_avals)

  assert nr_args == len(in_avals), (nr_args, in_avals)
  assert replicated_args is None or nr_args == len(replicated_args), \
    (nr_args, replicated_args)
  assert arg_shardings is None or nr_args == len(arg_shardings), \
    (nr_args, arg_shardings)
  assert arg_layouts is None or nr_args == len(arg_layouts), \
    (nr_args, arg_layouts)
  assert arg_memory_kinds is None or nr_args == len(arg_memory_kinds), \
    (nr_args, arg_memory_kinds)
  assert arg_names is None or nr_args == len(arg_names), (nr_args, arg_names)

  # Function inputs: *dim_var_values, *tokens, *const_args, *actual_inputs
  input_types = map(aval_to_ir_type, in_avals)
  output_types = map(aval_to_ir_type, jaxpr.out_avals)
  num_tokens = len(effects)

  token_types = [token_type() for _ in effects]
  token_avals = [core.abstract_token] * num_tokens
  # Order of arguments: dim vars, tokens, const_args, array inputs
  input_avals = dim_var_avals + token_avals + list(in_avals)  # type: ignore
  input_types = [*dim_var_types, *token_types, *input_types]
  output_avals = [core.abstract_token] * num_tokens + jaxpr.out_avals
  output_types = [*token_types, *output_types]

  if input_output_aliases is not None:
    prefix_input_output_aliases = [None] * (num_dim_vars + num_tokens)
    input_output_aliases = [*prefix_input_output_aliases, *input_output_aliases]
    # Update the existing aliases to account for the new output values
    input_output_aliases = [None if a is None
                            else a + num_tokens
                            for a in input_output_aliases]

  if arg_shardings is not None:
    prefix_shardings = [None] * (num_dim_vars + num_tokens)
    arg_shardings = [*prefix_shardings, *arg_shardings]
  if result_shardings is not None:
    token_shardings = [None] * num_tokens
    result_shardings = [*token_shardings, *result_shardings]
  if replicated_args is not None:
    prefix_replicated_args = [False] * (num_dim_vars + num_tokens)
    replicated_args = [*prefix_replicated_args, *replicated_args]
  if arg_memory_kinds is not None:
    prefix_memory_kinds = [None] * (num_dim_vars + num_tokens)
    arg_memory_kinds = [*prefix_memory_kinds, *arg_memory_kinds]
  if result_memory_kinds is not None:
    token_memory_kinds = [None] * num_tokens
    result_memory_kinds = [*token_memory_kinds, *result_memory_kinds]
  if arg_layouts is not None:
    prefix_layouts = [None] * (num_dim_vars + num_tokens)
    arg_layouts = [*prefix_layouts, *arg_layouts]
  if result_layouts is not None:
    token_layouts = [None] * num_tokens
    result_layouts = [*token_layouts, *result_layouts]
  if xla_donated_args is not None:
    xla_donated_args = [*([False] * (num_dim_vars + num_tokens)),
                        *xla_donated_args]

  flat_input_types = flatten_ir_types(input_types)
  flat_output_types = flatten_ir_types(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  func_name = "main" if main_function else name
  func_op = func_dialect.FuncOp(func_name, ftype, ip=ctx.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get(
      "public" if main_function else "private")
  ctx.symbol_table.insert(func_op)

  ir_arg_shardings = None
  if arg_shardings is not None:
    ir_arg_shardings = util.flatten(
        [[_to_physical_op_sharding(ctx, a, s)] * len_ir_types(types)
         for a, s, types in zip(input_avals, arg_shardings, input_types)])

  ir_arg_memory_kinds = None
  if arg_memory_kinds is not None:
    ir_arg_memory_kinds = util.flatten(
        [[mk] * len_ir_types(types)
         for mk, types in zip(arg_memory_kinds, input_types)])

  ir_arg_layouts = None
  if arg_layouts is not None:
    ir_arg_layouts = util.flatten(
        [[_to_xla_layout(l, a)] * len_ir_types(types)
         for l, a, types in zip(arg_layouts, input_avals, input_types)])

  ir_donated_args = None
  if xla_donated_args is not None:
    ir_donated_args = util.flatten(
        [[is_donated] * len_ir_types(types)
         for is_donated, types in zip(xla_donated_args, input_types)])

  ir_result_shardings = None
  unconstrained_variants = None
  if result_shardings is not None:
    ir_result_shardings = util.flatten(
        [[_to_physical_op_sharding(ctx, a, s)] * len_ir_types(types)
         for a, s, types in zip(output_avals, result_shardings, output_types)])
    unconstrained_variants = util.flatten(
        [[_get_unconstrained_variants(s, a)] * len_ir_types(types)
         for a, s, types in zip(output_avals, result_shardings, output_types)])

  ir_result_memory_kinds = None
  custom_call_ir_result_memory_kinds = None
  if result_memory_kinds is not None:
    if propagated_out_mem_kinds is None:
      propagated_out_mem_kinds = (None,) * len(result_memory_kinds)
    res, custom_call_res = [], []
    for pom, mk, types in zip(propagated_out_mem_kinds, result_memory_kinds,
                              output_types):
      if pom is not None and mk is None:
        res.append([pom] * len_ir_types(types))
      else:
        res.append([mk] * len_ir_types(types))  # type: ignore
      # To add the custom call on the output to signal a transfer, only do it
      # if memory kind comes from out_shardings on `jit` and result_memory_kinds
      # comes from out_shardings on `jit`.
      custom_call_res.append([mk] * len_ir_types(types))
    ir_result_memory_kinds = util.flatten(res)
    custom_call_ir_result_memory_kinds = util.flatten(custom_call_res)

  ir_result_layouts = None
  if result_layouts is not None:
    ir_result_layouts = util.flatten(
        [[_to_xla_layout(l, a)] * len_ir_types(types)
         for l, a, types in zip(result_layouts, output_avals, output_types)])

  # Populate arg_attrs
  if (
      replicated_args is not None
      or ir_arg_shardings is not None
      or ir_arg_memory_kinds is not None
      or ir_arg_layouts is not None
      or input_output_aliases is not None
      or ir_donated_args is not None
      or arg_names is not None
      or num_tokens > 0
      or num_dim_vars > 0
      or num_const_args > 0
  ):
    arg_attrs: list[dict[str, ir.Attribute]] = [
        {} for _ in range(len(flat_input_types))]

    if replicated_args is not None:
      replicated_ir_args = [[replicated] * len_ir_types(types) for replicated, types
                            in zip(replicated_args, input_types)]
      for attrs, replicated in zip(arg_attrs, util.flatten(replicated_ir_args)):
        if replicated:
          attrs["mhlo.is_same_data_across_replicas"] = ir.BoolAttr.get(True)

    if use_sharding_annotations and ir_arg_shardings is not None:
      for attrs, sharding in zip(arg_attrs, ir_arg_shardings):
        if sharding is not None:
          if config.use_shardy_partitioner.value:
            attrs["sdy.sharding"] = get_sharding_attr(sharding)
          else:
            attrs["mhlo.sharding"] = get_sharding_attr(sharding)

    if ir_arg_memory_kinds is not None:
      for attrs, memory_kind in zip(arg_attrs, ir_arg_memory_kinds):
        if memory_kind is not None:
          attrs["mhlo.memory_kind"] = ir.StringAttr.get(memory_kind)

    if ir_arg_layouts is not None:
      for attrs, layout in zip(arg_attrs, ir_arg_layouts):
        if layout is not None:
          attrs["mhlo.layout_mode"] = ir.StringAttr.get(layout)

    if ir_donated_args is not None:
      for attrs, is_donated in zip(arg_attrs, ir_donated_args):
        if is_donated:
          attrs["jax.buffer_donor"] = ir.BoolAttr.get(True)

    if input_output_aliases is not None:
      output_ids = util.unflatten(
        list(range(len(flat_output_types))), map(len_ir_types, output_types))
      aliases: list[int | None] = []
      for itypes, alias in zip(input_types, input_output_aliases):
        if alias is None:
          aliases.extend([None] * len_ir_types(itypes))
        else:
          aliases.extend(output_ids[alias])
      for attrs, alias in zip(arg_attrs, aliases):
        if alias is not None:
          attrs["tf.aliasing_output"] = i32_attr(alias)

    if num_dim_vars > 0:
      for var_name, attrs in zip(ctx.shape_poly_state.dim_vars,
                                 arg_attrs[:num_dim_vars]):
        attrs["jax.global_constant"] = ir.StringAttr.get(var_name)
    elif ctx.lowering_parameters.global_constant_computation:
      for attrs in arg_attrs:
        attrs["jax.global_constant"] = ir.StringAttr.get("")

    if num_tokens > 0:
      token_arg_attrs = arg_attrs[num_dim_vars:num_dim_vars + num_tokens]
      for attrs in token_arg_attrs:
        attrs["jax.token"] = ir.BoolAttr.get(True)

    if num_const_args > 0:
      const_arg_attrs = arg_attrs[num_dim_vars + num_tokens :
                                  num_dim_vars + num_tokens + num_const_args]
      for attrs in const_arg_attrs:
        attrs["jax.const"] = ir.BoolAttr.get(True)

    func_op.arg_attrs = ir.ArrayAttr.get(
        [ir.DictAttr.get(attrs) for attrs in arg_attrs])
    # End populate arg_attrs

  result_attrs: list[dict[str, ir.Attribute]] = [
      {} for _ in range(len(flat_output_types))]

  if num_tokens > 0:
    token_result_attrs = result_attrs[:num_tokens]
    for attrs in token_result_attrs:
      attrs["jax.token"] = ir.BoolAttr.get(True)

  if result_names:
    named_result_attrs = result_attrs[num_tokens:]
    if len(named_result_attrs) == len(result_names):
      for attrs, name_ in zip(named_result_attrs, result_names):
        attrs['jax.result_info'] = ir.StringAttr.get(name_)

  if use_sharding_annotations and ir_result_shardings is not None:
    for attrs, sharding, uv in zip(result_attrs, ir_result_shardings,
                                   unconstrained_variants):  # type: ignore
      if sharding is not None and not uv.contains_unconstrained:
        if config.use_shardy_partitioner.value:
          attrs["sdy.sharding"] = get_sharding_attr(sharding)
        else:
          attrs["mhlo.sharding"] = get_sharding_attr(sharding)

  if ir_result_memory_kinds is not None:
    for attrs, mem_kind in zip(result_attrs, ir_result_memory_kinds):
      if mem_kind is not None:
        attrs['mhlo.memory_kind'] = ir.StringAttr.get(mem_kind)

  if ir_result_layouts is not None:
    for attrs, layout in zip(result_attrs, ir_result_layouts):
      if layout is not None:
        attrs['mhlo.layout_mode'] = ir.StringAttr.get(layout)

  func_op.result_attrs = ir.ArrayAttr.get(
      [ir.DictAttr.get(attrs) for attrs in result_attrs])

  if arg_names:
    arg_locs = [ir.Location.unknown()] * (num_dim_vars + num_tokens)
    for n in arg_names:
      arg_locs.append(ir.Location.name(n) if n else ir.Location.unknown())
    entry_block = func_op.add_entry_block(arg_locs)
  else:
    with ir.Location.unknown():
      entry_block = func_op.add_entry_block()

  # When lowering a function out of line, we do not include name context from
  # the caller. A function might have multiple callers, and it would be
  # incorrect to include any one caller's context. Exception: The main function
  # has no caller, so we include its name in the name stack.
  name_stack = (
      source_info_util.new_name_stack(name)
      if main_function
      else source_info_util.new_name_stack()
  )

  with ir.InsertionPoint(entry_block):
    flat_args = entry_block.arguments
    dim_var_values, _, const_arg_values, _ = util.split_list(
        flat_args, [num_dim_vars, num_tokens, num_const_args])
    const_args = core.jaxpr_const_args(jaxpr.jaxpr)
    if num_const_args == 0:
      # If we did not hoist the constants out of this function, lower them now
      const_arg_values = [ir_constant(c, canonicalize_dtype=True)
                          for c in  const_args]
    const_lowering = {id(c): c_arg
                      for c, c_arg in zip(const_args, const_arg_values)}

    # A lowering context just for function body entry/exit code.
    entry_lowering_ctx = LoweringRuleContext(
        module_context=ctx, name_stack=name_stack, traceback=None, primitive=None,
        avals_in=[], avals_out=None,
        tokens_in=TokenSet.create([]), tokens_out=None,
        axis_size_env=None, dim_var_values=dim_var_values,
        const_lowering=const_lowering)
    if not use_sharding_annotations and ir_arg_shardings is not None:
      flat_args = [
          a if s is None else wrap_with_sharding_op(entry_lowering_ctx, a, a_aval, s)
          for a, s, a_aval in zip(flat_args, ir_arg_shardings, input_avals)]

    if ir_arg_shardings is not None and main_function:
      flat_args = [
          replicate_trailing_dims(entry_lowering_ctx, o, a)
          if (a is not core.abstract_token and
              dtypes.issubdtype(a.dtype, dtypes.extended) and
              (s is None or all_unconstrained(rs, a))) else o  # pytype: disable=attribute-error
          for o, s, a, rs in zip(flat_args, ir_arg_shardings, input_avals,
                                 arg_shardings)  # type: ignore
      ]

    _, token_args, _, unflattened_args = util.split_list(
        unflatten_ir_values_like_types(flat_args, input_types),
        [num_dim_vars, num_tokens, num_const_args])
    tokens_in = TokenSet(zip(effects, token_args))
    args: list[IrValues] = unflattened_args
    unique_consts = {id(c): ir_constant(xla.canonicalize_dtype(c))
                     for c in jaxpr.consts}
    consts_for_constvars = [unique_consts[id(c)] for c in jaxpr.consts]

    out_vals, tokens_out = jaxpr_subcomp(
        ctx, jaxpr.jaxpr, name_stack, tokens_in,
        consts_for_constvars, *args, dim_var_values=dim_var_values,
        const_lowering=const_lowering)
    outs: list[IrValues] = []
    for eff in effects:
      outs.append(tokens_out.get(eff))
    outs.extend(out_vals)

    flat_outputs = flatten_ir_values(outs)

    if not use_sharding_annotations and ir_result_shardings is not None:
      flat_outputs = [
          o if s is None else wrap_with_sharding_op(entry_lowering_ctx, o, o_aval, s)
          for o, s, o_aval in zip(flat_outputs, ir_result_shardings, output_avals)]

    if ir_result_shardings is not None:
      temp_flat_outputs = []
      for o, s, o_aval, uv in zip(flat_outputs, ir_result_shardings,
                                  output_avals, unconstrained_variants):  # type: ignore
        if (s is not None and uv.contains_unconstrained and
            not uv.all_unconstrained):
          if config.use_shardy_partitioner.value:
            s = modify_sdy_sharding_wrt_axis_types(s, o_aval.sharding.mesh)
            unconstrained_dims = None  # delete this after shardy is default
          else:
            unconstrained_dims = (
                set(range(o_aval.ndim)) if o_aval.sharding.mesh._any_axis_auto
                else None)
          temp_flat_outputs.append(wrap_with_sharding_op(
              entry_lowering_ctx, o, o_aval, s,
              unspecified_dims=unconstrained_dims))
        else:
          temp_flat_outputs.append(o)
      flat_outputs = temp_flat_outputs

    # Insert a custom call if output is on host because XLA needs that to do the
    # transfer.
    if custom_call_ir_result_memory_kinds is not None and main_function:
      flat_outputs = [
          o if mk is None else wrap_with_memory_kind(o, mk, o_aval)
          for o, mk, o_aval in zip(
              flat_outputs, custom_call_ir_result_memory_kinds, output_avals)]

    if ir_result_shardings is not None and main_function:
      flat_outputs = [
          replicate_trailing_dims(entry_lowering_ctx, o, a)
          if (a is not core.abstract_token and
              dtypes.issubdtype(a.dtype, dtypes.extended) and
              (s is None or all_unconstrained(rs, a))) else o  # pytype: disable=attribute-error
          for o, s, a, rs in zip(flat_outputs, ir_result_shardings, output_avals,
                                 result_shardings)  # type: ignore
      ]

    func_dialect.return_(flat_outputs)

  return func_op


def wrap_with_memory_kind(
    x: ir.Value, memory_kind: str, aval_out: core.AbstractValue) -> ir.Value:
  if aval_out is None:
    result_type = x.type
  else:
    typ = aval_to_ir_type(aval_out)
    assert isinstance(typ, ir.Type), typ
    result_type = typ
  op = custom_call("annotate_device_placement", result_types=[result_type],
                   operands=[x], has_side_effect=True, api_version=1)
  dict_attr = {"_xla_buffer_placement": ir.StringAttr.get(memory_kind)}
  op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(dict_attr)
  return op.result


def replicate_trailing_dims(ctx, val: ir.Value, aval) -> ir.Value:
  # Set the sharding of extended dtypes to be UNCONSTRAINED
  # (i.e. XLA will choose) on aval.shape.
  # For the trailing dims i.e. the dimension of key_shape on the base_array,
  # the sharding is set to be REPLICATED always.
  # For example: if the key.shape is (8, 2) and key_data(key).shape is (8, 2, 2),
  # then the sharding will be P(P.UNCONSTRAINED, P.UNCONSTRAINED, None).
  # The below custom call achieves the sharding like above example.
  assert isinstance(aval, (core.ShapedArray, core.DShapedArray))
  if config.use_shardy_partitioner.value:
    physical_ndim = core.physical_aval(aval).ndim
    s = SdyArray(
        mesh_shape=None,
        dim_shardings=[
            sharding_impls.SdyDim(axes=[], is_open=i < aval.ndim)
            for i in range(physical_ndim)
        ])
    return wrap_with_sharding_op(ctx, val, aval, s)
  else:
    return wrap_with_sharding_op(
      ctx, val, aval, xc.HloSharding.replicate().to_proto(),
      unspecified_dims=set(range(aval.ndim)))

class HashableLiteral:
  """Hashable wrapper of core.Literal, used for deduplicating IR constants."""

  __slots__ = ["value", "data"]

  value: core.Literal

  # Copy of the value suitable for an equality comparison. We are careful to
  # avoid floating point comparisons here, because in particular we don't want
  # 0.0 and -0.0 to be considered equal, but we are fine with NaNs being equal.
  data: bytes | int | bool | None

  def __init__(self, value):
    self.value = value
    if isinstance(value.val, (np.generic, np.ndarray)):
      self.data = value.val.tobytes()
    elif isinstance(value.val, (bool, int)):
      self.data = value.val
    elif isinstance(value.val, float):
      self.data = np.float64(value.val).tobytes()
    elif isinstance(value.val, complex):
      self.data = np.complex128(value.val).tobytes()
    else:
      self.data = None  # Unhandled case.

  def __hash__(self):
    return hash(self.data)

  def __eq__(self, other):
    if type(self.value.val) != type(other.value.val):
      return False
    if self.value.aval != other.value.aval:
      return False
    if self.data is None:
      return self is other
    return self.data == other.data

_uncacheable_primitives: set[core.Primitive] = set()

def jaxpr_subcomp(ctx: ModuleContext, jaxpr: core.Jaxpr,
                  name_stack: source_info_util.NameStack,
                  tokens: TokenSet,
                  consts_for_constvars: Sequence[IrValues],
                  *args: IrValues,
                  dim_var_values: Sequence[ir.Value],
                  const_lowering: dict[int, IrValues],
                  ) -> tuple[Sequence[IrValues], TokenSet]:
  """Lowers a jaxpr into MLIR, inlined into an existing function.

  Assumes that an MLIR context, location, and insertion point are set.

  consts_for_constvars: the constants corresponding to jaxpr.constvars.
  dim_var_values: the list of dimension variables values in the current
    IR function, in the order of ctx.shape_poly_state.dim_vars.
  const_lowering: the lowering for constants, by constant id.
    See https://docs.jax.dev/en/latest/internals/constants.html
  """
  assert "gpu" not in ctx.platforms
  cached_ir_consts: dict[HashableLiteral, IrValues] = {}

  def read(v: core.Atom) -> IrValues:
    if type(v) is core.Literal:
      h = HashableLiteral(v)
      c = cached_ir_consts.get(h)
      if c is None:
        c = ir_constant(v.val, const_lowering, canonicalize_dtype=True)
        # TODO(necula): do we really need the cache, now that we hoist?
        cached_ir_consts[h] = c
      return c
    else:
      assert isinstance(v, core.Var)
      return env[v]

  def aval(v: core.Atom) -> core.AbstractValue:
    if type(v) is core.Literal:
      return core.abstractify(v.val)
    else:
      return v.aval

  def write(v: core.Var, node: IrValues):
    assert node is not None
    w: IrValues
    if isinstance(node, ir.Value):
      w = node
    else:
      if len(node) == 1:
        warnings.warn(
            "JAX lowering rules should not wrap singleton values in tuples. "
            "It will be an error to wrap a singleton value in a tuple in a "
            "future version of JAX.",
            DeprecationWarning, stacklevel=2)
        w = node[0]
      else:
        w = tuple(node)
    env[v] = w

  env: dict[core.Var, IrValues] = {}

  assert all(_is_ir_values(v) for v in args), args
  assert all(_is_ir_values(v) for v in consts_for_constvars), \
    consts_for_constvars
  assert isinstance(name_stack, source_info_util.NameStack), type(name_stack)
  assert len(args) == len(jaxpr.invars), (jaxpr, args)
  assert len(consts_for_constvars) == len(jaxpr.constvars), \
    (jaxpr, consts_for_constvars)
  assert len(ctx.shape_poly_state.dim_vars) == len(dim_var_values), \
    (ctx.shape_poly_state.dim_vars, dim_var_values)
  foreach(write, jaxpr.constvars, consts_for_constvars)
  foreach(write, jaxpr.invars, args)
  last_used = core.last_used(jaxpr)
  for eqn in jaxpr.eqns:
    in_nodes = tuple(map(read, eqn.invars))
    assert all(_is_ir_values(v) for v in in_nodes), (eqn, in_nodes)

    avals_in = tuple(map(aval, eqn.invars))
    ordered_effects = list(effects_lib.ordered_effects.filter_in(eqn.effects))
    tokens_in = tokens.subset(ordered_effects)

    eqn_name_stack = name_stack + eqn.source_info.name_stack
    loc = source_info_to_location(ctx, eqn.primitive, eqn_name_stack,
                                  eqn.source_info.traceback)
    with (source_info_util.user_context(eqn.source_info.traceback), loc,
          eqn.ctx.manager):
      # TODO(mattjj, phawkins): support caching for dynamic shapes.
      can_cache_lowering = (
          eqn.primitive not in _uncacheable_primitives and
          not config.dynamic_shapes.value)
      if can_cache_lowering:
        loc = source_info_to_location(ctx, None, eqn_name_stack,
                                      eqn.source_info.traceback)
        with loc:
          out_nodes, tokens_out = _cached_lowering(
              ctx, eqn, tokens_in, tuple(dim_var_values), const_lowering,
              *in_nodes, **eqn.params)
      else:
        # If we cannot cache the lowering, lower inline.
        axis_size_env = None
        if config.dynamic_shapes.value:
          axis_size_env = {d: read(d)
                           for a in avals_in if type(a) is core.DShapedArray
                           for d in a.shape if type(d) is core.Var}
        rule_ctx = LoweringRuleContext(
            module_context=ctx, primitive=eqn.primitive,
            name_stack=eqn_name_stack,
            traceback=eqn.source_info.traceback,
            avals_in=avals_in,
            avals_out=map(aval, eqn.outvars), tokens_in=tokens_in,
            tokens_out=None, jaxpr_eqn_ctx=eqn.ctx,
            dim_var_values=dim_var_values,
            axis_size_env=axis_size_env,
            const_lowering=const_lowering)
        out_nodes, _inline = _uncached_lowering(
            eqn.primitive, eqn.ctx, eqn.effects, rule_ctx, *in_nodes,
            **eqn.params)
        tokens_out = rule_ctx.tokens_out

      assert len(out_nodes) == len(eqn.outvars), (out_nodes, eqn)
      if ordered_effects:
        tokens = tokens.update_tokens(tokens_out)

    foreach(write, eqn.outvars, out_nodes)
    core.clean_up_dead_vars(eqn, env, last_used)
  return tuple(read(v) for v in jaxpr.outvars), tokens

def _cached_lowering(ctx: ModuleContext, eqn: core.JaxprEqn,
                     tokens_in: TokenSet,
                     dim_var_values: tuple[ir.Value, ...],
                     const_lowering: dict[int, IrValues],
                     *args, **params) -> tuple[Sequence[IrValues], TokenSet]:
  """Lowers a jaxpr equation, using a cache.

  The jaxpr equation's lowering is emitted as an out-of-line MLIR function, and
  that function's construction is cached in the event that we see a similar
  equation. For each such equation we either inline the function body or emit
  an out-of-line call to it, depending on whether any of the lowering rules
  opted out of inlining."""
  def aval(v: core.Atom) -> core.AbstractValue:
    if type(v) is core.Literal:
      return core.abstractify(v.val)
    else:
      return v.aval

  avals_in = tuple(map(aval, eqn.invars))
  ordered_effects = list(effects_lib.ordered_effects.filter_in(eqn.effects))
  cache_key = LoweringCacheKey(
      primitive=eqn.primitive,
      eqn_ctx=eqn.ctx,
      avals_in=avals_in,
      effects=frozenset(eqn.effects),
      params=frozen_dict.FrozenDict(eqn.params),
      platforms=tuple(ctx.platforms),
  )
  try:
    cache_entry = ctx.lowering_cache.get(cache_key, None)
  except TypeError:
    print("Unable to hash key: ", eqn)
    raise
  if cache_entry is None:
    avals_out = map(lambda v: v.aval, eqn.outvars)
    cache_entry = _emit_lowering_rule_as_fun(
        partial(_uncached_lowering, eqn.primitive, eqn.ctx, eqn.effects), ctx,
        eqn.ctx, eqn.primitive, ordered_effects, avals_in,
        avals_out, **params)
    ctx.lowering_cache[cache_key] = cache_entry

  tokens_in_args = tuple(tokens_in.get(eff) for eff in ordered_effects)
  const_arg_values = tuple(ir_constant(c, const_lowering,
                                       canonicalize_dtype=True)
                           for c in cache_entry.const_args)
  args = flatten_ir_values(
      dim_var_values + tokens_in_args + const_arg_values + args)
  if cache_entry.inline:
    outs = _jax_mlir_ext.inlined_func_call(
        cache_entry.func, args, ir.InsertionPoint.current.block)
  else:
    outs = func_dialect.CallOp(
        flatten_ir_types(cache_entry.output_types),
        ir.FlatSymbolRefAttr.get(cache_entry.func.sym_name.value),
        args
    ).results
  out_nodes = unflatten_ir_values_like_types(outs, cache_entry.output_types)
  token_outs, out_nodes = util.split_list(out_nodes, [len(ordered_effects)])
  return out_nodes, TokenSet(zip(ordered_effects, token_outs))


def _emit_lowering_rule_as_fun(
    lowering_rule: LoweringRule,
    ctx: ModuleContext,
    eqn_ctx: core.JaxprEqnContext,
    primitive: core.Primitive,
    ordered_effects: Sequence[core.Effect],
    avals_in: Sequence[core.AbstractValue],
    avals_out: Sequence[core.AbstractValue],
    **params
) -> LoweringCacheValue:
  """Emits the contents of a lowering rule as a private function."""
  num_dim_vars = len(ctx.shape_poly_state.dim_vars)
  # TODO(necula) maybe only pass the dim_vars if they are needed?
  dim_var_types = [
    aval_to_ir_type(core.ShapedArray((), dtypes.canonicalize_dtype(np.int64)))
  ] * num_dim_vars

  const_args =  core.eqn_params_const_args(params)
  const_arg_avals = tuple(core.shaped_abstractify(c) for c in const_args)

  input_types = map(aval_to_ir_type, const_arg_avals + avals_in)  # type: ignore
  output_types = map(aval_to_ir_type, avals_out)
  token_types = [token_type() for _ in ordered_effects]
  input_types = [*dim_var_types, *token_types, *input_types]
  output_types = [*token_types, *output_types]

  flat_input_types = flatten_ir_types(input_types)
  flat_output_types = flatten_ir_types(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  func_op = func_dialect.FuncOp(primitive.name, ftype,
                                ip=ctx.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
  ctx.symbol_table.insert(func_op).value
  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    unflattened_args = unflatten_ir_values_like_types(
      entry_block.arguments, input_types)
    dim_var_values, token_args, const_arg_values, unflattened_args = \
      util.split_list(unflattened_args,
                      [num_dim_vars, len(ordered_effects), len(const_args)])
    const_lowering = {id(c): c_arg
                      for c, c_arg in zip(const_args, const_arg_values)}
    sub_ctx = LoweringRuleContext(
        module_context=ctx, primitive=primitive,
        name_stack=source_info_util.new_name_stack(),
        traceback=None,
        avals_in=avals_in, avals_out=avals_out,
        tokens_in=TokenSet(zip(ordered_effects, token_args)),
        tokens_out=None, jaxpr_eqn_ctx=eqn_ctx, dim_var_values=dim_var_values,
        const_lowering=const_lowering)
    with ir.Location.name(str(primitive.name)):
      outs, inline = lowering_rule(sub_ctx, *unflattened_args, **params)
    if sub_ctx.tokens_out:
      outs = [*[sub_ctx.tokens_out.get(eff) for eff in ordered_effects], *outs]
    outs = flatten_ir_values(outs)
    func_dialect.return_(outs)
  return LoweringCacheValue(func_op, output_types, const_args, inline)


def _get_override_lowering_rule(
    ctx: ModuleContext, primitive: core.Primitive
) -> LoweringRule | None:
  if ctx.lowering_parameters.override_lowering_rules is None:
    return None
  for p, rule in ctx.lowering_parameters.override_lowering_rules:
    if primitive is p:
      return rule
  return None

def _uncached_lowering(
    primitive: core.Primitive,
    eqn_ctx: core.JaxprEqnContext,
    effects: effects_lib.Effects,
    ctx: LoweringRuleContext,
    *args,
    **params,
):
  inline = True  # Should calls to this lowering rule be inlined?
  override_rule = _get_override_lowering_rule(ctx.module_context, primitive)
  platform_rules: dict[str, LoweringRule] = {}
  default_rule: LoweringRule | None = None
  # See mlir.lower_per_platform for meaning of `platform_rules` and `default_rule`
  if override_rule is not None:
    default_rule = override_rule
    assert not isinstance(default_rule, LoweringRuleEntry)
  else:
    # First the platform-specific rules
    for p in _platforms_for_eqn_ctx(eqn_ctx) or ctx.module_context.platforms:
      if primitive in _platform_specific_lowerings[p]:
        r = _platform_specific_lowerings[p][primitive]
        platform_rules[p] = r.rule
        inline = inline and r.inline
    # Now the default rule
    if primitive in _lowerings:
      r = _lowerings[primitive]
      default_rule = r.rule
      assert not isinstance(default_rule, LoweringRuleEntry)
      inline = inline and r.inline

  assert not isinstance(default_rule, LoweringRuleEntry)
  assert not any(isinstance(r, LoweringRuleEntry) for r in platform_rules.values())
  ans = lower_per_platform(ctx, str(primitive), platform_rules, default_rule,
                           effects, *args, **params)
  try:
    rets = tuple(ans)
  except TypeError as e:
    raise ValueError("Output of translation rule must be iterable: "
                      f"{primitive}, got output {ans}") from e

  if ctx.tokens_in.effects():
    # If there were ordered effects in the primitive, there should be output
    # tokens we need for subsequent ordered effects.
    tokens_out = ctx.tokens_out
    if tokens_out is None:
      raise ValueError(
          f'Lowering rule for `{primitive}` needs to set `tokens_out` '
          f'because it has effects: {effects}.')
    if tokens_out.effects() != ctx.tokens_in.effects():
      raise ValueError(
          f"Lowering rule for `{primitive}` returns incorrect set of output"
          f" tokens. Expected: {tuple(ctx.tokens_in.effects())} vs. Actual:"
          f" {tuple(tokens_out.effects())}"
      )
  return rets, inline


def _platforms_for_eqn_ctx(eqn_ctx: core.JaxprEqnContext | None
                           ) -> tuple[str, ...]:
  """Returns platforms to override based on compute type of jaxpr equation."""
  if eqn_ctx is None:
    return ()
  if eqn_ctx.compute_type == 'device_host':
    return ('cpu',)
  if eqn_ctx.compute_type == 'tpu_sparsecore':
    return ('tpu',)
  return ()

def _platforms_for_eqn(ctx: LoweringRuleContext) -> tuple[str, ...]:
  """The lowering platforms for the current eqn"""
  return tuple(_platforms_for_eqn_ctx(ctx.jaxpr_eqn_ctx) or
               ctx.platforms or ctx.module_context.platforms)


def lower_per_platform(ctx: LoweringRuleContext,
                       description: str,
                       platform_rules: dict[str, LoweringRule],
                       default_rule: LoweringRule | None,
                       effects: effects_lib.Effects,
                       *rule_args: ir.Value | tuple[ir.Value, ...],
                       **rule_kwargs) -> Sequence[ir.Value]:
  """Emits code for a primitive for the current lowering platform(s).

  For example, given
      platform_rules = dict(tpu=rule0, cpu=rule0)
      default_rule = rule1

  and
      ctx.module_context.lowering_parameters.platforms = ("cpu",)

  emits:
      rule0(ctx, *rule_args, **rule_kwargs)

  In case of multi-platform lowering, e.g., if
      ctx.module_context.lowering_parameters.platforms = ("cpu", "cuda", "tpu")

  emits:
    rule_idx = case current_platform_idx:
                   0: return 0  # cpu rule index
                   1: return 1  # cuda rule index
                   2: return 0  # tpu rule index
    output = case rule_idx
               0: return rule0(*rule_args, **rule_kwargs)
               1: return rule1(*rule_args, **rule_kwargs)

  Args:
   ctx: lowering context.
   description: a string to include in error messages.
   platform_rules: map platform names, e.g., "cpu", "cuda", to
     `LoweringRule`s, for the platforms that have non-default lowering.
   default_rule: an optional rule to use for platforms not in `platform_rules`.
   effects: the set of effects for the current primitive.
   rule_args: the args of the lowering rules.
   rule_kwargs: the kwargs of the lowering rules.
  """
  platforms: Sequence[str] = _platforms_for_eqn(ctx)
  # Special case the common case (single-platform lowering)
  if len(platforms) == 1:
    rule = platform_rules.get(platforms[0], default_rule)
    if rule is None:
      raise NotImplementedError(
        f"MLIR translation rule for primitive '{description}' not "
        f"found for platform {platforms[0]}")

  # Multi-platform lowering
  kept_rules: list[LoweringRule] = []  # Only the rules for the platforms of interest
  platform_to_kept_rules_idx: dict[str, int] = {}
  for p, prule in platform_rules.items():
    if p not in platforms:
      continue
    platform_to_kept_rules_idx[p] = len(kept_rules)
    kept_rules.append(prule)

  platforms_without_specific_rule = [p for p in platforms
                                     if p not in platform_to_kept_rules_idx]
  if platforms_without_specific_rule:
    if default_rule is None:
      raise NotImplementedError(
        f"MLIR translation rule for primitive '{description}' not "
        f"found for platforms {platforms_without_specific_rule}")
    for p in platforms_without_specific_rule:
      platform_to_kept_rules_idx[p] = len(kept_rules)
    kept_rules.append(default_rule)

  assert kept_rules
  # If there is a single rule left just apply the rule, without conditionals.
  if len(kept_rules) == 1:
    output = kept_rules[0](ctx, *rule_args, **rule_kwargs)
    foreach(
        lambda o: wrap_compute_type_in_place(ctx, o.owner),
        filter(_is_not_block_argument, flatten_ir_values(output)),
    )
    foreach(
        lambda o: wrap_xla_metadata_in_place(ctx, o.owner),
        flatten_ir_values(output),
    )
    return output

  assert len(platforms) > 1 and len(kept_rules) >= 2, (platforms, kept_rules)
  assert len(ctx.dim_var_values) >= 1, "Must have a platform_index variable"

  # The first dim_var_values is the platform index
  current_platform_idx = ctx.dim_var_values[0]
  # Compute the rule index based on the current platform
  i32_type = aval_to_ir_type(core.ShapedArray((), dtype=np.int32))
  if current_platform_idx.type != i32_type:
    current_platform_idx = hlo.convert(i32_type, current_platform_idx)
  rule_idx_op = hlo.CaseOp([i32_type],
                           index=current_platform_idx,
                           num_branches=len(platforms))
  for i, p in enumerate(platforms):
    branch = rule_idx_op.regions[i].blocks.append()
    with ir.InsertionPoint(branch):
      hlo.return_([ir_constant(np.int32(platform_to_kept_rules_idx[p]))])
  ordered_effects = effects_lib.ordered_effects.filter_in(effects)
  rule_out_avals = [core.abstract_token] * len(ordered_effects) + ctx.avals_out
  output_types = map(aval_to_ir_type, rule_out_avals)
  case_op = hlo.CaseOp(flatten_ir_types(output_types),
                      index=rule_idx_op,
                      num_branches=len(kept_rules))
  for i, rule in enumerate(kept_rules):
    platforms_for_this_rule = [p
                               for p, rule_idx in platform_to_kept_rules_idx.items()
                               if rule_idx == i]
    inner_ctx = ctx.replace(platforms=platforms_for_this_rule)
    branch = case_op.regions[i].blocks.append()
    with ir.InsertionPoint(branch):
      output = rule(inner_ctx, *rule_args, **rule_kwargs)
      try:
        out_nodes = flatten_ir_values(output)
      except TypeError as e:
        raise ValueError("Output of translation rule must be iterable: "
                        f"{description}, got output {output}") from e
      foreach(
          lambda o: wrap_compute_type_in_place(ctx, o.owner),
          filter(_is_not_block_argument, out_nodes),
      )
      foreach(
          lambda o: wrap_xla_metadata_in_place(ctx, o.owner),
          out_nodes,
      )
      if inner_ctx.tokens_out is not None:
        assert len(ordered_effects) == len(inner_ctx.tokens_out)
        out_nodes = [inner_ctx.tokens_out.get(eff)
                     for eff in ordered_effects] + out_nodes
      hlo.return_(out_nodes)

  results = case_op.results
  if ordered_effects:
    tokens, results = util.split_list(
      unflatten_ir_values_like_types(results, output_types),
      [len(ordered_effects)])
    tokens_out = ctx.tokens_in.update_tokens(TokenSet(zip(ordered_effects,
                                                          tokens)))
    ctx.set_tokens_out(tokens_out)
  return results

def _ir_consts(consts) -> list[IrValues]:
  uniq_consts = {id(c): ir_constant(c, canonicalize_dtype=True) for c in consts}
  return [uniq_consts[id(c)] for c in consts]

def lower_fun(fun: Callable, multiple_results: bool = True) -> Callable:
  """Converts a traceable JAX function `fun` into a lowering rule.

  The returned function does not use `avals_out`, so callers may pass any value
  as `avals_out`."""
  def f_lowered(ctx: LoweringRuleContext, *args, **params):
    f = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params,
        debug_info=api_util.debug_info("lower_fun", fun, args, params))
    manager = (contextlib.nullcontext() if ctx.jaxpr_eqn_ctx is None else
               ctx.jaxpr_eqn_ctx.manager)

    with manager:
      if config.dynamic_shapes.value:
        # We might be applying this function to arguments with dynamic shapes,
        # i.e. there might be Vars in the shape tuples of ctx.avals_in. In that
        # case, we need to form a jaxpr with leading binders for those axis size
        # arguments (by computing an InputType and using trace_to_jaxpr_dynamic2),
        # and we need to call jaxpr_subcomp with these arguments made explicit.
        assert ctx.axis_size_env is not None
        args = (*ctx.axis_size_env.values(), *args)
        idx = {d: core.DBIdx(i) for i, d in enumerate(ctx.axis_size_env)}
        i32_aval = core.ShapedArray((), np.dtype('int32'))
        implicit_args = [(i32_aval, False)] * len(ctx.axis_size_env)
        explicit_args = [(a.update(shape=tuple(idx.get(d, d) for d in a.shape))  # type: ignore
                          if type(a) is core.DShapedArray else a, True)
                        for a in ctx.avals_in]
        wrapped_fun = lu.annotate(wrapped_fun, (*implicit_args, *explicit_args))
        jaxpr, _, consts_for_constvars = pe.trace_to_jaxpr_dynamic2(wrapped_fun)
      else:
        jaxpr, _, consts_for_constvars = pe.trace_to_jaxpr_dynamic(wrapped_fun,
                                                                   ctx.avals_in)
        # TODO(frostig,mattjj): check ctx.avals_out against jaxpr avals out?

      if ctx.platforms is not None:
        sub_context = ctx.module_context.replace(platforms=ctx.platforms)
      else:
        sub_context = ctx.module_context
      out, tokens = jaxpr_subcomp(
          sub_context, jaxpr, ctx.name_stack, ctx.tokens_in,
          _ir_consts(consts_for_constvars), *args,
          dim_var_values=ctx.dim_var_values,
          const_lowering=ctx.const_lowering)
      ctx.set_tokens_out(tokens)
      return out

  return f_lowered


def _lower_jaxpr_to_fun_cached(ctx: ModuleContext,
                               fn_name, call_jaxpr: core.ClosedJaxpr,
                               num_const_args: int,
                               effects,
                               in_avals,
                               arg_names=None, result_names=None):
  assert num_const_args + len(call_jaxpr.in_avals) == len(in_avals)
  if not call_jaxpr.consts and arg_names is result_names is None:
    # Cacheable.
    key = (fn_name, call_jaxpr.jaxpr, tuple(effects))
    try:
      func_op, _, _ = ctx.cached_primitive_lowerings[key]
    except KeyError:
      num_callbacks = len(ctx.host_callbacks)
      func_op = lower_jaxpr_to_fun(
          ctx, fn_name, call_jaxpr, effects,
          num_const_args=num_const_args,
          in_avals=in_avals,
          arg_names=arg_names,
          result_names=result_names)

      # If this Jaxpr includes callbacks, we can't cache the lowering because
      # on TPU every callback must have a globally unique channel, but the
      # channel gets assigned during lowering.
      has_callbacks = len(ctx.host_callbacks) > num_callbacks
      if not has_callbacks or "tpu" not in ctx.platforms:
        ctx.cached_primitive_lowerings[key] = func_op, func_op.name.value, func_op.type.results
  else:
    func_op = lower_jaxpr_to_fun(
        ctx, fn_name, call_jaxpr, effects,
        num_const_args=num_const_args, in_avals=in_avals,
        arg_names=arg_names, result_names=result_names)
  return func_op


def check_backend_matches(inner_backend: str | None,
                          lowering_platforms: Sequence[str]):
  # For nested calls, the outermost call sets the backend for all inner calls;
  # it's an error if the inner call has a conflicting explicit backend spec.
  if inner_backend is None:
    return
  outer_backend, *more_lowering_platforms = lowering_platforms
  if more_lowering_platforms:
    raise NotImplementedError(
        "Multi-platform lowering when a backend= parameter is specified")
  if (inner_backend != outer_backend and
      outer_backend not in xb.expand_platform_alias(inner_backend)):
    raise ValueError(
        f"Outer-jit backend specification {outer_backend} must match explicit "
        f"inner-jit backend specification {inner_backend}.")


def lower_called_computation(
    fn_name,
    call_jaxpr: core.ClosedJaxpr | core.Jaxpr,
    ctx: ModuleContext,
    num_const_args: int,
    in_avals,
    out_avals,
    tokens_in,
    backend=None,
    arg_names=None,
    result_names=None,
):
  if isinstance(call_jaxpr, core.Jaxpr):
    call_jaxpr = pe.close_jaxpr(call_jaxpr)
  check_backend_matches(backend, ctx.platforms)
  effects = list(tokens_in.effects())
  output_types = map(aval_to_ir_type, out_avals)
  output_types = [token_type()] * len(effects) + output_types
  func_op = _lower_jaxpr_to_fun_cached(
      ctx,
      fn_name,
      call_jaxpr,
      num_const_args,
      effects,
      in_avals=in_avals,
      arg_names=arg_names,
      result_names=result_names,
  )
  return func_op, output_types, effects


def call_lowering(fn_name, call_jaxpr: core.ClosedJaxpr | core.Jaxpr, backend,
                  ctx: ModuleContext, in_avals,
                  out_avals, tokens_in, *args,
                  dim_var_values: Sequence[ir.Value],
                  const_lowering: dict[int, IrValues],
                  arg_names=None, result_names=None):
  # TODO(necula): clean up the types
  if isinstance(call_jaxpr, core.ClosedJaxpr):
    const_args = core.jaxpr_const_args(call_jaxpr.jaxpr)
  else:
    const_args = core.jaxpr_const_args(call_jaxpr)
  const_arg_values = [ir_constant(c, const_lowering, canonicalize_dtype=True)
                      for c in const_args]
  args = tuple(const_arg_values) + args
  if arg_names is not None:
    arg_names = [""] * len(const_args) + arg_names
  in_avals = (*(core.shaped_abstractify(c) for c in const_args), *in_avals)

  func_op, output_types, effects = lower_called_computation(
      fn_name, call_jaxpr, ctx, len(const_args), in_avals, out_avals,
      tokens_in,
      backend=backend, arg_names=arg_names, result_names=result_names)
  symbol_name = func_op.name.value
  flat_output_types = flatten_ir_types(output_types)
  tokens = [tokens_in.get(eff) for eff in effects]
  args = (*dim_var_values, *tokens, *args)
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(symbol_name),
                             flatten_ir_values(args))
  out_nodes = unflatten_ir_values_like_types(call.results, output_types)
  tokens, out_nodes = util.split_list(out_nodes, [len(effects)])
  tokens_out = tokens_in.update_tokens(TokenSet(zip(effects, tokens)))
  return out_nodes, tokens_out

def core_call_lowering(ctx: LoweringRuleContext,
                       *args, name, backend=None,
                       call_jaxpr: core.ClosedJaxpr):
  out_nodes, tokens = call_lowering(
      name, call_jaxpr, backend, ctx.module_context,
      ctx.avals_in, ctx.avals_out, ctx.tokens_in, *args,
      dim_var_values=ctx.dim_var_values,
      const_lowering=ctx.const_lowering)
  ctx.set_tokens_out(tokens)
  return out_nodes

register_lowering(core.call_p, partial(core_call_lowering, name="core_call"))
# TODO(phawkins): Not cacheable because of debug_print on TPU.
register_lowering(core.closed_call_p,
                  partial(core_call_lowering, name="closed_call"),
                  cacheable=False)

def map_compute_type(c_type: str) -> str:
  if c_type == "device_host":
    return "host"
  elif c_type == "device":
    return "dense"
  elif c_type == "tpu_sparsecore":
    return "sparse"
  raise ValueError(f"Invalid compute type {c_type}. Current supported values "
                   "are `device_host`, `device` and `tpu_sparsecore`")

def wrap_compute_type_in_place(ctx: LoweringRuleContext, op: ir.Operation) -> None:
  if ctx.jaxpr_eqn_ctx is not None and ctx.jaxpr_eqn_ctx.compute_type is not None:
    if ctx.jaxpr_eqn_ctx.compute_type.startswith("gpu_stream:"):
      stream = ctx.jaxpr_eqn_ctx.compute_type.split(":")[1]
      dict_attr = {
        "_xla_stream_annotation": ir.StringAttr.get(stream),
        "inlineable": ir.StringAttr.get("false"),
      }
      op.operation.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(dict_attr)
    else:
      dict_attr = {"_xla_compute_type": ir.StringAttr.get(
          map_compute_type(ctx.jaxpr_eqn_ctx.compute_type))}
      op.operation.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(dict_attr)


def wrap_xla_metadata_in_place(ctx: LoweringRuleContext, op: ir.Operation) -> None:
  if ctx.jaxpr_eqn_ctx is not None and ctx.jaxpr_eqn_ctx.xla_metadata:
    ctx_attributes, existing_attributes = {}, {}
    for k, v in ctx.jaxpr_eqn_ctx.xla_metadata.items():
      ctx_attributes[k] = ir.StringAttr.get(str(v).lower())
    if isinstance(op, ir.Operation):
      # combine with existing mhlo.frontend_attributes
      for attr in op.attributes:
        if attr.name == "mhlo.frontend_attributes":
          for a in attr.attr:
            existing_attributes[a.name] = a.attr
      op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
          ctx_attributes | existing_attributes
      )


def broadcast_in_dim(ctx: LoweringRuleContext, op, aval_out: core.AbstractValue, *,
                     broadcast_dimensions) -> ir.Value:
  # broadcast_dimension[i] is the axis of the result where the axis i of
  # op is broadcast.
  # Lower a possibly-dynamic broadcast_in_dim
  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):  # type: ignore
    elt_shape = core.physical_element_aval(aval_out.dtype).shape  # type: ignore
    trailing_dims = [aval_out.ndim + i for i in range(len(elt_shape))]  # type: ignore
    broadcast_dimensions = [*broadcast_dimensions, *trailing_dims]
    physical_aval_out = core.physical_aval(aval_out)
    return broadcast_in_dim(
        ctx, op, physical_aval_out, broadcast_dimensions=broadcast_dimensions)
  else:
    if not core.is_constant_shape(aval_out.shape):  # type: ignore
      shape = eval_dynamic_shape_as_tensor(ctx, aval_out.shape)  # type: ignore
      out = hlo.dynamic_broadcast_in_dim(
          aval_to_ir_type(aval_out), op,
          shape,
          dense_int_array(broadcast_dimensions),
      )
    else:
      assert all(d != ir.ShapedType.get_dynamic_size()
                 for d in aval_out.shape), aval_out  # type: ignore
      out = hlo.broadcast_in_dim(
          aval_to_ir_type(aval_out), op,
          dense_int_array(broadcast_dimensions))
    wrap_compute_type_in_place(ctx, out.owner)
    return out

def multi_broadcast_in_dim(ctx: LoweringRuleContext,
                           ops: Sequence[ir.Value],
                           ops_avals: Sequence[core.AbstractValue],
                           out_shape: core.Shape,
                           out_sharding=None) -> Sequence[ir.Value]:
  """Broadcasts multiple ops to the out_shape."""
  out = []
  for op, op_aval in zip(ops, ops_avals):
    op_aval_shape = op_aval.shape  # type: ignore
    if core.definitely_equal_shape(op_aval_shape, out_shape):
      out.append(op)
    else:
      assert len(op_aval_shape) <= len(out_shape), (op_aval_shape, out_shape)
      broadcast_dimensions = list(range(len(out_shape) - len(op_aval_shape), len(out_shape)))
      out_aval = core.ShapedArray(
          out_shape, op_aval.dtype, sharding=out_sharding)  # type: ignore
      b_out = broadcast_in_dim(
          ctx, op, out_aval, broadcast_dimensions=broadcast_dimensions)
      b_out = lower_with_sharding_in_types(ctx, b_out, out_aval)
      out.append(b_out)
  return out

def reshape(ctx: LoweringRuleContext, op, aval_out: core.AbstractValue) -> ir.Value:
  aval_out = core.physical_aval(aval_out)
  if not core.is_constant_shape(aval_out.shape):  # type: ignore
    shape = eval_dynamic_shape_as_tensor(ctx, aval_out.shape)  # type: ignore
    return hlo.dynamic_reshape(
        aval_to_ir_type(aval_out), op, shape,
    )
  else:
    return hlo.reshape(aval_to_ir_type(aval_out), op)

def slice_op(ctx: LoweringRuleContext, x, aval_out, *,
             start_indices, limit_indices, strides) -> ir.Value:
  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):
    elt_shape = core.physical_element_aval(aval_out.dtype).shape
    trailing_zeros = [0] * len(elt_shape)
    trailing_ones  = [1] * len(elt_shape)
    start_indices = (*start_indices, *trailing_zeros)
    limit_indices = (*limit_indices, *elt_shape)
    strides = (*strides, *trailing_ones)
    physical_aval_out = core.physical_aval(aval_out)
    return slice_op(ctx, x, physical_aval_out, start_indices=start_indices,
                    limit_indices=limit_indices, strides=strides)
  else:
    if any(not core.is_constant_shape(s) for s in (start_indices, limit_indices, strides)):
      start_indices = eval_dynamic_shape_as_tensor(ctx, start_indices)
      limit_indices = eval_dynamic_shape_as_tensor(ctx, limit_indices)
      strides = eval_dynamic_shape_as_tensor(ctx, strides)
      return hlo.real_dynamic_slice(
        aval_to_ir_type(aval_out),
        x, start_indices, limit_indices, strides)
    else:
      return hlo.slice(x,
                       dense_int_array(start_indices),
                       dense_int_array(limit_indices),
                       dense_int_array(strides))

def dynamic_slice(ctx: LoweringRuleContext, aval_out, x, *,
                  start_indices) -> ir.Value:
  x_aval = ctx.avals_in[0]
  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):
    elt_shape = core.physical_element_aval(aval_out.dtype).shape
    index_avals = ctx.avals_in[1:]
    dtype = dtypes.canonicalize_dtype(
        index_avals[0].dtype if index_avals else 'int64')  # type: ignore
    trailing_zeros = [ir_constant(np.array(0, dtype))] * len(elt_shape)
    start_indices = (*start_indices, *trailing_zeros)
    aval_out = core.physical_aval(aval_out)
    x_aval = core.physical_aval(x_aval)

  slice_sizes = aval_out.shape
  if not core.is_constant_shape(slice_sizes):
    # lax.dynamic_slice clamps the start indices, but we are going to
    # lower to RealDynamicSliceOp, which is a version of SliceOp, and does
    # not have the clamping behavior. We clamp start ourselves.
    slice_sizes = eval_dynamic_shape_as_tensor(ctx, slice_sizes)
    clamped_start = hlo.clamp(
      shape_tensor([0] * len(start_indices)),
      shape_tensor(start_indices),
      hlo.subtract(
        eval_dynamic_shape_as_tensor(ctx, x_aval.shape),  # type: ignore
        slice_sizes))
    return hlo.real_dynamic_slice(
        aval_to_ir_type(aval_out), x,
        clamped_start,
        hlo.add(clamped_start, slice_sizes),
        shape_tensor([1] * len(start_indices))
    )
  else:
    return hlo.dynamic_slice(x, start_indices, dense_int_array(slice_sizes))

def dynamic_update_slice(ctx: LoweringRuleContext, aval_out, x, update, *,
                         start_indices) -> ir.Value:
  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):
    elt_shape = core.physical_element_aval(aval_out.dtype).shape
    index_avals = ctx.avals_in[2:]
    dtype = dtypes.canonicalize_dtype(
        index_avals[0].dtype if index_avals else 'int64')  # type: ignore
    zeros = [ir_constant(np.array(0, dtype=dtype))] * len(elt_shape)
    start_indices = (*start_indices, *zeros)
    physical_aval_out = core.physical_aval(aval_out)
    return dynamic_update_slice(ctx, physical_aval_out, x, update,
                                start_indices=start_indices)
  else:
    # TODO(necula): handle dynamic shapes
    return hlo.dynamic_update_slice(x, update, start_indices)

def pad(ctx: LoweringRuleContext, aval_out,
        x, padding_value,
        padding_low, padding_high, padding_interior) -> ir.Value:
  if all(core.is_constant_shape(s) for s in (padding_low,
                                             padding_high, padding_interior)):
    return hlo.pad(x, padding_value,
                   dense_int_array(padding_low),
                   dense_int_array(padding_high),
                   dense_int_array(padding_interior))
  else:
    padding_low = eval_dynamic_shape_as_tensor(ctx, padding_low)
    padding_high = eval_dynamic_shape_as_tensor(ctx, padding_high)
    padding_interior = eval_dynamic_shape_as_tensor(ctx, padding_interior)
    return hlo.dynamic_pad(
        aval_to_ir_type(aval_out),
        x, padding_value, padding_low, padding_high, padding_interior)

def iota(ctx: LoweringRuleContext, aval_out, *, dimension: int):
  if not core.is_constant_shape(aval_out.shape):
    shape = eval_dynamic_shape_as_tensor(ctx, aval_out.shape)
    return hlo.dynamic_iota(
        aval_to_ir_type(aval_out),
        shape,
        i64_attr(dimension),
    )
  else:
    return hlo.iota(aval_to_ir_type(aval_out), i64_attr(dimension))

def full_like_aval(ctx: LoweringRuleContext, value, aval: core.ShapedArray) -> ir.Value:
  """Returns an IR constant shaped full of `value` shaped like `aval`."""
  zero = ir_constant(np.array(value, dtypes.canonicalize_dtype(aval.dtype)))
  return broadcast_in_dim(ctx, zero, aval, broadcast_dimensions=())

def add_jaxvals_lowering(ctx, x, y):
  if (isinstance(a := ctx.avals_in[0], core.ShapedArray) and
      dtypes.issubdtype(a.dtype, dtypes.extended)):
    return lower_fun(lambda x, y: [a.dtype._rules.add(a.dtype, x, y)])(ctx, x, y)
  return [hlo.add(x, y)]
register_lowering(ad_util.add_jaxvals_p, add_jaxvals_lowering)

register_lowering(ad_util.stop_gradient_p, lambda ctx, x: [x])


def compare_hlo(x, y, direction: str, comparison_type: str | None = None):
  """Creates CompareOp."""
  if comparison_type is None:
    elem_type = ir.RankedTensorType(x.type).element_type
    if isinstance(elem_type, ir.IntegerType):
      comparison_type = "UNSIGNED" if elem_type.is_unsigned else "SIGNED"
    else:
      comparison_type = "FLOAT"

  return hlo.compare(
      x,
      y,
      hlo.ComparisonDirectionAttr.get(direction),
      compare_type=hlo.ComparisonTypeAttr.get(comparison_type))

def _minmax_hlo(op, cmp, x, y):
  """Min/max that compares complex values lexicographically as pairs."""
  tensor_type = ir.RankedTensorType(x.type)
  if ir.ComplexType.isinstance(tensor_type.element_type):
    rx = hlo.real(x)
    ry = hlo.real(y)
    real_eq = compare_hlo(rx, ry, "EQ", "FLOAT")
    real_cmp = compare_hlo(rx, ry, cmp, "FLOAT")
    imag_cmp = compare_hlo(hlo.imag(x), hlo.imag(y), cmp, "FLOAT")
    which = hlo.select(real_eq, imag_cmp, real_cmp)
    return hlo.select(which, x, y)
  else:
    return op(x, y)

min_hlo = partial(_minmax_hlo, hlo.minimum, "LT")
max_hlo = partial(_minmax_hlo, hlo.maximum, "GT")


def convert_hlo(ctx: LoweringRuleContext, x, aval_in, aval_out):
  """Variant of convert that has HLO semantics.

  In particular, treat casts to boolean as x != 0, rather than truncating
  integer values (b/209440332)."""
  if (not dtypes.issubdtype(aval_out.dtype, dtypes.extended) and
      aval_out.dtype == np.dtype(np.bool_)):
    if dtypes.issubdtype(aval_in.dtype, np.inexact):
      compare_type = "FLOAT"
    elif dtypes.issubdtype(aval_in.dtype, np.signedinteger):
      compare_type = "SIGNED"
    else:
      compare_type = "UNSIGNED"
    x = compare_hlo(x, full_like_aval(ctx, 0, aval_in), "NE", compare_type)
    # continue, to adjust the shape if needed
  return hlo.convert(aval_to_ir_type(aval_out), x)

def _wrap_with_spmd_op(name: str,
                       ctx: LoweringRuleContext,
                       x: ir.Value,
                       aval_out: core.AbstractValue,
                       sharding: xc.OpSharding | SdyArray,
                       unspecified_dims: set[int] | None = None,
                       has_side_effect: bool = False,
                       allow_shardy_lowering: bool = False):
  if config.use_shardy_partitioner.value and allow_shardy_lowering:
    return dialects.sdy.ShardingConstraintOp(x, sharding.build()).result  # type: ignore

  # unspecified_dims indicate dimensions whose shardings are not specified and
  # XLA sharding propagation can change them.
  if unspecified_dims:
    backend_config = "unspecified_dims=[" + ",".join(
        [str(i) for i in sorted(unspecified_dims)]) + "]"
  else:
    backend_config = ""
  result_type = aval_to_ir_type(aval_out)
  assert isinstance(result_type, ir.Type), result_type
  out_shape = core.physical_aval(aval_out).shape  # type: ignore
  if core.is_constant_shape(out_shape):
    result_shapes = None
  else:
    result_shapes = [eval_dynamic_shape_as_tensor(ctx, out_shape)]

  op = custom_call(name, result_types=[result_type], operands=[x],
                   backend_config=backend_config,
                   api_version=1,
                   result_shapes=result_shapes,
                   has_side_effect=has_side_effect)
  set_sharding(op, sharding)
  return op.result


wrap_with_sharding_op = partial(_wrap_with_spmd_op, "Sharding",
                                allow_shardy_lowering=True)
wrap_with_full_to_shard_op = partial(_wrap_with_spmd_op, "SPMDFullToShardShape")
wrap_with_shard_to_full_op = partial(_wrap_with_spmd_op, "SPMDShardToFullShape")


def lower_with_sharding_in_types(ctx, op, aval, sharding_proto=None):
  if aval.sharding.mesh.empty:
    return op
  # Don't emit a wsc under full manual mode to avoid increasing HLO size.
  if aval.sharding.mesh._are_all_axes_manual:
    return op
  if aval.sharding.mesh._are_all_axes_auto:
    return op
  # TODO(yashkatariya): If all the axes in pspec are AUTO or collective,
  # `return op` early and avoid bloating HLO size.
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    aval = core.physical_aval(aval)
  if config.use_shardy_partitioner.value:
    proto = (aval.sharding._to_sdy_sharding(aval.ndim)
             if sharding_proto is None else sharding_proto)
    proto = modify_sdy_sharding_wrt_axis_types(proto, aval.sharding.mesh)
    return wrap_with_sharding_op(ctx, op, aval, proto)
  else:
    proto = (aval.sharding._to_xla_hlo_sharding(aval.ndim).to_proto()
            if sharding_proto is None else sharding_proto)
    unspecified_dims = None
    if aval.sharding.mesh._any_axis_auto:
      unspecified_dims = set(range(aval.ndim))
    return wrap_with_sharding_op(ctx, op, aval, proto, unspecified_dims)


def set_sharding(op, sharding: xc.OpSharding | SdyArray | SdyArrayList):
  if isinstance(sharding, (SdyArray, SdyArrayList)):
    op.attributes["sdy.sharding"] = get_sharding_attr(sharding)
  else:
    op.attributes["mhlo.sharding"] = get_sharding_attr(sharding)


def get_sharding_attr(
    sharding: xc.OpSharding | SdyArray | SdyArrayList
) -> ir.Attribute:
  if isinstance(sharding, (SdyArray, SdyArrayList)):
    return sharding.build()  # type: ignore
  else:
    # If there are very large numbers of devices, use the proto representation.
    # The MHLO to HLO conversion supports both, and the proto representation is
    # more compact.
    if len(sharding.tile_assignment_devices) > 100:  # type: ignore
      return ir.StringAttr.get(sharding.SerializeToString())  # type: ignore
    else:
      return ir.StringAttr.get(repr(xc.HloSharding.from_proto(sharding)))  # type: ignore[arg-type]


def wrap_with_layout_op(ctx: LoweringRuleContext,
                        x: ir.Value,
                        aval_out: core.AbstractValue,
                        layout: Layout,
                        aval_in: core.AbstractValue):
  result_type = aval_to_ir_type(aval_out)
  assert isinstance(result_type, ir.Type), result_type
  out_shape = core.physical_aval(aval_out).shape  # type: ignore
  if core.is_constant_shape(out_shape):
    result_shapes = None
  else:
    result_shapes = [eval_dynamic_shape_as_tensor(ctx, out_shape)]

  op = custom_call('LayoutConstraint', result_types=[result_type], operands=[x],
                   api_version=1,
                   result_shapes=result_shapes,
                   # Set operand layouts to anything. XLA will ignore it.
                   operand_layouts=[list(range(aval_in.ndim))],  # type: ignore
                   # TODO(yashkatariya): Figure out how to pass tiling to the
                   # custom call.
                   result_layouts=[layout.major_to_minor[::-1]])
  return op.result


# MLIR lowerings for lax primitives

def merge_mlir_modules(dst_module: ir.Module,
                       sym_name: str,
                       src_module: ir.Module,
                       dst_symtab: ir.SymbolTable | None = None) -> str:
  """
  Args:
    dst_module: the module into which the contents of src_module should be
      moved. Nothing in dst_module will be renamed.
    sym_name: the desired name for the "main" function of src_module after
      merging. This is a hint: the true name may be different because of symbol
      uniquification, and the true name is returned by this function.
    src_module: the module whose contents are to be alpha-renamed, set to
      private visibility, and merged into dst_module. src_module must contain
      exactly one symbol named "main".
    dst_symtab: the symbol table of `dst_module`

      Functions in src_module will be renamed such that they do not collide with
      functions in dst_module.

      This function mutates `src_module`. On return, `src_module` is left in an
      undefined state.

  Returns:
    the name of src_module's main() function, after renaming.
  """
  assert dst_module.context == src_module.context

  src_symtab = ir.SymbolTable(src_module.operation)
  dst_symtab = dst_symtab or ir.SymbolTable(dst_module.operation)
  used_names = set()

  # Rename all symbols in src_module that clash with names in dst_module, or
  # are the "main" symbol.
  renamings = {}
  for op in src_module.body.operations:
    name = op.sym_name.value
    should_rename = name in dst_symtab or name == "main"
    if should_rename:
      base_name = sym_name if name == "main" else name
      new_name = base_name
      i = 0
      # Replacements are chosen such that the new names are present in neither
      # src_module, dst_module, or the set of fresh names we've already used.
      # Since we rename names one at a time, if new names were in src_module,
      # they might themselves collide with a later renaming.
      while (new_name in src_symtab or new_name in dst_symtab or
             new_name in used_names):
        new_name = f"{base_name}_{i}"
        i += 1
      renamings[name] = new_name
      used_names.add(new_name)

  # Apply the symbol renamings to symbol definitions.
  private = ir.StringAttr.get("private")
  for op in src_module.body.operations:
    name = op.sym_name.value
    if name in renamings:
      src_symtab.set_symbol_name(op, renamings[name])
    op.attributes["sym_visibility"] = private

  # Apply the symbol renamings to symbol uses.
  for old_name, new_name in renamings.items():
    for op in src_module.body.operations:
      src_symtab.replace_all_symbol_uses(old_name, new_name, op)

  for op in src_module.body.operations:
    dst_module.body.append(op)
    dst_symtab.insert(op)

  return renamings["main"]


DEVICE_TO_DEVICE_TYPE = 1
SEND_TO_HOST_TYPE = 2
RECV_FROM_HOST_TYPE = 3

def build_mlir_module_helper(
    closed_jaxpr: core.ClosedJaxpr, *, name: str,
    platforms: Sequence[str],
    backend: xb.XlaBackend | None,
    axis_context: AxisContext) -> ir.Module:
  """Helper to generate pmap-style XLA computations for custom partitioners."""
  unlowerable_effects = lowerable_effects.filter_not_in(closed_jaxpr.effects)
  if unlowerable_effects:
    raise ValueError(f'Cannot lower jaxpr with effects: {closed_jaxpr.effects}')
  lowering_result = lower_jaxpr_to_module(name, closed_jaxpr,
      num_const_args=0,
      in_avals=closed_jaxpr.in_avals,
      backend=backend, ordered_effects=[],
      donated_args=[False] * len(closed_jaxpr.jaxpr.invars),
      axis_context=axis_context, platforms=platforms,
      lowering_parameters=LoweringParameters(hoist_constants_as_args=False))
  return lowering_result.module

def custom_call(
    call_target_name: str,
    *,
    result_types: Sequence[ir.Type],
    operands: Sequence[ir.Value],
    backend_config: str | bytes | dict[str, ir.Attribute] = "",
    has_side_effect: bool = False,
    result_shapes: Sequence[ir.Value] | None = None,
    called_computations: Sequence[str] = (),
    api_version: int = 2,
    operand_output_aliases: dict[int, int] | None = None,
    operand_layouts: Sequence[Sequence[int]] | None = None,
    result_layouts: Sequence[Sequence[int]] | None = None,
    extra_attributes: dict[str, ir.Attribute] | None = None,
) -> ir.Operation:
  """Helper function for building an hlo.CustomCall.

  Args:
    call_target_name: the name of the custom call target
    result_types: the MLIR types of the results of the custom call
    operands: the MLIR IR values that are arguments to the custom call
    backend_config: an opaque string passed to the custom call kernel
    has_side_effect: if True, marks the custom call as effectful
    result_shapes: tensors that represent the result shapes, to be used when
      the results have dynamic shapes. If not-None, its length must match the
      number of the results.
    called_computations: the list of function names called by the custom call.
    api_version: the ABI contract version of the custom call
    operand_output_aliases: a dict mapping operand numbers to outputs they alias
    operand_layouts: a sequence of layouts (dimension orders) for each operand
    result_layouts: a sequence of layouts (dimension orders) for each result
    extra_attributes: additional IR attributes to apply to the custom_call.
  """
  operands = list(operands)

  if backend_config is None:
    backend_config_attr = ir.StringAttr.get("")
  elif isinstance(backend_config, (str, bytes)):
    backend_config_attr = ir.StringAttr.get(backend_config)
  elif isinstance(backend_config, dict):
    # TODO(necula): it seems that the CustomCallOp constructor requires that
    # backend_config_attr be a string attribute, even though in some cases we
    # need it to be a DictAttr, e.g., for ApproxTopK on TPU.
    # "Verification failed: 'stablehlo.custom_call' op attribute 'backend_config' failed to satisfy constraint: string attribute"
    # To workaround this limitation we first set it to the empty string and we
    # use an unregistered attribute mhlo.backend_config to hold the DictAttr.
    # We must also use api_version=1 to ensure that mhlo.backend_config is
    # handled properly.
    backend_config_attr = ir.StringAttr.get("")
    api_version = 1
  else:
    raise ValueError("custom_call backend_config unexpected type: " + str(backend_config))
  attributes = dict(
      call_target_name=ir.StringAttr.get(call_target_name),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      backend_config=backend_config_attr,
      api_version=i32_attr(api_version),
      called_computations=ir.ArrayAttr.get(
          [ir.FlatSymbolRefAttr.get(name) for name in called_computations]
      ),
  )
  if operand_output_aliases is not None:
    attributes["output_operand_aliases"] = ir.ArrayAttr.get([
      hlo.OutputOperandAlias.get(
          # if len(result_types) == 1 then the aliasing refers implicitly to
          # the only output.
          output_tuple_indices=[output_idx] if len(result_types) > 1 else [],
          operand_index=input_idx,
          operand_tuple_indices=[],
      )
      for input_idx, output_idx in (operand_output_aliases.items() or ())
    ])

  if extra_attributes is not None:
    attributes.update(extra_attributes)

  if result_shapes is not None:
    # We add the result_shapes at the end of the operands, and must pass
    # the indices_of_output_operands attribute. This attribute is not yet
    # accepted by the CustomCall constructor, so we use build_generic
    attributes["indices_of_shape_operands"] = ir.DenseIntElementsAttr.get(
        np.asarray(list(range(len(operands), len(operands) + len(result_shapes))),
                   dtype=np.int64))
    if operand_layouts is not None:
      assert len(operand_layouts) == len(operands), (operand_layouts, operands)
      operand_layouts = list(operand_layouts) + [(0,)] * len(result_shapes)
    operands = list(operands) + list(result_shapes)

  if operand_layouts is not None:
    attributes["operand_layouts"] = ir.ArrayAttr.get([
        ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(l, dtype=np.int64)),
            type=ir.IndexType.get()) for l in operand_layouts
    ])
  if result_layouts is not None:
    assert result_layouts is not None
    assert len(result_layouts) == len(result_types), (
        result_layouts, result_types)
    attributes["result_layouts"] = ir.ArrayAttr.get([
        ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(l, dtype=np.int64)),
            type=ir.IndexType.get()) for l in result_layouts
    ])

  op = hlo.CustomCallOp.build_generic(results=result_types, operands=operands,
                                      attributes=attributes)
  if isinstance(backend_config, dict):
    op.operation.attributes["mhlo.backend_config"] = ir.DictAttr.get(
        backend_config)
  return op


def reduce_window(
    ctx: LoweringRuleContext,
    *,
    # Base name to be used for the reducer function
    reducer_name: str,
    # Compute the reducer body given the reducer.
    reducer_body: Callable[[ir.Block], Sequence[ir.Value]],
    operands: Sequence[ir.Value],
    init_values: Sequence[ir.Value],
    init_values_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue],
    window_dimensions, window_strides, padding, base_dilation, window_dilation):
  """Builds a ReduceWindowOp, with support for dynamic shapes."""

  scalar_types = flatten_ir_types([aval_to_ir_type(aval) for aval in init_values_avals])
  if any(not core.is_constant_shape(s)
         for s in [window_dimensions, window_dilation, window_strides, base_dilation, *padding]):
    # d_padding will be an array i32[N, 2] with pad_lo and pad_hi for each
    # spatial dimension.
    int2d = aval_to_ir_type(core.ShapedArray((1, 2), np.int32))
    def prep_one_pad(pad_lo_hi: tuple[core.DimSize, core.DimSize]):
      pads = eval_dynamic_shape_as_tensor(ctx, pad_lo_hi)  # i32[2]
      return hlo.reshape(int2d, pads)
    d_padding = hlo.concatenate(list(map(prep_one_pad, padding)), i64_attr(0))
    # Build the reducer
    reducer_type = ir.FunctionType.get(
      scalar_types + scalar_types, scalar_types)
    with ir.InsertionPoint.at_block_begin(ctx.module_context.module.body):
      reducer = func_dialect.FuncOp(reducer_name, reducer_type)
    ctx.module_context.symbol_table.insert(reducer)
    entry_block = reducer.add_entry_block()
    with ir.InsertionPoint(entry_block):
      hlo.return_(reducer_body(entry_block))

    rw = custom_call(
      "stablehlo.dynamic_reduce_window",
      result_types=flatten_ir_types(map(aval_to_ir_type, out_avals)),
      operands=[
        *operands, *init_values,
        eval_dynamic_shape_as_tensor(ctx, window_dimensions),
        eval_dynamic_shape_as_tensor(ctx, window_strides),
        eval_dynamic_shape_as_tensor(ctx, base_dilation),
        eval_dynamic_shape_as_tensor(ctx, window_dilation),
        d_padding],
       called_computations=[reducer.name.value],
    )
  else:  # Static shapes
    rw = hlo.ReduceWindowOp(
        list(map(aval_to_ir_type, out_avals)),
        operands, init_values,
        dense_int_array(window_dimensions),
        window_strides=dense_int_array(window_strides),
        base_dilations=dense_int_array(base_dilation),
        window_dilations=dense_int_array(window_dilation),
        padding=ir.DenseIntElementsAttr.get(np.asarray(padding, np.int64),
                                            shape=[len(padding), 2]))
    reducer = rw.regions[0].blocks.append(*(scalar_types + scalar_types))
    with ir.InsertionPoint(reducer):
      hlo.return_(reducer_body(reducer))
  return [lower_with_sharding_in_types(ctx, r, aval)
          for r, aval in zip(rw.results, ctx.avals_out)]


def refine_polymorphic_shapes(module: ir.Module) -> ir.Module:
  """Refines the polymorphic shapes inside a module.

  Given a module with static input shapes, but using dynamic shapes due to
  shape polymorphism, runs shape refinement to resolve all the dynamic shapes.
  Then verifies that there are no more dynamic shapes in the module.
  """
  try:
    refine_polymorphic_shapes = partial(_jax.mlir.refine_polymorphic_shapes,
            mlir_module=module_to_bytecode(module),
            enable_shape_assertions=True,
            validate_static_shapes=True)
    refined_module_str = refine_polymorphic_shapes(
        enable_shardy=config.use_shardy_partitioner.value)
  except Exception as e:
    raise ValueError(
        "Error refining shapes. " +
        dump_module_message(module, "before_refine_polymorphic_shapes")) from e

  context = make_ir_context()
  with context:
    return ir.Module.parse(refined_module_str)

########################### pvary ##################################

register_lowering(core.pvary_p, lambda ctx, *x, axes, axis_index_groups: x)
