import ctypes
from pathlib import Path

import numpy as np
import torch
from cuda.bindings import driver

from ._kernels import BM, BN, BN_LOCAL, STORE_N, KernelSpec


TMA_UINT8 = 0
TMA_BFLOAT16 = 9
TMA_SWIZZLE_NONE = 0
TMA_SWIZZLE_128B = 3

_libcuda = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
_encode_tiled = _libcuda.cuTensorMapEncodeTiled
_encode_tiled.restype = ctypes.c_int
_encode_tiled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def _cu(result):
    err, *values = result
    if err != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(err)
        raise RuntimeError(f"CUDA driver error: {name.decode()}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _tensor_map(
    *,
    dtype,
    rank,
    pointer,
    global_dim,
    global_strides,
    box_dim,
    swizzle,
):
    descriptor = np.zeros(128, dtype=np.uint8)
    dimensions = (ctypes.c_uint64 * rank)(*global_dim)
    strides = (ctypes.c_uint64 * (rank - 1))(*global_strides)
    box = (ctypes.c_uint32 * rank)(*box_dim)
    element_strides = (ctypes.c_uint32 * rank)(*([1] * rank))
    result = _encode_tiled(
        descriptor.ctypes.data,
        dtype,
        rank,
        pointer,
        dimensions,
        strides,
        box,
        element_strides,
        0,        # interleave
        swizzle,
        0,        # L2 promotion
        0,        # OOB fill
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUresult={result}")
    return descriptor


def _value_map(tensor, rows, cols, tile_rows, tile_cols, element_bytes, dtype):
    swizzle_elements = 128 // element_bytes
    return _tensor_map(
        dtype=dtype,
        rank=5,
        pointer=tensor.data_ptr(),
        global_dim=[
            swizzle_elements,
            rows,
            cols // swizzle_elements,
            1,
            1,
        ],
        global_strides=[
            cols * element_bytes,
            128,
            rows * cols * element_bytes,
            rows * cols * element_bytes,
        ],
        box_dim=[
            swizzle_elements,
            tile_rows,
            tile_cols // swizzle_elements,
            1,
            1,
        ],
        swizzle=TMA_SWIZZLE_128B,
    )


def _scale_map(tensor, outer, k_tiles):
    return _tensor_map(
        dtype=TMA_UINT8,
        rank=4,
        pointer=tensor.data_ptr(),
        global_dim=[16, 32, k_tiles, outer],
        global_strides=[16, 32 * 16, k_tiles * 32 * 16],
        box_dim=[16, 32, 1, 1],
        swizzle=TMA_SWIZZLE_NONE,
    )


def _by_value(descriptor):
    return (ctypes.c_byte * 128).from_buffer_copy(descriptor.tobytes())


class Runtime:
    def __init__(self, device_index):
        major, minor = torch.cuda.get_device_capability(device_index)
        if (major, minor) != (10, 0):
            raise RuntimeError(
                "MMC's bundled kernels require a B200-class sm_100 GPU; "
                f"device {device_index} reports compute capability {major}.{minor}"
            )

        _cu(driver.cuInit(0))
        self.device_index = device_index
        self.device = _cu(driver.cuDeviceGet(device_index))
        self.context = _cu(driver.cuDevicePrimaryCtxRetain(self.device))
        _cu(driver.cuCtxSetCurrent(self.context))
        self.sm_count = _cu(driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            self.device,
        ))
        self.driver_version = _cu(driver.cuDriverGetVersion())
        self._functions = {}
        self._launch_cache = {}

    def _function(self, spec):
        if spec.name not in self._functions:
            path = Path(__file__).with_name("cubins") / f"{spec.name}.cubin"
            module = _cu(driver.cuModuleLoadData(path.read_bytes()))
            function = _cu(driver.cuModuleGetFunction(module, b"matmul_cluster"))
            _cu(driver.cuFuncSetAttribute(
                function,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                spec.shared_bytes,
            ))
            self._functions[spec.name] = (module, function)
        return self._functions[spec.name][1]

    def _build_launch_args(self, spec, a, b, sfa, sfb, out, stream):
        m, k = a.shape
        n = b.shape[0]
        descriptors = (
            _value_map(a, m, k, BM, spec.bk, 1, TMA_UINT8),
            _scale_map(sfa, m // BM, k // 128),
            _value_map(b, n, k, BN_LOCAL, spec.bk, 1, TMA_UINT8),
            _scale_map(sfb, n // BN_LOCAL, k // 128),
            _value_map(out, m, n, BM, STORE_N, 2, TMA_BFLOAT16),
        )
        argument_storage = [_by_value(descriptor) for descriptor in descriptors]
        argument_storage.extend([
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
        ])
        kernel_params = (ctypes.c_void_p * len(argument_storage))(
            *[ctypes.addressof(arg) for arg in argument_storage]
        )
        grid = self.sm_count - self.sm_count % 2
        function = self._function(spec)
        launch_args = (
            function,
            grid, 1, 1,
            spec.threads, 1, 1,
            spec.shared_bytes,
            stream,
            kernel_params,
            0,
        )
        # kernel_params contains addresses into argument_storage, so retain both.
        return launch_args, argument_storage

    def launch(self, spec, a, b, sfa, sfb, out):
        m, k = a.shape
        n = b.shape[0]
        stream = torch.cuda.current_stream(self.device_index).cuda_stream
        key = (
            spec.name,
            a.data_ptr(),
            b.data_ptr(),
            sfa.data_ptr(),
            sfb.data_ptr(),
            out.data_ptr(),
            m,
            n,
            k,
            stream,
        )
        if key not in self._launch_cache:
            self._launch_cache[key] = self._build_launch_args(
                spec, a, b, sfa, sfb, out, stream
            )

        # Keep the ctypes objects behind kernel_params from being garbage-collected.
        launch_args, _argument_storage = self._launch_cache[key]
        _cu(driver.cuLaunchKernel(*launch_args))


_RUNTIMES = {}


def runtime_for(device_index):
    if device_index not in _RUNTIMES:
        _RUNTIMES[device_index] = Runtime(device_index)
    return _RUNTIMES[device_index]
