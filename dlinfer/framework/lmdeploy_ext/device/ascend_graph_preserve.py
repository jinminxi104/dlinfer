import dataclasses
import glob
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

ENABLE_ENV = 'LMDEPLOY_ASCEND_PRESERVE_GRAPHS'
VLLM_ASCEND_PATH_ENV = 'LMDEPLOY_ASCEND_VLLM_ASCEND_PATH'

HandleType = tuple[int, int, int, int]


def _env_to_bool(env_var: str, default: bool = False) -> bool:
    value = os.getenv(env_var)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def enabled() -> bool:
    return _env_to_bool(ENABLE_ENV, False)


def skip_empty_cache() -> bool:
    return enabled()


@dataclasses.dataclass(frozen=True)
class _RuntimeBindings:
    lib_path: str
    init_module: Callable[[Callable[[HandleType], None], Callable[[int], HandleType]], None]
    python_create_and_map: Callable[[int, int, int, int], None]
    python_unmap_and_release: Callable[[int, int, int, int], None]
    memcpy: Callable[..., Any]


_RUNTIME_BINDINGS: _RuntimeBindings | None = None


def _prepare_runtime_import_path() -> None:
    extra_path = os.getenv(VLLM_ASCEND_PATH_ENV)
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

    cann_paths = sorted(glob.glob('/usr/local/Ascend/cann-*/python/site-packages'))
    for path in reversed(cann_paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def _check_allocator_conf() -> None:
    alloc_conf = os.environ.get('PYTORCH_NPU_ALLOC_CONF', '')
    if 'expandable_segments:True' in alloc_conf:
        raise RuntimeError(
            'LMDEPLOY_ASCEND_PRESERVE_GRAPHS requires PYTORCH_NPU_ALLOC_CONF without '
            '"expandable_segments:True" because the NPU pluggable allocator is incompatible with it.'
        )


def _load_runtime() -> _RuntimeBindings:
    global _RUNTIME_BINDINGS
    if _RUNTIME_BINDINGS is not None:
        return _RUNTIME_BINDINGS

    _check_allocator_conf()
    _prepare_runtime_import_path()

    try:
        import acl  # type: ignore
        import vllm_ascend.vllm_ascend_C as ext
    except Exception as exc:
        raise RuntimeError(
            'Failed to load vllm_ascend_C. Ensure torch is importable, '
            'LMDEPLOY_ASCEND_VLLM_ASCEND_PATH points to the updated vllm-ascend checkout, '
            'and the shared object is available.'
        ) from exc

    required = ('init_module', 'python_create_and_map', 'python_unmap_and_release')
    missing = [name for name in required if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f'vllm_ascend_C is missing required symbols: {missing}')

    lib_path = getattr(ext, '__file__', None)
    if lib_path is None:
        raise RuntimeError('Failed to resolve vllm_ascend_C shared object path.')

    _RUNTIME_BINDINGS = _RuntimeBindings(
        lib_path=lib_path,
        init_module=ext.init_module,
        python_create_and_map=ext.python_create_and_map,
        python_unmap_and_release=ext.python_unmap_and_release,
        memcpy=acl.rt.memcpy,
    )
    logger.info(f'Loaded vllm_ascend_C from {lib_path}.')
    return _RUNTIME_BINDINGS


def ensure_available() -> bool:
    if not enabled():
        return False
    _load_runtime()
    return True


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    is_mapped: bool = True
    cpu_backup_tensor: torch.Tensor | None = None


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(value) < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    if unit == 'B':
        return f'{int(value)}{unit}'
    return f'{value:.2f}{unit}'


def create_and_map(allocation_handle: HandleType) -> None:
    _load_runtime().python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    _load_runtime().python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
):
    runtime = _load_runtime()
    runtime.init_module(python_malloc_fn, python_free_func)
    return torch.npu.memory.NPUPluggableAllocator(runtime.lib_path, 'my_malloc', 'my_free')


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
):
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.npu.memory.MemPool(new_alloc._allocator)
    with torch.npu.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class AscendGraphMemoryPool:
    instance = None
    default_tag: str = 'default'

    @staticmethod
    def get_instance() -> 'AscendGraphMemoryPool':
        if AscendGraphMemoryPool.instance is None:
            AscendGraphMemoryPool.instance = AscendGraphMemoryPool()
        return AscendGraphMemoryPool.instance

    def __init__(self):
        _check_allocator_conf()
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = AscendGraphMemoryPool.default_tag
        self.allocator_and_pools: dict[str, Any] = {}

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        ptr = allocation_handle[2]
        self.pointer_to_data[ptr] = AllocationData(allocation_handle, self.current_tag)

    def python_free_callback(self, ptr: int) -> HandleType:
        data = self.pointer_to_data.pop(ptr)
        data.cpu_backup_tensor = None
        return data.handle

    def _backup_to_cpu(self, ptr: int, data: AllocationData) -> None:
        size_in_bytes = data.handle[1]
        cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)
        cpu_ptr = cpu_backup_tensor.data_ptr()
        acl_memcpy_device_to_host = 2
        _load_runtime().memcpy(cpu_ptr, cpu_ptr + size_in_bytes * 2, ptr, size_in_bytes, acl_memcpy_device_to_host)
        data.cpu_backup_tensor = cpu_backup_tensor

    def _restore_from_cpu(self, ptr: int, data: AllocationData) -> None:
        cpu_backup_tensor = data.cpu_backup_tensor
        if cpu_backup_tensor is None:
            return
        size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
        cpu_ptr = cpu_backup_tensor.data_ptr()
        acl_memcpy_host_to_device = 1
        _load_runtime().memcpy(ptr, ptr + size_in_bytes * 2, cpu_ptr, size_in_bytes, acl_memcpy_host_to_device)
        data.cpu_backup_tensor = None

    def sleep(self, level: int) -> None:
        logger.info(
            'Ascend preserved graph allocator sleep(level=%s): %s',
            level,
            self.format_snapshot(),
        )
        offload_tags = {'weights'} if level == 1 else set()
        for ptr, data in self.pointer_to_data.items():
            if not data.is_mapped:
                continue
            if data.tag in offload_tags:
                self._backup_to_cpu(ptr, data)
            else:
                data.cpu_backup_tensor = None
            unmap_and_release(data.handle)
            data.is_mapped = False
        logger.info(
            'Ascend preserved graph allocator slept(level=%s): %s',
            level,
            self.format_snapshot(),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.info(
            'Ascend preserved graph allocator wakeup(tags=%s): %s',
            tags if tags is not None else 'all',
            self.format_snapshot(),
        )
        for ptr, data in self.pointer_to_data.items():
            if tags is not None and data.tag not in tags:
                continue
            if data.is_mapped:
                continue
            handle = data.handle
            create_and_map(handle)
            data.is_mapped = True
            self._restore_from_cpu(ptr, data)
        logger.info(
            'Ascend preserved graph allocator woke(tags=%s): %s',
            tags if tags is not None else 'all',
            self.format_snapshot(),
        )

    def snapshot(self) -> dict[str, Any]:
        total_tracked_bytes = 0
        total_mapped_bytes = 0
        tags: dict[str, dict[str, int]] = {}

        for data in self.pointer_to_data.values():
            size_in_bytes = data.handle[1]
            total_tracked_bytes += size_in_bytes
            if data.is_mapped:
                total_mapped_bytes += size_in_bytes

            tag_stats = tags.setdefault(
                data.tag,
                {
                    'allocations': 0,
                    'tracked_bytes': 0,
                    'mapped_allocations': 0,
                    'mapped_bytes': 0,
                    'backed_up_allocations': 0,
                    'backed_up_bytes': 0,
                },
            )
            tag_stats['allocations'] += 1
            tag_stats['tracked_bytes'] += size_in_bytes
            if data.is_mapped:
                tag_stats['mapped_allocations'] += 1
                tag_stats['mapped_bytes'] += size_in_bytes
            if data.cpu_backup_tensor is not None:
                tag_stats['backed_up_allocations'] += 1
                tag_stats['backed_up_bytes'] += size_in_bytes

        return {
            'allocations': len(self.pointer_to_data),
            'total_tracked_bytes': total_tracked_bytes,
            'total_mapped_bytes': total_mapped_bytes,
            'tags': tags,
        }

    def format_snapshot(self, snapshot: dict[str, Any] | None = None) -> str:
        snapshot = self.snapshot() if snapshot is None else snapshot
        tags = snapshot['tags']
        tag_summaries = []
        for tag in sorted(tags):
            stats = tags[tag]
            tag_summaries.append(
                f'{tag}(allocs={stats["allocations"]},tracked={_format_bytes(stats["tracked_bytes"])},'
                f'mapped_allocs={stats["mapped_allocations"]},mapped={_format_bytes(stats["mapped_bytes"])},'
                f'backed_up_allocs={stats["backed_up_allocations"]},'
                f'backed_up={_format_bytes(stats["backed_up_bytes"])})'
            )
        tags_repr = ', '.join(tag_summaries) if tag_summaries else 'none'
        return (
            f'allocs={snapshot["allocations"]}, '
            f'tracked={_format_bytes(snapshot["total_tracked_bytes"])}, '
            f'mapped={_format_bytes(snapshot["total_mapped_bytes"])}, '
            f'tags=[{tags_repr}]'
        )

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        if tag is None:
            tag = AscendGraphMemoryPool.default_tag

        old_tag = self.current_tag
        self.current_tag = tag
        with use_memory_pool_with_allocator(self.python_malloc_callback, self.python_free_callback) as data:
            self.allocator_and_pools[tag] = data
            try:
                yield
            finally:
                self.current_tag = old_tag


@contextmanager
def use_memory_pool(tag: str | None = None):
    if not enabled():
        yield
        return
    ensure_available()
    allocator = AscendGraphMemoryPool.get_instance()
    with allocator.use_memory_pool(tag):
        yield


def sleep(level: int) -> None:
    ensure_available()
    allocator = AscendGraphMemoryPool.get_instance()
    allocator.sleep(level)


def wakeup(tags: list[str] | None = None) -> None:
    ensure_available()
    allocator = AscendGraphMemoryPool.get_instance()
    allocator.wake_up(tags=tags)


def get_allocator_snapshot() -> dict[str, Any]:
    ensure_available()
    allocator = AscendGraphMemoryPool.get_instance()
    return allocator.snapshot()


def format_allocator_snapshot(snapshot: dict[str, Any] | None = None) -> str:
    ensure_available()
    allocator = AscendGraphMemoryPool.get_instance()
    return allocator.format_snapshot(snapshot)
