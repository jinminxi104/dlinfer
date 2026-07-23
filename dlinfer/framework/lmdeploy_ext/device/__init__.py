# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
import torch
from functools import lru_cache
from dlinfer.vendor import vendor_name

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

vendor = ["camb", "ascend"]


def fake_torch_compile(dynamic=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def pre_rms_norm(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance_q = (q * q).sum(-1, keepdim=True)
    variance_k = (k * k).sum(-1, keepdim=True)
    variance = torch.stack([variance_q, variance_k], dim=0)
    return variance


def post_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    variance: torch.Tensor,
    eps: float,
    embed_dim: int,
    dtype: torch.dtype,
):
    """Post rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance = variance / embed_dim + eps
    variance_q, variance_k = variance
    q = q * torch.rsqrt(variance_q)
    q = q.to(dtype) * weight_q
    k = k * torch.rsqrt(variance_k)
    k = k.to(dtype) * weight_k
    return q, k


def patch_compiled_func():
    import torch

    real_torch_compile = torch.compile
    torch.compile = fake_torch_compile
    from lmdeploy.pytorch.models import internvl, internvl3_hf

    internvl.pre_rms_norm = pre_rms_norm
    internvl.post_rms_norm = post_rms_norm
    internvl3_hf.pre_rms_norm = pre_rms_norm
    internvl3_hf.post_rms_norm = post_rms_norm
    torch.compile = real_torch_compile


def patch_async_sampling_logits():
    from torch.profiler import record_function
    from lmdeploy.pytorch.engine.model_agent import BaseModelAgent
    from lmdeploy.pytorch.engine.model_agent.agent import BatchedLogProbs
    from lmdeploy.pytorch.engine.logits_process import (
        SamplingInputs,
        FusedLogitsProcessor,
    )

    async def async_sampling_logits(
        self, logits: torch.Tensor, sampling_inputs: SamplingInputs
    ):
        """Sampling logits."""
        # record function does not support async function
        # so we can not decorate it on async_sampling_logits
        with record_function("sampling_logits"):
            logits = logits.to(torch.float32)
            logits_processor = FusedLogitsProcessor(
                sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
                guided_decoding_manager=self.guided_decoding_manager,
            )
            origin_logits = logits
            logits, raw_logprobs = await logits_processor(origin_logits)
            next_token_ids = logits_processor.sampling(logits)
            logprobs = logits_processor.compute_logprobs(raw_logprobs, next_token_ids)
            if logprobs is not None:
                logprobs = BatchedLogProbs(
                    vals=logprobs[0],
                    indices=logprobs[1],
                )

        return next_token_ids, logprobs

    BaseModelAgent.async_sampling_logits = async_sampling_logits


##### patch cache engine #####
def patch_contiguous_cache_engine():
    from lmdeploy.pytorch.config import CacheConfig, ModelConfig
    from functools import reduce
    from math import gcd
    from lmdeploy.pytorch.engine import cache_engine

    @classmethod
    def _cache_engine_allocate_caches(
        cls,
        num_blocks: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        world_size: int,
        device: str,
    ):
        """Allocate caches."""
        num_layers = model_config.num_layers

        # get all descs
        k_cache_desc = cls.get_k_cache_desc(model_config, cache_config, world_size)
        v_cache_desc = cls.get_v_cache_desc(model_config, cache_config, world_size)
        quant_cache_descs = cls.get_quant_cache_descs(
            k_cache_desc, v_cache_desc, model_config, cache_config
        )
        custom_cache_descs = cls.get_custom_cache_descs(model_config, cache_config)
        cache_descs = (
            [k_cache_desc, v_cache_desc] + quant_cache_descs + custom_cache_descs
        )

        # get mempool size
        mem_pool_size = 0
        alignments = []
        for desc in cache_descs:
            mem_pool_size += desc.aligned_size
            alignments.append(desc.alignment)

        # compute gcd of alignments
        alignments_gcd = reduce(gcd, alignments) if alignments else 1
        assert (
            mem_pool_size % alignments_gcd == 0
        ), "mem_pool_size must be divisible by alignments_gcd"

        # create pool
        mem_pool = torch.zeros(
            (mem_pool_size // alignments_gcd, num_layers, num_blocks, alignments_gcd),
            dtype=torch.uint8,
            device=device,
        )

        # slice caches
        caches = []
        remain_pool = mem_pool
        for desc in cache_descs:
            cache = (
                remain_pool[: desc.size // alignments_gcd, :, :, :]
                .view(desc.dtype)
                .view((num_layers, num_blocks, *desc.shape))
            )
            remain_pool = remain_pool[desc.aligned_size // alignments_gcd :, :, :, :]
            caches.append(cache)
        return mem_pool, caches

    cache_engine.CacheEngine.allocate_caches = _cache_engine_allocate_caches


##### patch state cache engine #####
def patch_state_cache_engine():
    from typing import List, Tuple
    from lmdeploy.pytorch.engine import cache_engine
    from lmdeploy.pytorch.engine.cache_engine import CacheDesc

    @staticmethod
    def _state_cache_engine_allocate_caches(
        num_caches: int,
        state_shapes: List[Tuple[Tuple[int], torch.dtype]],
        device: torch.device,
    ):
        """Allocate cache implement.

        Each state is allocated as an independent contiguous tensor of shape
        (num_caches, *shape).  A single shared pool of shape
        (num_caches, total_pool_size) would give views with stride[0] ==
        total_pool_size instead of the per-state numel, making every slice
        non-contiguous and breaking NPU ops that require contiguous input.
        """

        cache_dtype = torch.int8
        if len(state_shapes) == 0 or num_caches == 0:
            return torch.empty((0, 0), dtype=cache_dtype, device=device), []

        cache_descs = [CacheDesc(shape, dtype) for shape, dtype in state_shapes]

        # Allocate each state as a separate contiguous tensor.
        caches = []
        for desc in cache_descs:
            cache = torch.zeros(
                (num_caches, *desc.shape), dtype=desc.dtype, device=device
            )
            caches.append(cache)

        # mem_pool is used by two callers:
        #   1. get_cache_state_size(): always calls with device='meta' to compute byte
        #      counts — the tensor is never materialised on a real device.
        #   2. init_caches(): patched below to zero individual caches directly, so it
        #      no longer touches mem_pool at all.
        # Therefore we only need a correctly-sized pool on 'meta'; for real devices we
        # return an empty placeholder to avoid doubling the state-cache memory footprint.
        total_bytes = sum(desc.aligned_size for desc in cache_descs)
        if str(device) == "meta":
            mem_pool = torch.empty(
                (num_caches, total_bytes), dtype=cache_dtype, device=device
            )
        else:
            mem_pool = torch.empty(0, dtype=cache_dtype, device=device)
        return mem_pool, caches

    def _state_cache_engine_init_caches(self, idx: torch.Tensor, mask: torch.Tensor):
        """Initialize state caches by zeroing each individual cache tensor."""
        if idx is None:
            return
        if len(self._state_caches) <= 0:
            return
        num_caches = self.cache_config.num_state_caches
        cache_masks = torch.zeros((num_caches,), dtype=torch.bool, device=idx.device)
        cache_masks.index_copy_(0, idx, mask)
        for cache in self._state_caches:
            reshaped_mask = cache_masks.view((-1,) + (1,) * (cache.dim() - 1))
            cache.masked_fill_(reshaped_mask, 0)

    cache_engine.StateCacheEngine.allocate_caches = _state_cache_engine_allocate_caches
    cache_engine.StateCacheEngine.init_caches = _state_cache_engine_init_caches


def patch_ascend_preserved_graph_sleep():
    from lmdeploy.pytorch.engine.model_agent import agent as agent_mod
    from lmdeploy.pytorch.weight_loader import model_weight_loader as weight_loader_mod
    from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import (
        capture_graph_input_buffer_snapshot,
    )

    from . import ascend_graph_preserve

    if getattr(agent_mod.BaseModelAgent, '_dlinfer_ascend_preserve_graph_patched', False):
        return

    if ascend_graph_preserve.enabled():
        ascend_graph_preserve.ensure_available()

    origin_build_model = agent_mod.BaseModelAgent.build_model
    origin_build_cache_engine = agent_mod.BaseModelAgent.build_cache_engine
    origin_get_free_mem = agent_mod.BaseModelAgent.get_free_mem
    origin_update_params = agent_mod.BaseModelAgent.update_params
    origin_sleep = agent_mod.BaseModelAgent.sleep
    origin_wakeup = agent_mod.BaseModelAgent.wakeup
    origin_release = agent_mod.BaseModelAgent.release

    def _capture_ptr_snapshot(self):
        snapshot = dict(weights=None, kv_cache=None, state_cache=None, state_caches=None, spec_kv_cache=None)
        if self.patched_model is not None:
            model = self.patched_model.get_model()
            try:
                name, param = next(model.named_parameters())
                snapshot['weights'] = (name, param.data_ptr())
            except StopIteration:
                snapshot['weights'] = None
        if self.cache_engine is not None and hasattr(self.cache_engine, 'full_gpu_cache'):
            snapshot['kv_cache'] = self.cache_engine.full_gpu_cache.data_ptr()
        if self.state_cache_engine is not None:
            mem_pool = getattr(self.state_cache_engine, 'mem_pool', None)
            if torch.is_tensor(mem_pool):
                snapshot['state_cache'] = mem_pool.data_ptr()
            state_caches = getattr(self.state_cache_engine, 'state_caches', ())
            snapshot['state_caches'] = tuple(cache.data_ptr() for cache in state_caches if torch.is_tensor(cache))
        spec_cache_engine = getattr(self.spec_agent, 'cache_engine', None)
        if spec_cache_engine is not None and hasattr(spec_cache_engine, 'full_gpu_cache'):
            snapshot['spec_kv_cache'] = spec_cache_engine.full_gpu_cache.data_ptr()
        return snapshot

    def _capture_input_buffer_snapshot(self):
        if self.patched_model is None:
            return {}
        return capture_graph_input_buffer_snapshot(self.patched_model)

    def _log_ptr_snapshot(action: str, rank: int, snapshot: dict):
        logger.info(
            'Ascend preserved graph %s rank[%s]: weight=%s kv_cache=%s state_cache=%s spec_kv_cache=%s',
            action,
            rank,
            snapshot.get('weights'),
            snapshot.get('kv_cache'),
            snapshot.get('state_cache'),
            snapshot.get('spec_kv_cache'),
        )

    def _log_input_buffer_snapshot(action: str, rank: int, snapshot: dict):
        if not snapshot:
            return
        logger.debug(
            'Ascend preserved graph %s rank[%s]: input_buffers=%s',
            action,
            rank,
            snapshot,
        )

    def _assert_ptrs_unchanged(rank: int, before: dict, after: dict, tags: list[str]):
        if 'weights' in tags and before.get('weights') != after.get('weights'):
            raise RuntimeError(
                f'Preserved-graph weight dataptr changed on rank[{rank}]: '
                f'before={before.get("weights")} after={after.get("weights")}'
            )
        if 'kv_cache' in tags and before.get('kv_cache') != after.get('kv_cache'):
            raise RuntimeError(
                f'Preserved-graph kv_cache dataptr changed on rank[{rank}]: '
                f'before={before.get("kv_cache")} after={after.get("kv_cache")}'
            )
        if 'kv_cache' in tags and before.get('state_cache') != after.get('state_cache'):
            raise RuntimeError(
                f'Preserved-graph state_cache dataptr changed on rank[{rank}]: '
                f'before={before.get("state_cache")} after={after.get("state_cache")}'
            )
        if 'kv_cache' in tags and before.get('state_caches') != after.get('state_caches'):
            raise RuntimeError(
                f'Preserved-graph state_caches dataptr changed on rank[{rank}]: '
                f'before={before.get("state_caches")} after={after.get("state_caches")}'
            )
        if 'kv_cache' in tags and before.get('spec_kv_cache') != after.get('spec_kv_cache'):
            raise RuntimeError(
                f'Preserved-graph spec kv_cache dataptr changed on rank[{rank}]: '
                f'before={before.get("spec_kv_cache")} after={after.get("spec_kv_cache")}'
            )

    def _assert_input_buffer_snapshot_unchanged(rank: int, before: dict, after: dict):
        if before != after:
            raise RuntimeError(
                f'Preserved-graph input_buffers changed on rank[{rank}]: '
                f'before={before} after={after}'
            )

    def _format_num_bytes(num_bytes: int) -> str:
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

    def _save_preserved_model_buffers(self, level: int):
        self._ascend_preserve_saved_model_buffers = {}
        if level != 2 or self.patched_model is None:
            return

        model = self.patched_model.get_model()
        saved_buffers = {}
        saved_buffer_details = []
        total_bytes = 0
        for name, buffer in model.named_buffers():
            if not torch.is_tensor(buffer) or buffer.device.type == 'meta':
                continue
            saved_buffers[name] = buffer.detach().cpu().clone()
            num_bytes = buffer.numel() * buffer.element_size()
            total_bytes += num_bytes
            saved_buffer_details.append(
                f'{name}:shape={tuple(buffer.shape)},dtype={buffer.dtype},size={_format_num_bytes(num_bytes)}'
            )

        self._ascend_preserve_saved_model_buffers = saved_buffers
        if saved_buffers:
            logger.info(
                'Ascend preserved graph saved %s model buffers (%s) before level=2 sleep on rank[%s].',
                len(saved_buffers),
                _format_num_bytes(total_bytes),
                self.rank,
            )
            logger.info(
                'Ascend preserved graph saved model buffer details rank[%s]: %s',
                self.rank,
                saved_buffer_details,
            )

    def _restore_preserved_model_buffers(self):
        saved_buffers = getattr(self, '_ascend_preserve_saved_model_buffers', None)
        if not saved_buffers or self.patched_model is None:
            return

        model = self.patched_model.get_model()
        current_buffers = dict(model.named_buffers())
        restored = 0
        skipped = []
        for name, saved_buffer in saved_buffers.items():
            buffer = current_buffers.get(name)
            if buffer is None:
                skipped.append(name)
                continue
            if tuple(buffer.shape) != tuple(saved_buffer.shape) or buffer.dtype != saved_buffer.dtype:
                skipped.append(name)
                continue
            buffer.copy_(saved_buffer.to(device=buffer.device, non_blocking=True))
            restored += 1

        self._ascend_preserve_saved_model_buffers = {}
        logger.info(
            'Ascend preserved graph restored %s/%s model buffers after weight remap on rank[%s]. skipped=%s',
            restored,
            len(saved_buffers),
            self.rank,
            skipped[:8],
        )

    def _reset_preserved_weight_derived_caches(self):
        if self.patched_model is None:
            return
        model = self.patched_model.get_model()
        reset_count = 0
        for _, mod in model.named_modules():
            if hasattr(mod, 'A_log_exp'):
                mod.A_log_exp = None
                reset_count += 1
        if reset_count:
            logger.info(
                'Ascend preserved graph reset %s Qwen3.5 A_log_exp caches on rank[%s].',
                reset_count,
                self.rank,
            )

    def _reset_preserved_runtime_caches(self):
        reset_targets = []
        if self.cache_engine is not None and hasattr(self.cache_engine, 'full_gpu_cache'):
            self.cache_engine.full_gpu_cache.zero_()
            reset_targets.append('kv_cache')

        if self.state_cache_engine is not None:
            state_caches = getattr(self.state_cache_engine, 'state_caches', ())
            for cache in state_caches:
                cache.zero_()
            if len(state_caches) > 0:
                reset_targets.append('state_cache')

        spec_cache_engine = getattr(self.spec_agent, 'cache_engine', None)
        if spec_cache_engine is not None and hasattr(spec_cache_engine, 'full_gpu_cache'):
            spec_cache_engine.full_gpu_cache.zero_()
            reset_targets.append('spec_kv_cache')

        if reset_targets:
            logger.info(
                'Ascend preserved graph cold reset rank[%s]: zeroed_or_reset=%s',
                self.rank,
                ','.join(reset_targets),
            )

    def _build_model(self):
        if not ascend_graph_preserve.enabled():
            return origin_build_model(self)

        ascend_graph_preserve.ensure_available()
        with self.all_context(), ascend_graph_preserve.use_memory_pool('weights'):
            self._build_model()
            self.spec_agent.build_model(
                self.misc_config.empty_init,
                self.patched_model,
                build_model_ctx=self.build_model_ctx,
            )

    def _build_cache_engine(self):
        if not ascend_graph_preserve.enabled():
            return origin_build_cache_engine(self)

        ascend_graph_preserve.ensure_available()
        with self.all_context(), ascend_graph_preserve.use_memory_pool('kv_cache'):
            origin_build_cache_engine(self)
            self._ascend_preserve_ptr_snapshot = _capture_ptr_snapshot(self)
            _log_ptr_snapshot('build-cache', self.rank, self._ascend_preserve_ptr_snapshot)

    def _get_free_mem(self):
        if not ascend_graph_preserve.enabled():
            return origin_get_free_mem(self)

        with self.all_context():
            torch.cuda.empty_cache()
            gpu_mem_physical_free, _ = agent_mod.get_gpu_memory()
            return gpu_mem_physical_free

    @torch.inference_mode()
    def _update_params(self, request):
        ret = origin_update_params(self, request)
        if ascend_graph_preserve.enabled() and getattr(request, 'finished', False):
            with self.all_context():
                _reset_preserved_weight_derived_caches(self)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        return ret

    @torch.inference_mode()
    async def _sleep(self, level: int = 1):
        if not ascend_graph_preserve.enabled():
            return await origin_sleep(self, level)

        ascend_graph_preserve.ensure_available()
        self.state.is_sleeping = True
        if self.dist_config.dp > 1:
            await self.state.to_sleep.wait()
        try:
            with self.all_context():
                _save_preserved_model_buffers(self, level)
                self._ascend_preserve_ptr_snapshot = _capture_ptr_snapshot(self)
                self._ascend_preserve_input_buffer_snapshot = _capture_input_buffer_snapshot(self)
                _log_ptr_snapshot(f'pre-sleep(level={level})', self.rank, self._ascend_preserve_ptr_snapshot)
                _log_input_buffer_snapshot(
                    f'pre-sleep(level={level})',
                    self.rank,
                    self._ascend_preserve_input_buffer_snapshot,
                )
                # Drain in-flight ops referencing pool memory, then unmap. The trailing
                # empty_cache() releases torch's default-pool cache (graph workspace
                # spillover, IPC clone leftovers from update_params, restore-buffer
                # device temps, sampler scratchpads). Without it, fragmentation grows
                # every cycle because PRESERVE_GRAPHS=1 mandates expandable_segments=off.
                torch.cuda.synchronize()
                ascend_graph_preserve.sleep(level)
                torch.cuda.empty_cache()
        finally:
            self.state.to_sleep.clear()

    @torch.inference_mode()
    def _wakeup(self, tags: list[str] | None = None):
        if not ascend_graph_preserve.enabled():
            return origin_wakeup(self, tags)

        ascend_graph_preserve.ensure_available()
        if tags is None:
            tags = ['weights', 'kv_cache']

        with self.all_context():
            before = getattr(self, '_ascend_preserve_ptr_snapshot', None)
            before_input_buffers = getattr(self, '_ascend_preserve_input_buffer_snapshot', None)
            ascend_graph_preserve.wakeup(tags)
            if 'weights' in tags:
                _restore_preserved_model_buffers(self)
                _reset_preserved_weight_derived_caches(self)
            torch.cuda.synchronize()
            if 'kv_cache' in tags:
                self._debug_log_next_real_forward_after_wakeup = True
                _reset_preserved_runtime_caches(self)
                torch.cuda.synchronize()
            after = _capture_ptr_snapshot(self)
            after_input_buffers = _capture_input_buffer_snapshot(self)
            _log_ptr_snapshot(f'post-wakeup(tags={tags})', self.rank, after)
            _log_input_buffer_snapshot(f'post-wakeup(tags={tags})', self.rank, after_input_buffers)
            if before is not None:
                _assert_ptrs_unchanged(self.rank, before, after, tags)
                logger.info(
                    'Ascend preserved graph dataptr check passed on rank[%s] for tags=%s',
                    self.rank,
                    tags,
                )
            if before_input_buffers:
                _assert_input_buffer_snapshot_unchanged(self.rank, before_input_buffers, after_input_buffers)
                logger.info(
                    'Ascend preserved graph input-buffer check passed on rank[%s] for tags=%s',
                    self.rank,
                    tags,
                )
            self._ascend_preserve_ptr_snapshot = after
            self._ascend_preserve_input_buffer_snapshot = after_input_buffers
            # Release torch default-pool cache picked up by the wakeup path:
            # CPU->NPU intermediate tensors from buffer restore, dropped A_log_exp
            # caches, and stale per-iteration scratchpads from the prior rollout.
            # Mirrors the empty_cache that the original sleep() relied on.
            torch.cuda.empty_cache()
        if 'kv_cache' in tags:
            self.state.is_sleeping = False
            if self.dist_config.dp > 1:
                self.state.to_wakeup.set()
            print(f'=====> ModelAgent rank[{self.rank}] wakeup done.')

    def _release(self):
        if not ascend_graph_preserve.enabled():
            return origin_release(self)

        self.reset_graph_runner()
        self.patched_model = None
        self.cache_engine = None
        self.state_cache_engine = None
        self.state.is_sleeping = False

    def _get_pt_weights_iterator(file: str, prefix: str):
        state = torch.load(file, weights_only=True, map_location='cpu')
        try:
            if prefix is None:
                yield from state.items()
            else:
                for k, v in state.items():
                    yield f'{prefix}{k}', v
        finally:
            del state
            if not ascend_graph_preserve.skip_empty_cache():
                torch.cuda.empty_cache()

    agent_mod.BaseModelAgent.build_model = _build_model
    agent_mod.BaseModelAgent.build_cache_engine = _build_cache_engine
    agent_mod.BaseModelAgent.get_free_mem = _get_free_mem
    agent_mod.BaseModelAgent.update_params = _update_params
    agent_mod.BaseModelAgent.sleep = _sleep
    agent_mod.BaseModelAgent.wakeup = _wakeup
    agent_mod.BaseModelAgent.release = _release
    weight_loader_mod._get_pt_weights_iterator = _get_pt_weights_iterator
    agent_mod.BaseModelAgent._dlinfer_ascend_preserve_graph_patched = True


def patch_gated_delta_net():
    import torch.nn.functional as F
    from typing import Any, Sequence, Tuple
    from torch.profiler import record_function

    from lmdeploy.pytorch.nn import gated_delta
    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaMeta

    from dlinfer.vendor.ascend.triton_ops import RMSNormGated
    from dlinfer.vendor.ascend.triton_ops import (
        causal_conv1d_fn,
        causal_conv1d_update_npu,
    )
    from dlinfer.vendor.ascend.triton_ops import (
        chunk_gated_delta_rule,
        fused_sigmoid_gating_delta_rule_update,
    )

    from dlinfer.vendor.ascend.triton_ops.fla import l2norm_fwd

    class AscendGatedDeltaMeta:

        def __init__(
            self,
            num_tokens: int,
            conv_kernel_size: int,
            state_ids: torch.Tensor,
            attn_metadata: Any,
        ):
            self.is_decoding = attn_metadata.is_decoding
            self.cu_seqlens = attn_metadata.q_start_loc

            # state_ids, fill invalid state with 0
            self.state_ids = state_ids.clamp(0)
            self.has_initial_state = attn_metadata.has_initial_state
            self.conv_state_indices = self.state_ids

    def build_rmsnorm_gated(hidden_size: int, eps=1e-6, **kwargs):
        device = kwargs["device"]
        return RMSNormGated(hidden_size, eps=eps, norm_before_gate=True, device=device)

    class AscendCausalConv1dFunc:

        def __init__(self, activation: str = "silu"):
            self.causal_conv1d_fn = causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d_update_npu
            self.activation = activation

        def conv1d_func(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            """
            x: (b, seqlen, dim)
            seqlen: (b)
            out: (b, seqlen, dim)
            conv_state: (b, dim, kernel_size)
            """
            out = self.causal_conv1d_fn(
                x.t(),
                weight,
                bias,
                activation=self.activation,
                conv_states=conv_state.transpose(1, 2),
                has_initial_state=gated_delta_meta.has_initial_state,
                cache_indices=gated_delta_meta.conv_state_indices,
                query_start_loc=gated_delta_meta.cu_seqlens,
            )

            out = out.t().unsqueeze(0)

            return out, conv_state

        # 替换
        def conv1d_update(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            conv_state_indices: torch.Tensor,
        ):
            out = self.causal_conv1d_update(
                x,
                conv_state,
                weight.t().contiguous(),
                bias,
                self.activation,
                conv_state_indices=conv_state_indices,
                validate_data=True,
            )
            return out.unsqueeze(0), conv_state

        @record_function("causal_conv1d")
        def __call__(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            weight_reshaped = weight.squeeze(1)
            x = x.squeeze(0)

            if gated_delta_meta.is_decoding:
                conv_state_indices = gated_delta_meta.conv_state_indices
                return self.conv1d_update(
                    x, weight_reshaped, bias, conv_state, conv_state_indices
                )
            return self.conv1d_func(
                x, weight_reshaped, bias, conv_state, gated_delta_meta=gated_delta_meta
            )

    class AscendGatedDelta:

        def __init__(self, use_qk_l2norm_in_kernel: bool = True):
            self.fused_sigmoid_gating_delta_rule_update = (
                fused_sigmoid_gating_delta_rule_update
            )
            self.chunk_gated_delta_rule = chunk_gated_delta_rule
            self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
            self.l2norm_fwd = l2norm_fwd

        def __call__(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            recurrent_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            """call."""

            is_decoding = gated_delta_meta.is_decoding

            if is_decoding:
                indices = gated_delta_meta.state_ids
                cu_seqlens = gated_delta_meta.cu_seqlens
                core_attn_out = self.fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    q=query.contiguous(),
                    k=key.contiguous(),
                    v=value.contiguous(),
                    a=a.contiguous(),
                    b=b.contiguous(),
                    initial_state_source=recurrent_state,
                    initial_state_indices=indices,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
                last_recurrent_state = None
            else:
                beta = b.sigmoid()
                # If the model is loaded in fp16, without the .float() here, A might be -inf
                g = (-A_log.float().exp()) * F.softplus(a.float() + dt_bias)

                initial_state = recurrent_state[gated_delta_meta.state_ids]
                initial_state[~gated_delta_meta.has_initial_state, ...] = 0
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value.contiguous(),
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=gated_delta_meta.cu_seqlens,
                    head_first=False,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                )
                recurrent_state[gated_delta_meta.state_ids] = last_recurrent_state.to(
                    recurrent_state.dtype
                )

            return core_attn_out, last_recurrent_state

    gated_delta.GatedDeltaMeta = AscendGatedDeltaMeta
    gated_delta.CausalConv1dFunc = AscendCausalConv1dFunc
    gated_delta.GatedDelta = AscendGatedDelta
    gated_delta.build_rmsnorm_gated = build_rmsnorm_gated


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in vendor:
        importlib.import_module(f".{vendor_name_str}", __package__)


def patch_qwen3_5():
    import torch
    from typing import List

    from lmdeploy.utils import is_bf16_supported
    from lmdeploy.pytorch.configurations.default import DefaultModelConfigBuilder
    from lmdeploy.pytorch.configurations.qwen3_next import _check_env_qwen3_next
    from lmdeploy.vl.constants import Modality

    from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader
    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaMeta, CausalConv1d
    from lmdeploy.pytorch.model_inputs import StepContext
    from lmdeploy.pytorch.configurations.qwen3_5 import Qwen3_5ModelConfigBuilder
    from lmdeploy.pytorch.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5GatedDeltaNet,
    )

    @classmethod
    def custom_build(cls, hf_config, model_path: str = None, tp: int = 1, **kwargs):
        """build."""
        text_config = hf_config.text_config
        # propagate quantization_config from top-level hf_config into text_config
        quantization_config = getattr(hf_config, "quantization_config", None)
        if quantization_config is not None and not hasattr(
            text_config, "quantization_config"
        ):
            text_config.quantization_config = quantization_config
        cfg = DefaultModelConfigBuilder.build(text_config, model_path, tp=tp, **kwargs)

        # update num layers
        num_layers = cfg.num_layers
        layer_types = text_config.layer_types
        num_delta_layers = sum([1 for lt in layer_types if lt == "linear_attention"])
        num_full_layers = num_layers - num_delta_layers
        cfg.num_layers = num_full_layers

        # set state shapes
        head_k_dim = text_config.linear_key_head_dim
        head_v_dim = text_config.linear_value_head_dim
        num_v_heads = text_config.linear_num_value_heads // tp
        num_k_heads = text_config.linear_num_key_heads // tp
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        conv_kernel_size = text_config.linear_conv_kernel_dim

        # Ascend Patch
        conv_state_shape = (conv_kernel_size, conv_dim)
        recurrent_state_shape = (num_v_heads, head_k_dim, head_v_dim)

        device_type = kwargs.get("device_type", "cuda")
        if is_bf16_supported(device_type):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # Ascend Patch
        # Use per-layer shapes so each cache slice is (num_caches, ...) — contiguous by
        # construction. Storing num_delta_layers as the first shape dim would require a
        # transpose later, producing non-contiguous views.
        cfg.states_shapes = [(conv_state_shape, dtype)] * num_delta_layers + [
            (recurrent_state_shape, torch.float32)
        ] * num_delta_layers

        cfg.is_gated_delta = True
        cfg.check_env_func = _check_env_qwen3_next

        cfg.use_mrope = True
        return cfg

    def custom_prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # Ascend Patch
        # make past_key_values
        # state_caches holds num_delta_layers conv entries then num_delta_layers recurrent
        # entries, each already shaped (num_caches, ...) and contiguous.
        n = len(context.state_caches) // 2
        state_caches = list(zip(context.state_caches[:n], context.state_caches[n:]))

        past_key_values = list(past_key_values)
        new_past_key_values = []
        for layer_type in self.config.text_config.layer_types:
            if layer_type == "linear_attention":
                new_past_key_values.append(state_caches.pop(0))
            elif layer_type == "full_attention":
                new_past_key_values.append(past_key_values.pop(0))

        # vlm inputs
        pixel_values = None
        vis_cu_seqlens = None
        vis_pos_emb = None
        image_mask = None
        grid_thw = None
        pos_embeds = None
        if context.input_multimodals is not None:
            mm_inputs = [
                input_mm.get("mm_data", []) for input_mm in context.input_multimodals
            ]
            # flatten batch
            mm_inputs = [item for sublist in mm_inputs for item in sublist]

            if len(mm_inputs) > 0:
                modality = mm_inputs[0].modality
                pixel_values = torch.cat([inp.data for inp in mm_inputs])

                image_token_id = mm_inputs[0].meta.get("image_token_id")
                video_token_id = mm_inputs[0].meta.get("video_token_id")
                mm_token_id = (
                    image_token_id if modality == Modality.IMAGE else video_token_id
                )
                image_mask = input_ids == mm_token_id

                grid_thw = torch.cat(
                    [data.meta["grid_thw"] for data in mm_inputs]
                ).cpu()
                vis_pos_emb = self.model.visual.rot_pos_emb(grid_thw)
                pos_embeds = self.model.visual.fast_pos_embed_interpolate(grid_thw)
                vis_cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).to(pixel_values.device)
                vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                vis_pos_emb = vis_pos_emb.repeat(1, 2)
                vis_pos_emb = (vis_pos_emb.cos(), vis_pos_emb.sin())

        mrope_position_ids = getattr(context, "mrope_position_ids", None)

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(
                inputs_embeds
            )

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
            # vl inputs
            mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            image_mask=image_mask,
            grid_thw=grid_thw,
            pos_embeds=pos_embeds,
        )

    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
        gated_delta_meta: GatedDeltaMeta,
    ):
        """forward."""

        # load states
        conv_state, recurrent_state = self._load_state(past_key_value, gated_delta_meta)

        # inputs proj
        projected_states_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        # [..., ng, np/ng * hn] -> [..., np, hn]
        z = z.unflatten(-1, (-1, self.head_v_dim))
        projected_states_ba = self.in_proj_ba(hidden_states)
        b, a = self.fix_ba_ordering(projected_states_ba)

        mixed_qkv = projected_states_qkv
        mixed_qkv, conv_state = self.conv1d(
            mixed_qkv, conv_state, gated_delta_meta=gated_delta_meta
        )

        tp = (self.key_dim * 2 + self.value_dim) // mixed_qkv.size(-1)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // tp,
                self.key_dim // tp,
                self.value_dim // tp,
            ],
            dim=-1,
        )
        query = query.unflatten(-1, (-1, self.head_k_dim))
        key = key.unflatten(-1, (-1, self.head_k_dim))
        value = value.unflatten(-1, (-1, self.head_v_dim))

        # beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        # g = self.get_A_log_exp() * F.softplus(a.float() + self.dt_bias)
        # For decoding, the fused_sigmoid_gating_delta_rule_update kernel handles
        # GQA natively via i_h = i_hv // (HV // H), so repeat_interleave is not needed.
        # For prefill, chunk_gated_delta_rule requires H == HV (no native GQA support).
        if self.kv_ratio > 1 and not gated_delta_meta.is_decoding:
            query = query.repeat_interleave(self.kv_ratio, dim=-2)
            key = key.repeat_interleave(self.kv_ratio, dim=-2)

        core_attn_out, recurrent_state = self.gated_delta(
            query,
            key,
            value,
            self.A_log,
            self.dt_bias,
            a,
            b,
            recurrent_state=recurrent_state,
            gated_delta_meta=gated_delta_meta,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        output = self.out_proj(core_attn_out)
        return output

    def custom_weight_loader(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ):
        """Weight loader."""
        q, k, v = loaded_weight.split(self.split, dim=0)
        q = q.chunk(self.tp, dim=0)[self.rank]
        k = k.chunk(self.tp, dim=0)[self.rank]
        v = v.chunk(self.tp, dim=0)[self.rank]
        loaded_weight = torch.cat([q, k, v], dim=0)
        default_weight_loader(param, loaded_weight)
        

        # logger.error("------------------------- dlinfer patch conv1d update weights")
        #if param is self.weight:
        #    param.data = param.data.transpose(0, 2).contiguous()

    def custom_update_weights(self):
        """Reset derived weight caches after external weight updates."""
        self.A_log_exp = None

    # logger.error("-------------------- dlinfer patch fun")
    # CausalConv1d.weight_loader = custom_weight_loader
    Qwen3_5GatedDeltaNet.forward = custom_forward
    Qwen3_5GatedDeltaNet.update_weights = custom_update_weights
    Qwen3_5ModelConfigBuilder.build = custom_build
    Qwen3_5ForConditionalGeneration.prepare_inputs_for_generation = (
        custom_prepare_inputs_for_generation
    )


def vendor_device_init():
    import_vendor_module(vendor_name)
    patch_compiled_func()
    patch_async_sampling_logits()
    if vendor_name in ["camb", "ascend"]:
        patch_contiguous_cache_engine()
    if vendor_name == "ascend":
        patch_state_cache_engine()
        patch_ascend_preserved_graph_sleep()
        patch_gated_delta_net()
        patch_qwen3_5()


vendor_device_init()
