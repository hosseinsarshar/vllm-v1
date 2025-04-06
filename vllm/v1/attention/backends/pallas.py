# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch
# Required to register custom ops.
import torch_xla.experimental.custom_kernel  # noqa: F401

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
import torch_xla.core.xla_model as xm
import torch.nn.functional as F

# These are the 2 tunable parameters of the paged attention Pallas kernel.
NUM_QUERIES_PER_BLOCK = 32
NUM_KV_PAGES_PER_BLOCK = 128
from vllm.distributed.utils import get_shard_spec, is_spmd, disable_manual_sharding_wrapper, enable_manual_sharding_wrapper, get_device_ids
import torch_xla.debug.profiler as xp
import torch_xla.distributed.spmd as xs

class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> type["PallasMetadata"]:
        return PallasMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads * 2, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")


@dataclass
class PallasMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Used in the PallasAttentionBackendImpl
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_seqs: int


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError("Paged attention Pallas kernel does "
                             "not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if head_size % 128 != 0:
            raise NotImplementedError("Head size must be a multiple of 128.")
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "Attention logits soft-capping is not supported.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        tpu_version = torch_xla.tpu.version()
        if tpu_version < 4:
            raise NotImplementedError("TPU version must be 4 or higher.")
        # NOTE(chengjiyao): the TPU v4's vmem capacity is 16MB
        # TODO(chengjiyao): autotune NUM_QUERIES_PER_BLOCK,
        # NUM_KV_PAGES_PER_BLOCK and vmem_limit_bytes
        if tpu_version == 4:
            self.vmem_limit_bytes = 16 * 1024 * 1024
        else:
            self.vmem_limit_bytes = 64 * 1024 * 1024

    def forward(
        self,
        layer: AttentionLayer,
        query_org: torch.Tensor,
        key_org: torch.Tensor,
        value_org: torch.Tensor,
        kv_cache_org: torch.Tensor,
        attn_metadata: PallasMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print("hosseins: PallasAttentionBackendImpl.forward()")
        
        """Forward pass with Pallas attention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # For determine_available_memory case.
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(query_org)=} {query_org.shape=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(key_org)=} {key_org.shape=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(value_org)=} {value_org.shape=}")

        if kv_cache_org.numel() == 0:
            if output is None:
                output = torch.ones_like(query_org)
            return output
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(kv_cache_org)=} {kv_cache_org.shape=}")

        key = enable_manual_sharding_wrapper(key_org, (None, 'axis'))
        # key = key_org
        query = enable_manual_sharding_wrapper(query_org, (None, 'axis'))
        # query = query_org
        value = enable_manual_sharding_wrapper(value_org, (None, 'axis'))
        # value = value_org
        kv_cache = enable_manual_sharding_wrapper(kv_cache_org, (None, None, 'axis', None))
        # kv_cache = kv_cache_org

        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(query)=} {query.shape=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(key)=} {key.shape=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(value)=} {value.shape=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(kv_cache)=} {kv_cache.shape=}")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens, hidden_size = query_org.shape
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {self.num_heads=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {self.head_size=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {num_tokens=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {hidden_size=}")
        # with xp.Trace("PallasAttentionBackend.forward.view()"):
        query = query.view(num_tokens, max(1, self.num_heads // len(get_device_ids())) if (is_spmd() and True) else self.num_heads, self.head_size)
        # print(f"hosseins: PallasAttentionBackend.forward() after {get_shard_spec(query)=} {query.shape=}")

        # with xp.Trace("PallasAttentionBackend.forward.write_to_kv_cache()"):
        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            write_to_kv_cache(key, value, kv_cache, slot_mapping)

        # with xp.Trace("PallasAttentionBackend.forward.ragged_paged_attention()"):
        output = torch.ops.xla.ragged_paged_attention(
            query,
            kv_cache,
            attn_metadata.context_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
            num_kv_pages_per_block=NUM_KV_PAGES_PER_BLOCK,
            num_queries_per_block=NUM_QUERIES_PER_BLOCK,
            vmem_limit_bytes=self.vmem_limit_bytes,
            use_kernel=True,
            sm_scale=self.scale)
        
        # test_unsharded = torch.zeros((num_tokens, hidden_size // 4), device=output.device, dtype=torch.bfloat16)
        # # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(test_unsharded)=} {test_unsharded.shape=}")
        # test_unsharded_output = F.linear(test_unsharded, test_unsharded)
        # # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(test_unsharded_output)=} {test_unsharded_output.shape=}")
        # test_sharded = disable_manual_sharding_wrapper(test_unsharded, (None, 'axis'), torch.Size((num_tokens, hidden_size)))
        # # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(test_sharded)=} {test_sharded.shape=}")

        # key_man_shard = enable_manual_sharding_wrapper(key_org, (None, 'axis'))
        # # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(key_man_shard)=} {key_man_shard.shape=}")
        # key_sharded = disable_manual_sharding_wrapper(key_man_shard, (None, 'axis'), key_org.shape)
        # # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(key_sharded)=} {key_sharded.shape=}")
        
        # xm.mark_step()

        # # print(f"hosseins: PallasAttentionBackend.forward() 1 {get_shard_spec(test_unsharded)=} {test_unsharded.shape=}")
        # # print(f"hosseins: PallasAttentionBackend.forward() 1 {get_shard_spec(test_sharded)=} {test_sharded.shape=}")

        # print(f"hosseins: PallasAttentionBackend.forward() 1 {get_shard_spec(query_org)=} {query_org.shape=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 1 {get_shard_spec(key_org)=} {key_org.shape=}")
        # print(f"hosseins: PallasAttentionBackend.forward() 1 {get_shard_spec(value_org)=} {value_org.shape=}")

        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(output)=} {output.shape=}")
        output_shape = output.shape
        new_output = disable_manual_sharding_wrapper(output, (None, 'axis', None), torch.Size((output_shape[0], output_shape[1] * 4, output_shape[2])))
        # new_output = output
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(new_output)=} {new_output.shape=}")
        true_out = new_output.reshape(num_tokens, hidden_size)
        # print(f"hosseins: PallasAttentionBackend.forward() 0 {get_shard_spec(true_out)=} {true_out.shape=}")

        return true_out


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """ Write the key and values to the KV cache.

    Args:
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads *  head_size]
        kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]

    """
    _, _, num_combined_kv_heads, head_size = kv_cache.shape
    num_kv_heads = num_combined_kv_heads // 2
    # print(f"hosseins: write_to_kv_cache() 1 {get_shard_spec(key)=} {key.shape=}")
    # print(f"hosseins: write_to_kv_cache() 1 {get_shard_spec(value)=} {value.shape=}")
    # print(f"hosseins: write_to_kv_cache() 1 {get_shard_spec(kv_cache)=} {kv_cache.shape=}")

    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    # print(f"hosseins: write_to_kv_cache() 2 {get_shard_spec(key)=} {value.shape=}")
    # print(f"hosseins: write_to_kv_cache() 2 {get_shard_spec(value)=} {value.shape=}")


    kv = torch.cat([key, value], axis=-1).reshape(-1, num_combined_kv_heads,
                                                  head_size)

    # print(f"hosseins: write_to_kv_cache() 3 {get_shard_spec(kv)=} {kv.shape=}")

    torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)


    # # print(f"hosseins: write_to_kv_cache() {key_cache.shape=}")
    # # print(f"hosseins: write_to_kv_cache() {get_shard_spec(key_cache)=} {key_cache.shape=}")
    # # print(f"hosseins: write_to_kv_cache() {value_cache.shape=}")
    # # print(f"hosseins: write_to_kv_cache() {get_shard_spec(value_cache)=} {value_cache.shape=}")
    # print(f"hosseins: write_to_kv_cache() {slot_mapping.shape=}")
    # print(f"hosseins: write_to_kv_cache() {get_shard_spec(slot_mapping)=}")


    kv_cache = kv_cache.flatten(0, 1)

    # print(f"hosseins: write_to_kv_cache() 4 {get_shard_spec(kv_cache)=} {kv_cache.shape=}")

    kv_cache.index_copy_(0, slot_mapping, kv)

    # print(f"hosseins: write_to_kv_cache() 5 {get_shard_spec(kv_cache)=} {kv_cache.shape=}")

