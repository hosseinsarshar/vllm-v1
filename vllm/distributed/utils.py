# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import datetime
import pickle
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import torch
from torch.distributed import ProcessGroup, TCPStore
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                _unregister_process_group,
                                                is_nccl_available)
from torch.distributed.rendezvous import rendezvous

import vllm.envs as envs
from vllm.logger import init_logger
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.spmd.xla_sharding as xla_sharding

from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
import torch_xla.core.xla_model as xm
import torch_xla
from torch_xla.distributed.spmd import XLAShardedTensor, Mesh
from typing import Tuple, Union

import re
import ast
import os

logger = init_logger(__name__)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def get_pp_indices(num_hidden_layers: int, pp_rank: int,
                   pp_size: int) -> Tuple[int, int]:
    """Try to evenly distribute layers across partitions.

    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all but the last
    partition. The last partition is excluded because it often contains an
    additional norm layer and we are attempting to balance compute.

    If `pp_size > 2` and the number of remaining layers is
    `0 < x <= pp_size - 2` then the remaining layers are evenly distributed
    across the middle partitions. The first and last partitions are excluded
    because they contain the input and output embeddings respectively and we
    are attempting to reduce maximum memory consumption across partitions.
    """
    partition_list_str = envs.VLLM_PP_LAYER_PARTITION
    if partition_list_str is not None:
        try:
            partitions = [
                int(layer) for layer in partition_list_str.split(",")
            ]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(
                partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(
                f"{sum(partitions)=} does not match {num_hidden_layers=}.")
    else:
        layers_per_partition = num_hidden_layers // pp_size
        partitions = [layers_per_partition for _ in range(pp_size)]

        if remaining_layers := num_hidden_layers % pp_size:
            for i in range(2, remaining_layers + 2):
                partitions[-i] += 1
            logger.info("Hidden layers were unevenly partitioned: %s",
                        ",".join(str(p) for p in partitions))
            logger.info("This can be manually overridden using the "
                        "VLLM_PP_LAYER_PARTITION environment variable")

    start_layer = sum(partitions[:pp_rank])
    end_layer = start_layer + partitions[pp_rank]

    return (start_layer, end_layer)


@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store, and the rank, world_size of the
    group. Only use it to communicate metadata between processes.
    For data-plane communication, create NCCL-related objects.
    """
    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store
    data_expiration_seconds: int = 3600  # 1 hour

    # dst rank -> counter
    send_dst_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: Dict[int, int] = dataclasses.field(
        default_factory=dict)

    # A deque to store the data entries, with key and timestamp.
    entries: Deque[Tuple[str,
                         float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {
            i: 0
            for i in range(self.world_size)
        }

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank."""
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def expire_data(self):
        """Expire data that is older than `data_expiration_seconds` seconds."""
        while self.entries:
            # check the oldest entry
            key, timestamp = self.entries[0]
            if time.time() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(
                f"send_to/{self.rank}/{self.recv_src_counter[src]}"))
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Optional[Any], src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.expire_data()
            key = (f"broadcast_from/{src}/"
                   f"{self.broadcast_send_counter}")
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return obj
        else:
            key = (f"broadcast_from/{src}/"
                   f"{self.broadcast_recv_src_counter[src]}")
            recv_obj = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def barrier(self):
        """A barrier to synchronize all ranks."""
        for i in range(self.world_size):
            self.broadcast_obj(None, src=i)

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
        store_timeout: int = 300,
    ) -> "StatelessProcessGroup":
        """A replacement for `torch.distributed.init_process_group` that does not
        pollute the global state.

        If we have process A and process B called `torch.distributed.init_process_group`
        to form a group, and then we want to form another group with process A, B, C,
        D, it is not possible in PyTorch, because process A and process B have already
        formed a group, and process C and process D cannot join that group. This
        function is a workaround for this issue.

        `torch.distributed.init_process_group` is a global call, while this function
        is a stateless call. It will return a `StatelessProcessGroup` object that can be
        used for exchanging metadata. With this function, process A and process B
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """ # noqa
        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=datetime.timedelta(seconds=store_timeout),
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds)


def stateless_init_torch_distributed_process_group(
        host: str, port: int, rank: int, world_size: int,
        backend: str) -> ProcessGroup:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.

    This function is useful when we are not sure about the total number of
    processes in the process group. For example, we may have process
    1, 2, ..., 8 who want to communicate, and process 9 might be the same
    process as process 1, or it might be a different process; process 10
    might be the same process as process 5, or it might be a different process.
    In this case, how can we reliably form a communication channel within
    process 9 and 10, without affecting the communication channel within
    process 1, 2, ..., 8?

    One possible solution is to figure out if process 9 and 10 are the same
    as process 1 and 5 beforehand, and then form a communication channel
    based on the information, adjusting the ranks and world_size etc. However,
    figuring out the information is not always easy, and it will interfere
    with the main communication channel.

    Our solution is to always form a communication channel with process 1, 2,
    ..., 8, and then use this function to form another communication channel
    with process 9 and 10. This way, regardless of whether process 9 and 10
    are the same as process 1 and 5, the main communication channel is
    always formed with process 1, 2, ..., 8, and the additional communication
    channel is formed with process 9 and 10.
    """
    init_method = f"tcp://{host}:{port}"
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )

    if backend == "gloo":
        from torch.distributed.distributed_c10d import ProcessGroupGloo
        backend_class = ProcessGroupGloo(prefix_store,
                                         group_rank,
                                         group_size,
                                         timeout=timeout)
        backend_type = ProcessGroup.BackendType.GLOO
        device = torch.device("cpu")
    elif backend == "nccl":
        assert is_nccl_available()
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")
    else:
        raise RuntimeError(f"Unsupported torch distributed backend: {backend}")

    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)

    return pg


def stateless_destroy_torch_distributed_process_group(
        pg: ProcessGroup) -> None:
    """
    Destroy ProcessGroup returned by
        stateless_init_torch_distributed_process_group().
    """
    # Lazy import for non-CUDA backends.
    from torch.distributed.distributed_c10d import _shutdown_backend
    _shutdown_backend(pg)
    _unregister_process_group(pg.group_name)

def initialize_spmd():
    global _mesh, _device_ids
    if not is_spmd(): 
        return None
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh
    import numpy as np

    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, )
    # logger.info(f"hosseins: mesh_shape: [{mesh_shape=}]")
    _device_ids = np.array(range(num_devices))
    _mesh = Mesh(_device_ids, mesh_shape, ('axis', ))
    
    logger.info(f'Initializing SPMD engine with mesh=[{_mesh}]')
    return _mesh


def get_mesh():
    if not is_spmd(): 
        return None
    global _mesh
    if _mesh is None:
        logger.info('hosseins: creating mesh')
        _mesh = initialize_spmd()
    else:
        # logger.info('hosseins: returning mesh')
        return _mesh

_mesh = None
_device_ids = list(range(0, 4))

def get_device_ids():
    return _device_ids

def get_col_parallel_partition_spec():
    return ('axis', None)
    return (None, 'axis')

def get_row_parallel_partition_spec():
    return (None, 'axis')
    return ('axis', None)

def shard_spmd(data, mesh=None, partition_spec=None, show_visual=False, print_shard=False):
    if not is_spmd(): 
        return None
    assert isinstance(data, torch.Tensor), "Object is not an torch.Tensor"
    if mesh is None:
        mesh = _mesh

    xs.mark_sharding(data, mesh, partition_spec)
    xm.mark_step()
    # time.sleep(1)
    # # logger.info(f"hosseins: shard_spmd() -> [{type(data)=}]")
    torch._sync(data)

    if show_visual:
        # logger.info("hosseins: after sharding param")
        visualize_tensor_sharding(data, use_color=False)

    if print_shard:
        sharding = torch_xla._XLAC._get_xla_sharding_spec(data)
        logger.info(f"hosseins: shard_spmd() -> [{sharding=}]")


def get_shard_spec(tensor, show_visual=False):
    # # logger.info(f"hosseins: get_shard_spec() -> [{type(tensor)=}]")
    if not is_spmd(): 
        return None
    
    # xm.mark_step()
    sharding = torch_xla._XLAC._get_xla_sharding_spec(tensor)
    if show_visual:
        # logger.info("hosseins: after sharding param")
        visualize_tensor_sharding(tensor, use_color=False)
        
    return sharding

from torch.library import impl, custom_op


# @custom_op("xla::_spmd_full_to_shard_shape", mutates_args=())
# def _spmd_full_to_shard_shape(t: torch.Tensor) -> torch.Tensor:
#     return torch_xla._XLAC._spmd_full_to_shard_shape(t)
# 
# @_spmd_full_to_shard_shape.register_fake
# def _(t: torch.Tensor) -> torch.Tensor:
#   return torch.empty_like(t)


def enable_man_sharding(t) -> XLAShardedTensor:
    if not is_spmd(): 
        return None
    t = _spmd_full_to_shard_shape(unwrap_sharded_tensor(t))
    return t


def unwrap_sharded_tensor(
    t: Union[torch.Tensor, XLAShardedTensor]) -> torch.Tensor:
  if isinstance(t, XLAShardedTensor):
    return t.global_tensor
  return t


def wrap_as_sharded_tensor(
    t: Union[torch.Tensor, XLAShardedTensor]) -> XLAShardedTensor:
  if not isinstance(t, XLAShardedTensor):
    return XLAShardedTensor(t)
  return t

# @custom_op("xla::_spmd_full_to_shard_shape", mutates_args=())
def enable_manual_sharding(t: Union[torch.Tensor, XLAShardedTensor],
                           partition_spec: Tuple[Union[Tuple, int, str, None]],
                           mesh: Mesh = None) -> XLAShardedTensor:
  """
  This API enables manual sharding for the given tensor. Manual sharding disables SPMD sharding proporgation and auto
  partition for the given tensor and all subsequential tensors that produced by an op that uses the given tensor as
  input, and therefore allows the user to manually call collectives for the tensor and subsequential tensors. It
  requires the user to provide the partition spec to shard the tensor before enabling the manual sharding. To be noted,
  the leaf tensors need to pass to disable_manual_sharding before ending the graph.
  """
  mesh = get_global_mesh() if mesh is None else mesh
  t = xs.mark_sharding(unwrap_sharded_tensor(t), mesh, partition_spec)
  t = torch_xla._XLAC._spmd_full_to_shard_shape(unwrap_sharded_tensor(t))
  return wrap_as_sharded_tensor(t)

# @custom_op("xla::_spmd_full_to_shard_shape", mutates_args=())
# def _spmd_full_to_shard_shape(t: torch.Tensor) -> torch.Tensor:
#     return torch_xla._XLAC._spmd_full_to_shard_shape(t)
# 
# @_spmd_full_to_shard_shape.register_fake
# def _(t: torch.Tensor) -> torch.Tensor:
#   return torch.empty_like(t)

PartitionSpec = tuple[Union[tuple[Union[int, str], ...], int, str, None], ...]

# from torch_xla._XLAC import _spmd_full_to_shard_shape
allowed_spmd_full_to_shard_shape = torch.compiler.allow_in_graph(torch_xla._XLAC._spmd_full_to_shard_shape)

from typing import Optional, List, Tuple

@custom_op("xla::enable_manual_sharding_wrapper", mutates_args=())
def enable_manual_sharding_wrapper(tensor: torch.Tensor,
                           partition_spec_str: str
) -> torch.Tensor:
  if not is_spmd(): 
        return tensor
  if partition_spec_str is None:
      raise Exception("partition_spec is None 1")
  partition_spec = eval(partition_spec_str)
  # t = torch_xla._XLAC._spmd_full_to_shard_shape(unwrap_sharded_tensor(tensor))
  # return wrap_as_sharded_tensor(t)
  return xs.enable_manual_sharding(tensor, partition_spec=partition_spec, mesh=get_mesh()).global_tensor

@enable_manual_sharding_wrapper.register_fake
def enable_manual_sharding_wrapper_fake(tensor: torch.Tensor, partition_spec_str: str):
    partition_spec = eval(partition_spec_str)
    assert len(tensor.shape) == len(partition_spec), f"tensor and partition_spec lengths dont match - {tensor.shape=} {partition_spec_str=} {len(partition_spec)=}"

    ret_shape = tuple([(x if partition_spec[i] is None else x // len(get_device_ids())) for i, x in enumerate(tensor.shape)])
    tensor = torch.empty(ret_shape, dtype=tensor.dtype, device=tensor.device)

    return tensor
    
@custom_op("xla::disable_manual_sharding_wrapper", mutates_args=())
def disable_manual_sharding_wrapper(tensor: torch.Tensor, partition_spec_str: str, full_shape: List[int]) -> torch.Tensor:
  if not is_spmd(): 
        return tensor
  
  partition_spec = eval(partition_spec_str)
  return xs.disable_manual_sharding(tensor, partition_spec=partition_spec, 
                                    full_shape=tuple(full_shape), mesh=get_mesh()).global_tensor


@disable_manual_sharding_wrapper.register_fake
def disable_manual_sharding_wrapper_fake(tensor: torch.Tensor, partition_spec_str: str, full_shape: List[int]):
    tensor = torch.empty(full_shape, dtype=tensor.dtype, device=tensor.device)

    return tensor


def get_partition_spec(t):
    if not is_spmd(): 
        return None
    
    shard_spec = get_shard_spec(t)
    # logger.info(f"hosseins: get_partition_spec() -> [{shard_spec=}]")
    match = re.search(r"\[([^\]]+)\]", shard_spec)
    # # logger.info(f"hosseins: get_partition_spec() -> [{match=}]")

    if not match:
        return None

    shard_map = match.group(1)
    # # logger.info(f"hosseins: get_partition_spec() -> [{shard_map=}]")

    shard_map_list = ast.literal_eval(f"[{shard_map}]")
    # # logger.info(f"hosseins: get_partition_spec() -> [{shard_map_list=}]")
    return_val = ()

    if len(shard_map_list) == 0:
        return_val = ()
    
    return_val = tuple([None if x == 1 else 'axis' for x in shard_map_list])
    # # logger.info(f"hosseins: get_partition_spec() -> [{return_val=}]")

    return return_val

def is_spmd():
    if "USE_SPMD" in os.environ and os.environ['USE_SPMD'] == "1":
        return True
    else:
        return False
    
def get_torch_tensor_gbytes(tensor: torch.Tensor) -> int:
    """
    Calculates the total memory in bytes used by the data buffer of a
    PyTorch tensor on its assigned device (CPU or GPU).

    The calculation is: number_of_elements * bytes_per_element.

    Args:
        tensor: The input PyTorch tensor (torch.Tensor).

    Returns:
        int: The total memory in bytes occupied by the tensor's data buffer.

    Raises:
        TypeError: If the input is not a torch.Tensor.
    """
    # 1. Validate input type
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, but received type {type(tensor)}")

    # 2. Get the total number of elements in the tensor
    num_elements = tensor.numel()

    # 3. Get the size (in bytes) of a single element based on the tensor's data type
    bytes_per_element = tensor.element_size()

    # 4. Calculate total bytes
    total_bytes = num_elements * bytes_per_element

    return total_bytes / (10**9)
