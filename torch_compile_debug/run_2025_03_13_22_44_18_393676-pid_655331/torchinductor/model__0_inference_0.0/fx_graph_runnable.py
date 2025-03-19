
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/dev/shm/ray/torchinductor_hosseins'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = '+dynamo'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.specialize_float = True
torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.compile_threads = 1
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.view_replay_for_aliased_outputs = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.0
# torch cuda version: None
# torch git version: 79aa17489c3fc5ed6d5e972e9ffddf73e6dd0a5c


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
        ge = torch.ops.aten.ge.Scalar(arg1_1, arg2_1)
        lt = torch.ops.aten.lt.Scalar(arg1_1, arg3_1)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ge_1 = torch.ops.aten.ge.Scalar(arg1_1, arg4_1)
        lt_1 = torch.ops.aten.lt.Scalar(arg1_1, arg5_1);  arg5_1 = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        sub_6 = arg3_1 - arg2_1;  arg3_1 = None
        sub_7 = arg4_1 - sub_6;  arg4_1 = sub_6 = None
        sub_8 = sub_7 - arg6_1;  sub_7 = arg6_1 = None
        mul = torch.ops.aten.mul.Tensor(bitwise_and, arg2_1);  arg2_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(bitwise_and_1, sub_8);  sub_8 = None
        add_16 = torch.ops.aten.add.Tensor(mul, mul_2);  mul = mul_2 = None
        bitwise_or = torch.ops.aten.bitwise_or.Tensor(bitwise_and, bitwise_and_1);  bitwise_and = bitwise_and_1 = None
        sub_13 = torch.ops.aten.sub.Tensor(arg1_1, add_16);  arg1_1 = add_16 = None
        mul_6 = torch.ops.aten.mul.Tensor(bitwise_or, sub_13);  sub_13 = None
        bitwise_not = torch.ops.aten.bitwise_not.default(bitwise_or);  bitwise_or = None
        return (mul_6, bitwise_not)
        
def load_args(reader):
    reader.symint(8192)  # arg0_1
    buf0 = reader.storage(None, 4*s0, device=device(type='xla', index=0), dtype_hint=torch.int32)
    reader.tensor(buf0, (s0,), dtype=torch.int32, is_leaf=True)  # arg1_1
    reader.symint(64128)  # arg2_1
    reader.symint(96192)  # arg3_1
    reader.symint(128256)  # arg4_1
    reader.symint(128256)  # arg5_1
    reader.symint(0)  # arg6_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)