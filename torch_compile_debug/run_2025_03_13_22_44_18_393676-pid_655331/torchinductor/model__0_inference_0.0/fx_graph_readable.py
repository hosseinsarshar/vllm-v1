class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "i32[s0]", arg2_1: "Sym(s1)", arg3_1: "Sym(s2)", arg4_1: "Sym(s3)", arg5_1: "Sym(s4)", arg6_1: "Sym(s5)"):
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:146 in get_masked_input_and_mask, code: org_vocab_mask = (input_ >= org_vocab_start_index) & (
        ge: "b8[s0]" = torch.ops.aten.ge.Scalar(arg1_1, arg2_1)
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:147 in get_masked_input_and_mask, code: input_ < org_vocab_end_index)
        lt: "b8[s0]" = torch.ops.aten.lt.Scalar(arg1_1, arg3_1)
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:146 in get_masked_input_and_mask, code: org_vocab_mask = (input_ >= org_vocab_start_index) & (
        bitwise_and: "b8[s0]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:148 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
        ge_1: "b8[s0]" = torch.ops.aten.ge.Scalar(arg1_1, arg4_1)
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:149 in get_masked_input_and_mask, code: input_ < added_vocab_end_index)
        lt_1: "b8[s0]" = torch.ops.aten.lt.Scalar(arg1_1, arg5_1);  arg5_1 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:148 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
        bitwise_and_1: "b8[s0]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:151 in get_masked_input_and_mask, code: org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
        sub_6: "Sym(-s1 + s2)" = arg3_1 - arg2_1;  arg3_1 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:150 in get_masked_input_and_mask, code: added_offset = added_vocab_start_index - (
        sub_7: "Sym(s1 - s2 + s3)" = arg4_1 - sub_6;  arg4_1 = sub_6 = None
        sub_8: "Sym(s1 - s2 + s3 - s5)" = sub_7 - arg6_1;  sub_7 = arg6_1 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:152 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index *
        mul: "i64[s0]" = torch.ops.aten.mul.Tensor(bitwise_and, arg2_1);  arg2_1 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:153 in get_masked_input_and_mask, code: org_vocab_mask) + (added_offset * added_vocab_mask)
        mul_2: "i64[s0]" = torch.ops.aten.mul.Tensor(bitwise_and_1, sub_8);  sub_8 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:152 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index *
        add_16: "i64[s0]" = torch.ops.aten.add.Tensor(mul, mul_2);  mul = mul_2 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:154 in get_masked_input_and_mask, code: vocab_mask = org_vocab_mask | added_vocab_mask
        bitwise_or: "b8[s0]" = torch.ops.aten.bitwise_or.Tensor(bitwise_and, bitwise_and_1);  bitwise_and = bitwise_and_1 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:155 in get_masked_input_and_mask, code: input_ = vocab_mask * (input_ - valid_offset)
        sub_13: "i64[s0]" = torch.ops.aten.sub.Tensor(arg1_1, add_16);  arg1_1 = add_16 = None
        mul_6: "i64[s0]" = torch.ops.aten.mul.Tensor(bitwise_or, sub_13);  sub_13 = None
        
         # File: /home/hosseins/vllm-v1/vllm/model_executor/layers/vocab_parallel_embedding.py:156 in get_masked_input_and_mask, code: return input_, ~vocab_mask
        bitwise_not: "b8[s0]" = torch.ops.aten.bitwise_not.default(bitwise_or);  bitwise_or = None
        return (mul_6, bitwise_not)
        