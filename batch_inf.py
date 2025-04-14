from vllm import LLM, SamplingParams


# Create an LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          gpu_memory_utilization=.95,
          max_model_len=512,
          tensor_parallel_size=1,
          download_dir="/dev/shm"
          )


# Provide prompts
prompts = ["Here are some tips for taking care of your skin: ",
           "To cook this recipe, we'll need "]


# adjust sampling parameters as necessary for the task
sampling_params = SamplingParams(temperature=0.2,
                                 max_tokens=100,
                                 min_p=0.15,
                                 top_p=0.85)


# Generate texts from the prompts
outputs = llm.generate(prompts, sampling_params)

breakpoint()
# Print outputs
for i, o in zip(prompts, outputs):
    print(f"Prompt: {i}")
    print(f"Output: {o.outputs[0].text}")
    print("-"*30)
