import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation import (
    long_generate_approach1,
    long_generate_approach2,
    long_generate_approach3,
    long_generate_approach4,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 777

torch.cuda.empty_cache()

model_name_hf = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_name_hf, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_hf)


input_text = "Write a book about a dog that can talk and is a detective."

input = tokenizer(input_text, return_tensors="pt").to(model.device)


sampling_params = {
    "greedy": {"method": "greedy"},
    "top_k": {"method": "top_k", "temperature": 1.0, "top_k": 20},
    "top_p": {"method": "top_p", "temperature": 0.8, "top_p": 0.7},
}


gen = long_generate_approach1(
    model=model,
    inputs=input,
    max_new_tokens=8000,
    eos_token_id=tokenizer.eos_token_id,
    sampling_params=sampling_params["top_k"],
)

with open(f"results/approach{1}_txt", "w") as f:
    f.write(tokenizer.decode(gen.squeeze().tolist()))


gen = long_generate_approach2(
    model=model,
    inputs=input,
    max_new_tokens=8000,
    eos_token_id=tokenizer.eos_token_id,
    sampling_params=sampling_params["top_k"],
)

with open(f"results/approach{2}.txt", "w") as f:
    f.write(tokenizer.decode(gen.squeeze().tolist()))


input_ids = input["input_ids"]

gen = long_generate_approach3(
    model=model,
    tokenizer=tokenizer,
    input_ids=input_ids,
    max_new_tokens=2000,
    eos_token_id=tokenizer.eos_token_id,
    sampling_params=sampling_params["top_k"],
    # sampling_params={"method": "greedy"},
    n_remove=10,
)


input_ids = input["input_ids"]

gen = long_generate_approach4(
    model=model,
    tokenizer=tokenizer,
    input_ids=input_ids,
    max_new_tokens=2000,
    eos_token_id=tokenizer.eos_token_id,
    sampling_params=sampling_params["top_k"],
    # sampling_params={"method": "greedy"},
    n_remove=10,
)
