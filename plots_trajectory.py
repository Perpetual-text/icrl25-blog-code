from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 777

torch.cuda.empty_cache()

model_name_hf = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_name_hf, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_hf)


def token_probs(logits, token_id):
    eps = 1e-12
    stats = {"probs": [], "entropy": [], "varentropy": [], "info_content": []}
    num_gen_tokens = len(logits)

    for i in range(num_gen_tokens):
        out_logits_i = logits[i][0]  # torch.Size([vocab_size])
        prob_dist = F.softmax(out_logits_i, dim=-1)  # torch.Size([vocab_size])
        probs_dist_log = torch.log(prob_dist + eps)  # torch.Size([vocab_size])
        entropy = -torch.sum(prob_dist * probs_dist_log, dim=-1)

        info_content = -torch.log(prob_dist + eps)
        mean_info_content = torch.sum(prob_dist * info_content, dim=-1, keepdim=True)
        varentropy = torch.sum(
            prob_dist * (info_content - mean_info_content) ** 2, dim=-1
        )
        token_prob = prob_dist[token_id]

        token_info_content = info_content[token_id]

        stats["probs"].append(token_prob.item())
        stats["info_content"].append(token_info_content.item())
        stats["entropy"].append(entropy.item())
        stats["varentropy"].append(varentropy.item())

    return stats


def prob_trajectory(
    token_stats, block_size: int = 100, save_path=Path("./results/plots/temp")
):

    num_blocks = int(np.ceil(len(token_stats["probs"]) / block_size))

    block_indices = []
    block_avg_probs = []
    block_avg_info_content = []
    block_avg_entropy = []
    block_avg_varentropy = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, len(token_stats["probs"]))

        block_indices.append(i + 1)  # Block index starting from 1

        block_probs = token_stats["probs"][start_idx:end_idx]
        block_info_content = token_stats["info_content"][start_idx:end_idx]
        block_entropy = token_stats["entropy"][start_idx:end_idx]
        block_varentropy = token_stats["varentropy"][start_idx:end_idx]

        # Average value
        block_avg_probs.append(np.mean(block_probs))
        block_avg_info_content.append(np.mean(block_info_content))
        block_avg_entropy.append(np.mean(block_entropy))
        block_avg_varentropy.append(np.mean(block_varentropy))

    # Plotting the block-wise averages
    sns.set(style="whitegrid", context="paper", font_scale=1.5)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Average Probability of EOS Token per Block
    sns.lineplot(x=block_indices, y=block_avg_probs, marker="o", ax=axs[0, 0])
    axs[0, 0].set_title("Average Probability of EOS Token per Block")
    axs[0, 0].set_xlabel("Block Number")
    axs[0, 0].set_ylabel("Average Probability")

    # Plot 2: Average Information Content of EOS Token per Block
    sns.lineplot(
        x=block_indices,
        y=block_avg_info_content,
        marker="o",
        ax=axs[0, 1],
        color="orange",
    )
    axs[0, 1].set_title("Average Information Content of EOS Token per Block")
    axs[0, 1].set_xlabel("Block Number")
    axs[0, 1].set_ylabel("Average Information Content (nats)")

    # Plot 3: Average Entropy per Block
    sns.lineplot(
        x=block_indices, y=block_avg_entropy, marker="o", ax=axs[1, 0], color="green"
    )
    axs[1, 0].set_title("Average Entropy per Block")
    axs[1, 0].set_xlabel("Block Number")
    axs[1, 0].set_ylabel("Average Entropy (nats)")

    # Plot 4: Average Varentropy per Block
    sns.lineplot(
        x=block_indices, y=block_avg_varentropy, marker="o", ax=axs[1, 1], color="red"
    )
    axs[1, 1].set_title("Average Varentropy per Block")
    axs[1, 1].set_xlabel("Block Number")
    axs[1, 1].set_ylabel("Average Varentropy (nats²)")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(
        0.5,
        0.01,
        f"Block-wise analysis of EOS token statistics, showing trends in probability, information content, entropy, and varentropy across blocks of size {block_size}",
        ha="center",
        va="center",
        fontsize=12,
    )

    # plt.tight_layout()
    plt.savefig(f"./results/plots/eos_token_stats_blockwise_{block_size}.png", dpi=600)


input_text = "Write a book about a dog that can talk and is a detective."

inputs = tokenizer(input_text, return_tensors="pt").to(device)

input_length = inputs["input_ids"].shape[1]

eos_token_id = tokenizer.eos_token_id
eos_token = tokenizer.decode(eos_token_id)


model.generation_config.max_new_tokens = 8192
model.generation_config.max_length = 8192 + input_length


gen_utilities = {
    "return_dict_in_generate": True,
    # "output_scores": True,
    "output_logits": True,
    # "output_hidden_states": True,
    # "output_attentions": True,
}

gen_sampling = {
    "greedy": {"do_sample": False, "temperature": 1.0, "top_k": None, "top_p": None},
    "beam": {"num_beams": 3},
    "top_k": {"do_sample": True, "top_k": 50, "temperature": 1.0},
    "top_p": {"do_sample": True, "top_p": 0.8, "temperature": 0.7},
}


out_hf_gen = model.generate(**inputs, **gen_utilities, **gen_sampling["greedy"])

out_logits = out_hf_gen.logits
sequence = out_hf_gen.sequences

num_gen_tokens = len(sequence[0]) - input_length


token_id = tokenizer.eos_token_id
eos_token_stats = token_probs(out_logits, token_id)


for block_size in [100, 50, 25]:
    prob_trajectory(eos_token_stats, block_size=block_size)
