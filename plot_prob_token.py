import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 777

torch.cuda.empty_cache()

model_name_hf = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_name_hf, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_hf)


def plot_dist_pdf(tokenizer, logits, save_path, top_k=40, title=""):
    """
    Plot the distribution of token probabilities for the top `k` tokens at each generation step

    Args:
        logits (Tuple[torch.Tensor]): Tuple of tensors, each of shape (batch_size, vocab_size)


    """
    sns.set_theme()

    probabilities = [
        torch.softmax(score, dim=-1) for score in logits
    ]  # List of tensors, each of shape (batch_size, vocab_size)

    generated_text = []

    plt.rcParams["text.usetex"] = False

    with PdfPages(save_path) as pdf:
        for i, probs in enumerate(probabilities):
            y = (
                probs[0].detach().cpu().numpy()
            )  # Choosing the first sequence (batch_size=1)
            sorted_indices = np.argsort(y)[::-1][:top_k]
            sorted_probs = y[sorted_indices]

            tokens_top_5 = [tokenizer.decode([idx]) for idx in sorted_indices[:5]]

            tokens_top_5_text = ", ".join(tokens_top_5)

            generated_text.append(tokens_top_5[0])

            y_plot = sorted_probs
            x_plot = np.arange(top_k)

            plt.figure(figsize=(10, 6))
            plt.rcParams["text.usetex"] = False

            plt.plot(x_plot, y_plot, label="Top 5 words: " + tokens_top_5_text)
            # plt.xlabel("Token Index")
            plt.ylabel("Probability")
            plt.title(f"{i}: {generated_text[-10:]}")
            plt.legend()

            plt.xticks(
                x_plot,
                [
                    tokenizer.decode([idx]).replace("$", "d_sign")
                    for idx in sorted_indices
                ],
                rotation=90,
                family=["DejaVu Sans"],
            )
            # plt.xticks(x_plot)  # Displays all indices in x_plot

            plt.tight_layout()
            pdf.savefig()
            plt.close()


def token_probs(logits, token_id):

    probs = []
    num_gen_tokens = len(logits)

    for i in range(num_gen_tokens):
        out_logits_i = logits[i][0]  # torch.Size([vocab_size])
        prob_dist = F.softmax(out_logits_i, dim=-1)
        token_prob = prob_dist[token_id]
        probs.append(token_prob.item())

    return probs


# input_text = "Write a book about a dog that can talk and is a detective."
input_text = "Write a very long story about a dragon and a knight."


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


eos_probs = token_probs(out_logits, eos_token_id)


def token_prob_threshold(probs, threshold=0.0001):
    # return idxs and probs that are greater than the threshold
    idxs = [i for i, prob in enumerate(probs) if prob > threshold]
    return idxs, [probs[i] for i in idxs]


idxs_greater_than_threshold, _ = token_prob_threshold(eos_probs)

logits_selected = [out_logits[i] for i in idxs_greater_than_threshold]
for token_gen_idx in idxs_greater_than_threshold:
    print(f"Iteration: {token_gen_idx}, EOS prob: {eos_probs[token_gen_idx]}")

plot_dist_pdf(
    tokenizer,
    logits_selected,
    "./results/plots/prob_dist_eos.pdf",
    top_k=40,
    title="NEAR EOS",
)


def plot_dist_png(tokenizer, logits, save_dir, top_k=40, title=""):
    """
    Plot the distribution of token probabilities for the top `k` tokens at each generation step
    and save each plot as a high-quality PNG file.

    Args:
        logits (Tuple[torch.Tensor]): Tuple of tensors, each of shape (batch_size, vocab_size)
        save_dir (str): Directory to save the PNG files.
    """

    sns.set_theme()

    probabilities = [
        torch.softmax(score, dim=-1) for score in logits
    ]  # size = (batch_size, vocab_size)

    generated_text = []

    plt.rcParams["text.usetex"] = False

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for i, probs in enumerate(probabilities):
        y = probs[0].detach().cpu().numpy()
        sorted_indices = np.argsort(y)[::-1][:top_k]
        sorted_probs = y[sorted_indices]

        tokens_top_5 = [tokenizer.decode([idx]) for idx in sorted_indices[:5]]

        tokens_top_5_text = ", ".join(tokens_top_5)

        generated_text.append(tokens_top_5[0])

        y_plot = sorted_probs
        x_plot = np.arange(top_k)

        plt.figure(figsize=(10, 6))
        plt.rcParams["text.usetex"] = False

        plt.plot(x_plot, y_plot, label="Top 5 words: " + tokens_top_5_text)
        plt.ylabel("Probability")
        plt.title(f"{i}: {generated_text[-10:]}")
        plt.legend()

        plt.xticks(
            x_plot,
            [tokenizer.decode([idx]).replace("$", "d_sign") for idx in sorted_indices],
            rotation=90,
            family=["DejaVu Sans"],
        )

        plt.tight_layout()
        plt.savefig(f"{save_dir}/prob_dist_{i}.png", dpi=300)
        plt.close()


plot_dist_png(
    tokenizer,
    logits_selected,
    save_dir="./results/plots/png",  # Directory to save PNG files
    top_k=40,
    title="NEAR EOS",
)
