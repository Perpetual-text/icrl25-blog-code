from typing import Literal

import torch
import torch.nn.functional as F


def sampling(
    logits: torch.Tensor,
    method: Literal["greedy", "top_k", "top_p"] = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Sampling from the logits to predict the next token.

    Args:
        logits (torch.Tensor): The logits from the model. size = (batch_size, seq_len, vocab_size)
        method (Literal["greedy", "top_k", "top_p"], optional): The sampling method. Defaults to "greedy".
        temperature (float, optional): The temperature for the sampling. Defaults to 1.0.
        top_k (int, optional): The top k for the top_k sampling. Defaults to 50.
        top_p (float, optional): The top p for the top_p sampling. Defaults to 0.9.

    Returns:
        torch.Tensor: The sampled token. size = (batch_size, 1)
    """

    next_logits = logits[:, -1, :]  # size = (batch_size, vocab_size)

    if method == "greedy":
        _, next_token = next_logits.max(dim=-1, keepdim=True)

    else:
        next_logits /= temperature  # default temperature = 1.0
        next_probs = F.softmax(next_logits, dim=-1)

        if method == "top_k":
            probs, probs_indices = next_probs.topk(k=top_k, dim=-1)

        elif method == "top_p":

            probs, probs_indices = next_probs.sort(descending=True, dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)
            mask = cumulative_probs - probs > top_p
            probs[mask] = 0.0
            probs /= probs.sum(dim=-1, keepdim=True)

        else:
            raise ValueError(
                "Invalid method. Choose from 'greedy', 'top_k', or 'top_p'."
            )

        idx_sample = torch.multinomial(input=probs, num_samples=1)

        next_token = torch.gather(input=probs_indices, dim=-1, index=idx_sample)

    return next_token


def sampling_tokens(
    logits: torch.Tensor,
    method: Literal["greedy", "top_k", "top_p"] = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    num_return_tokens: int = 1,
) -> torch.Tensor:
    """
    Sampling from the logits to predict more than one token.

    Args:
        logits (torch.Tensor): The logits from the model. Size = (batch_size, seq_len, vocab_size)
        method (Literal["greedy", "top_k", "top_p"], optional): The sampling method. Defaults to "greedy".
        temperature (float, optional): The temperature for sampling. Defaults to 1.0.
        top_k (int, optional): The top k for top_k sampling. Defaults to 50.
        top_p (float, optional): The top p for top_p sampling. Defaults to 0.9.
        num_return_tokens (int, optional): The number of tokens to return. Defaults to 1.

    Returns:
        torch.Tensor: The sampled tokens. Size = (batch_size, num_return_tokens)

    """

    next_logits = logits[:, -1, :]  # Size = (batch_size, vocab_size)

    if method == "greedy":
        _, next_token = next_logits.topk(k=num_return_tokens, dim=-1)

    else:
        next_logits /= temperature  # default temperature = 1.0
        next_probs = F.softmax(next_logits, dim=-1)

        if method == "top_k":
            probs, probs_indices = next_probs.topk(k=top_k, dim=-1)

        elif method == "top_p":

            probs, probs_indices = next_probs.sort(descending=True, dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)
            mask = cumulative_probs - probs > top_p
            probs[mask] = 0.0
            probs /= probs.sum(dim=-1, keepdim=True)

        else:
            raise ValueError(
                "Invalid method. Choose from 'greedy', 'top_k', or 'top_p'."
            )

        # note: replace=False
        idx_sample = torch.multinomial(
            probs, num_samples=num_return_tokens, replacement=False
        )
        next_token = torch.gather(probs_indices, dim=-1, index=idx_sample)

    return next_token
