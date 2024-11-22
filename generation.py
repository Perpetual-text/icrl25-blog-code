from datetime import datetime

import torch
from transformers import DynamicCache

from sampling import sampling, sampling_tokens

now = lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# First Approach


@torch.inference_mode()
def long_generate_approach1(
    model, inputs, max_new_tokens, sampling_params, eos_token_id
):
    """
    1. Suppressing the EOS Token
    The first method involves suppressing the EOS token during the token generation process to prevent the model from ending the sequence prematurely.

    """

    past_key_values = DynamicCache()

    cache_position = torch.arange(
        inputs.input_ids.shape[1], dtype=torch.int64, device=model.device
    )

    generated_ids = inputs.input_ids

    eos_happened = []

    try:
        for token_gen_id in range(max_new_tokens):

            outputs = model(
                **inputs,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_ids = sampling_tokens(
                outputs.logits, **sampling_params, num_return_tokens=2
            )

            first_token, second_token = next_token_ids[:, 0], next_token_ids[:, 1]

            next_token_ids = first_token.unsqueeze(0)

            if first_token.item() == eos_token_id:
                # print("EOS token found")
                eos_happened.append(token_gen_id)

                next_token_ids = second_token.unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            cache_position = cache_position[-1:] + 1
            inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

        print(f"EOS token found at: {eos_happened}")

    except RuntimeError as e:
        print(f"Error: {e}")

    return generated_ids


# Second Approach


@torch.inference_mode()
def long_generate_approach2(
    model, inputs, max_new_tokens, sampling_params, eos_token_id
):
    """
    2. Modified Sampling Method Post-EOS Token
    The second method modifies the sampling strategy after the model predicts the EOS token to encourage more diverse continuations and avoid abrupt endings.
    """

    past_key_values = DynamicCache()

    cache_position = torch.arange(
        inputs.input_ids.shape[1], dtype=torch.int64, device=model.device
    )

    default_temp = sampling_params["temperature"]

    generated_ids = inputs.input_ids

    eos_happened = []

    try:
        for token_gen_id in range(max_new_tokens):

            outputs = model(
                **inputs,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if sampling_params["temperature"] > default_temp:
                sampling_params["temperature"] -= 0.1

            next_token_ids = sampling_tokens(
                outputs.logits, **sampling_params, num_return_tokens=2
            )

            first_token, second_token = next_token_ids[:, 0], next_token_ids[:, 1]

            next_token_ids = first_token.unsqueeze(0)

            if first_token.item() == eos_token_id:
                # print("EOS token found")
                eos_happened.append(token_gen_id)

                sampling_params["temperature"] = 2.0

                next_token_ids = second_token.unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            cache_position = cache_position[-1:] + 1
            inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

        print(f"EOS token found at: {eos_happened}")

    except RuntimeError as e:
        print(f"Error: {e}")

    return generated_ids


# Third Approach


@torch.inference_mode()
def long_generate_approach3(
    model,
    tokenizer,
    input_ids,
    max_new_tokens,
    eos_token_id,
    sampling_params,
    n_remove: int,
):
    """
    3. Regenerating Tokens Prior to the EOS Token
    The third method involves regenerating a portion of the sequence preceding the EOS token to provide the model with an opportunity to produce alternative continuations.

    """

    num_layers = model.config.num_hidden_layers  # being used in the cache

    attention_mask = torch.ones_like(input_ids).to(model.device)

    cache_position = torch.arange(
        input_ids.shape[1], dtype=torch.int64, device=model.device
    )

    generated_ids = input_ids

    next_token_id = input_ids

    kv_cache = None

    eos_happened = []

    for gen_token_id in range(max_new_tokens):

        outputs = model(
            input_ids=next_token_id,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=kv_cache,
            use_cache=True,
        )

        next_token_id = sampling(outputs.logits, **sampling_params)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
        )

        cache_position = cache_position[-1:] + 1

        kv_cache = outputs.past_key_values

        if next_token_id.squeeze().item() == eos_token_id:

            save_out_text = f"results/inter_static/{gen_token_id}_{now()}.txt"

            with open(save_out_text, "w") as f:
                f.write(tokenizer.decode(generated_ids.squeeze().tolist()))

            eos_happened.append(gen_token_id)

            generated_ids = generated_ids[:, :-n_remove]

            next_token_id = generated_ids[:, -1:]

            cache_position = cache_position[-1:] - n_remove

            attention_mask = attention_mask[:, :-n_remove]

            new_kv_cache = []

            for i in range(num_layers):
                past_key, past_value = kv_cache[i]
                past_key = past_key[:, :, n_remove:, :]
                past_value = past_value[:, :, n_remove:, :]
                new_kv_cache.append((past_key, past_value))

            kv_cache = tuple(new_kv_cache)

    print("EOS happened at", eos_happened)

    return generated_ids


# Forth Approach


@torch.inference_mode()
def long_generate_approach4(
    model,
    tokenizer,
    input_ids,
    max_new_tokens,
    eos_token_id,
    sampling_params,
    n_remove: int,
):
    """
    4. Regenerating and Resampling Tokens Prior to the EOS Token with Dynamic Temperature Adjustment
    The fourth method enhances the previous approach by incorporating a dynamic temperature adjustment during the regeneration of tokens, aiming to improve both diversity and coherence in the generated sequence.

    """

    num_layers = model.config.num_hidden_layers  # being used in the cache

    default_temp = sampling_params["temperature"]  # default_temp = 1.0

    attention_mask = torch.ones_like(input_ids).to(model.device)

    cache_position = torch.arange(
        input_ids.shape[1], dtype=torch.int64, device=model.device
    )

    generated_ids = input_ids

    next_token_id = input_ids

    kv_cache = None

    eos_happened = []

    for gen_token_id in range(max_new_tokens):

        outputs = model(
            input_ids=next_token_id,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=kv_cache,
            use_cache=True,
        )

        if sampling_params["temperature"] > default_temp:
            sampling_params["temperature"] -= 0.1

        next_token_id = sampling(outputs.logits, **sampling_params)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
        )

        cache_position = cache_position[-1:] + 1

        kv_cache = outputs.past_key_values

        if next_token_id.squeeze().item() == eos_token_id:

            eos_happened.append(gen_token_id)

            # sampling_params["method"] = "top_k"
            sampling_params["temperature"] = 2.0

            save_out_text = f"results/inter_dynamic/{gen_token_id}_{now()}.txt"

            with open(save_out_text, "w") as f:
                f.write(tokenizer.decode(generated_ids.squeeze().tolist()))

            generated_ids = generated_ids[:, :-n_remove]

            next_token_id = generated_ids[:, -1:]

            cache_position = cache_position[-1:] - n_remove

            attention_mask = attention_mask[:, :-n_remove]

            new_kv_cache = []
            for i in range(num_layers):
                past_key, past_value = kv_cache[i]
                past_key = past_key[:, :, n_remove:, :]
                past_value = past_value[:, :, n_remove:, :]
                new_kv_cache.append((past_key, past_value))
            kv_cache = tuple(new_kv_cache)

    print("EOS happened at", eos_happened)

    return generated_ids
