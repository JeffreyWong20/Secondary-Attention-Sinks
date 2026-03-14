import torch
import logging

MODEL_DICT = {
    # deepseek distill models
    "deepseek-1.5b" :    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-7b"   :      "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-14b"  :     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-32b"  :     "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-8b"   :      "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",

    # gpt oss models
    "gpt-20b": "openai/gpt-oss-20b",

    # qwen 3 models
    "qwen3-14b"     : "Qwen/Qwen3-14B",
    "qwen3-14b-base": "Qwen/Qwen3-14B-Base",

    "qwen3-32b"     : "Qwen/Qwen3-32B",
    "qwen3-32b-base": "Qwen/Qwen3-32B-Base",

    "qwen3-8b"      : "Qwen/Qwen3-8B",
    "qwen3-8b-base" : "Qwen/Qwen3-8B-Base",

    "qwen3-1.7b"     : "Qwen/Qwen3-1.7B",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",

    "qwen3-0.6b"     : "Qwen/Qwen3-0.6B",
    "qwen3-0.6b-base": "Qwen/Qwen3-0.6B-Base",

    "qwen3-4b"          : "Qwen/Qwen3-4B",
    "qwen3-4b-base"     : "Qwen/Qwen3-4B-Base",
    "qwen3-4b-thinking" : "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-4b-instruct" : "Qwen/Qwen3-4B-Instruct-2507",

    # not needed 
    "qwen3-30b-base": "Qwen/Qwen3-30B-A3B-Base",
    "qwen3-30b"     : "Qwen/Qwen3-30B-A3B",

    # qwen qwq
    "qwq-32b": "Qwen/QwQ-32B",

    # qwen 2
    "qwen2-1.5b"         : "Qwen/Qwen2-1.5B",
    "qwen2-1.5b-instruct": "Qwen/Qwen2-1.5B-Instruct",

    "qwen2-math-1.5b"         : "Qwen/Qwen2-Math-1.5B",
    "qwen2-math-1.5b-instruct": "Qwen/Qwen2-Math-1.5B-Instruct",

    "qwen2-7b"          : "Qwen/Qwen2-7B",
    "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",

    "qwen2-math-7b"         : "Qwen/Qwen2-Math-7B",
    "qwen2-math-7b-instruct": "Qwen/Qwen2-Math-7B-Instruct",

    # qwen 2.5
    "qwen2.5-1.5b"              : "Qwen/Qwen2.5-1.5B",
    "qwen2.5-1.5b-instruct"     : "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-math-1.5b"         : "Qwen/Qwen2.5-Math-1.5B",
    "qwen2.5-math-1.5b-instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",

    "qwen2.5-7b"                : "Qwen/Qwen2.5-7B",
    "qwen2.5-7b-instruct"       : "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-math-7b"           : "Qwen/Qwen2.5-Math-7B",
    "qwen2.5-math-7b-instruct"  : "Qwen/Qwen2.5-Math-7B-Instruct",

    "qwen2.5-14b"               : "Qwen/Qwen2.5-14B",
    "qwen2.5-14b-instruct"      : "Qwen/Qwen2.5-14B-Instruct",

    "qwen2.5-32b"           : "Qwen/Qwen2.5-32B",
    "qwen2.5-32b-instruct"  : "Qwen/Qwen2.5-32B-Instruct",

    # llama 3.1
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",

    # llama 3
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",

    # llama 2
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",

    # code llama models
    "codellama-7b": "codellama/CodeLlama-7b-hf",
    "codellama-7b-instruct": "codellama/CodeLlama-7b-Instruct-hf",
    "codellama-13b": "codellama/CodeLlama-13b-hf",
    "codellama-13b-instruct": "codellama/CodeLlama-13b-Instruct-hf",


    # Phi 4
    "phi4-15b": "microsoft/phi-4",
    "phi4-15b-reasoning": "microsoft/Phi-4-reasoning",
    "phi4-4b-instruct": "microsoft/Phi-4-mini-instruct",
    "phi4-4b-reasoning": "microsoft/Phi-4-mini-reasoning",
    # "phi4-15b-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
    # Mistral
    "mathstral-7b": "mistralai/Mathstral-7B-v0.1",

}


def compute_matrix_based_entropy(
    hidden_states,
    alpha=1.0,
    eps=1e-10
):
    # hidden_states shape: (batch_size, sequence_length, hidden_size)
    # return shape: (batch_size, sequence_length)
    rank = min(hidden_states.shape[1], hidden_states.shape[2])
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    if hidden_states.shape[1] < hidden_states.shape[2]:
        K = hidden_states @ hidden_states.transpose(1, 2)
    else:
        K = hidden_states.transpose(1, 2) @ hidden_states

    # convert k to float32 to avoid nan
    K = K.float()

    # Compute eigenvalues (symmetric matrix → use symeig)
    eigs = torch.linalg.eigvalsh(K)

    # Clip small negative eigenvalues due to numerical errors
    eigs = torch.clamp(eigs, min=0.0)

    tr = eigs.sum()
    probs = eigs / tr  # probabilities

    if alpha != 1.0:
        probs = probs[:, -rank:] ** alpha
        probs = torch.log(probs.sum(dim=-1))
        S = (1.0 / (1 - alpha)) * probs
    else:
        # if alpha is 1.0, then we need change to shannon entropy
        probs = probs[:, -rank:]
        probs = -probs * torch.log(probs)
        S = torch.nansum(probs, dim=-1)

    return S, eigs

def get_full_text(prompt: str, response: str, model_name: str) -> str:
    if "DeepSeek-R1-Distill-" in model_name:
        full_text = prompt + response
    elif "oss" in model_name:
        full_text = prompt + response
    else:
        logging.warning(f"Model Name: {model_name} possibly requires chat template check. We use simple concatenation for now.")
        full_text = prompt + response
    return full_text

def get_full_text_chat(prompt: str, response: str, model_name: str, tokenizer) -> str:
    message = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    full_text = tokenizer.decode(message) + response
    return full_text
