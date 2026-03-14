import json
import random
import argparse

import numpy as np
from tqdm import tqdm
import os

from utils import MODEL_DICT
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 16384,
    "math": 8192,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def main_vllm(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    fout = open(args.save_path, "w")

    prompts = []
    test_data = []

    with open(args.dataset_path) as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[args.dataset_name][0]

            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)

            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)

            if args.num_samples is not None and index + 1 >= args.num_samples:
                break

    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        batch_prompts = prompts[i : i + args.eval_batch_size]
        
        sampling_params = SamplingParams(
            temperature=args.temperature,   # Use 0.0 for deterministic output like the HF version
            top_p=args.top_p,
            max_tokens=args.max_length,
            n=args.n_gen,              # <-- number of samples per prompt
        )
        # vllm automatically handles tokenization internally, with adding special tokens on
        if args.use_chat_template:
            prompt_message = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
            texts =  [args.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) for messages in prompt_message]
            outputs = model.generate(texts, sampling_params)
        else:
            outputs = model.generate(batch_prompts, sampling_params)

        batch_token_stats = []
        batch_outputs = []
        
        for j, output in enumerate(outputs):
            # In vLLM, prefill_length is simply the length of prompt_token_ids
            prefill_length = len(output.prompt_token_ids)
            for generation_idx in range(args.n_gen):
                completion_tokens = len(output.outputs[generation_idx].token_ids)
                total_tokens = prefill_length + completion_tokens
                
                batch_token_stats.append(
                    {
                        "generation_idx": generation_idx,
                        "sample_idx": i + j,
                        "prefill_tokens": prefill_length,  # This is how you get prefill_length in vLLM
                        "output_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )
                
                # Extract the generated text directly from vLLM output
                batch_outputs.append(output.outputs[generation_idx].text)

        torch.cuda.empty_cache()

        for i in range(args.eval_batch_size):
            for j in range(len(batch_outputs)):
                sample_idx = batch_token_stats[i]["sample_idx"]
                generation_idx = batch_token_stats[j]["generation_idx"]
                test_data[sample_idx]["prompt"] = batch_prompts[i]
                test_data[sample_idx]["output"] = batch_outputs[j]
                test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
                test_data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
                test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
                test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]
                test_data[sample_idx]["generation_idx"] = batch_token_stats[j]["generation_idx"]

                fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

    fout.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_short_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # sampling config
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n_gen", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    args.dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    if args.max_length == -1: args.max_length = dataset2max_length[args.dataset_name]

    if args.model_path is None and args.model_short_name is None:
        raise ValueError("Either model_path or model_short_name must be provided.")
    elif args.model_path is None:
        args.model_path = MODEL_DICT.get(args.model_short_name, None)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    args.tokenizer = tokenizer
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        download_dir="/data/models",
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )

    main_vllm(args)
