import torch
from utils import compute_matrix_based_entropy, get_full_text, get_full_text_chat, MODEL_DICT
from typing import List
import os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hidden state based Methods

def hidden_state_norm_detection(
    hidden_states: List,
    bos_sink_pos: List[int],
    tokenized_ids,
    enable_sink_clipper: bool = True,
):
    """
    Detect the sink position based on hidden state norms.
    Args:
        hidden_states (List): List of hidden states from each layer. Each element has shape (batch_size, sequence_length, hidden_size).
        bos_sink_pos (List[int]): List of positions indicating the beginning of sink tokens for each sample in the batch.
    Returns:
        top_k_indices (torch.Tensor): Tensor of shape (batch_size, k) containing the
    """
    sink_map = {}

    if len(bos_sink_pos) == 1:
        bos_sink_index = bos_sink_pos[0]
    else:
        # pick the highest norm among the provided positions
        potential_bos_states = torch.stack([
            torch.stack(
                [layer[0, pos, :].unsqueeze(0) for pos in bos_sink_pos], dim=0
            )
            for layer in hidden_states
        ], dim=0)

        bos_sink_index = torch.argmax(
            torch.norm(potential_bos_states, dim=-1).mean(dim=0)
        ).item()
        logger.info(f"Multiple BOS sink positions provided. Selected position {bos_sink_index} with highest average norm across layers.")
        sink_map[bos_sink_index] = {
                        'layer': [i for i in range(len(hidden_states))],
                        'token_id': tokenized_ids['input_ids'][0, bos_sink_index].item(),
                    }
        
    start_layer, end_layer = 0, len(hidden_states)
    if enable_sink_clipper:
        # we only look at the layer when the BOS sink appears
        bos_sink_norm = torch.stack([
            layer[0, bos_sink_index, :].norm(dim=-1) for layer in hidden_states
        ])
        bos_sink_norm_diff = bos_sink_norm[1:]  - bos_sink_norm[:-1] 

        start_layer = torch.argmax(bos_sink_norm_diff).item() + 1  # +1 because of the diff
        end_layer = torch.argmin(bos_sink_norm_diff[start_layer:]).item() + start_layer + 1 # +1 because of the diff
        logger.info(f"Sink clipper enabled. Analyzing layers from {start_layer} to {end_layer}.")

        sink_map[bos_sink_index] = {
                        'layer': [i for i in range(start_layer, end_layer)],
                        'token_id': tokenized_ids['input_ids'][0, bos_sink_index].item(),
                    }


    for layer_idx in range(start_layer, end_layer):
        batched_layer_hidden_states = hidden_states[layer_idx]  # (batch_size, sequence_length, hidden_size)
        layer_hidden_states = batched_layer_hidden_states[0]    # (sequence_length, hidden_size)


        layer_bos_hidden_state = layer_hidden_states[bos_sink_index, :]  # (hidden_size)
        layer_rest_hidden_states = layer_hidden_states[bos_sink_index+1:, :]  # (sequence_length - 1, hidden_size)

        # compute cosine similarity
        cosine_similarities = torch.nn.functional.cosine_similarity(
            layer_rest_hidden_states,
            layer_bos_hidden_state.unsqueeze(0),
            dim=-1
        )  # (sequence_length)
        indices =  (cosine_similarities.abs() > 0.9).nonzero() + (bos_sink_index+1)

        if len(indices) > 0:
            for index in indices:
                index = index.squeeze().tolist()
                if index not in sink_map:
                    sink_map[index] = {
                        'layer': [],
                        'token_id': tokenized_ids['input_ids'][0, index].item(),
                    }
                sink_map[index]['layer'].append(layer_idx)

    return sink_map

    
def hidden_state_entropy_growth_detection(
    hidden_states: List,
    window_size: int,
    response_len: int,
    prompt_len: int,
    full_text_len: int = None,
    enable_with_prompt: bool = False,
):
    if full_text_len is None:
        full_text_len = prompt_len + response_len

    entropy_growth = torch.zeros(  # (layer, response_chunk)
        len(hidden_states),
        (response_len + window_size - 1) // window_size
    )
    entropy_growth_with_prompt = torch.zeros(  # (layer, response_chunk)
        len(hidden_states),
        (full_text_len + window_size - 1) // window_size
    )

    for layer in range(len(hidden_states)):
        logging.info(f"Response: Layer {layer}: {hidden_states[layer].shape}")
        for response_chunk in range(0, response_len, window_size):
            end = prompt_len + min(response_chunk + window_size, response_len)
            chunk_hidden_states = hidden_states[layer][:, prompt_len:end, :]  # shape: (1, chunk_size, hidden_size)

            c_entropy, _ = compute_matrix_based_entropy(chunk_hidden_states, alpha=1.0)
            entropy_growth[layer][response_chunk // window_size] = c_entropy


    if enable_with_prompt:
        for layer in range(len(hidden_states)):
            logging.info(f"BOS: Layer {layer}: {hidden_states[layer].shape}")
            for text_chunk in range(0, full_text_len, window_size):
                end = min(text_chunk + window_size, full_text_len)
                chunk_hidden_states = hidden_states[layer][:, :end, :]  # shape: (1, chunk_size, hidden_size)

                c_entropy, _ = compute_matrix_based_entropy(chunk_hidden_states, alpha=1.0)
                entropy_growth_with_prompt[layer][text_chunk // window_size] = c_entropy

    return entropy_growth, entropy_growth_with_prompt


def sink_labeling_machine(sink_map, model_layer_length):
    has_secondary = False
    for token_pos, info in sink_map.items():
        if type(info) is not dict:
            continue
        layers = info['layer']
        # bos sink usually appears in many layers, here we set a threshold of 15 layers
        if len(layers) < model_layer_length - 15 and len(layers) > 4:
            info['sink_type'] = 'secondary'
            has_secondary = True
        elif len(layers) >= model_layer_length - 15:
            info['sink_type'] = 'bos_sink'
        else:
            info['sink_type'] = 'noise'
        
    return sink_map, has_secondary
    

def prefill_steering_meansure(model, bos_token_activation_path, target_layer_idx, steer_threshold=100.0):
    """
    Suppress the norm of the sinks!
    """
    bos_token_activation = torch.load(bos_token_activation_path).to(model.device)
    def patch_qwen_mlp(self):
        def patched_forward(x):
            up = self.up_proj(x)
            gate = self.gate_proj(x)
            act = self.act_fn(gate)
            elementwise = act * up
            down = self.down_proj(elementwise)
            B, S, D = x.shape
            print(f"Patched MLP Forward: B={B}, S={S}, D={D}")
            mask = torch.cosine_similarity(bos_token_activation, down, dim=-1) > 0.98
            exclude_indices = [0,1,2]
            mask[:, exclude_indices] = False
            if mask.any():
                down[mask, :] = down[mask, :] * (steer_threshold / torch.norm(down[mask, :], dim=-1, keepdim=True))
            return down
        self.forward = patched_forward

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx != target_layer_idx:
            continue
        patch_qwen_mlp(layer.mlp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process model response entropy")
    parser.add_argument("--model_short_name", type=str, default="qwen2.5-7b", help="Short name of the model")
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU ID to use")
    parser.add_argument("--file_path", type=str, default="./outputs/output_n_1_deepseek_14b.jsonl", help="Path to the response sequences file")
    parser.add_argument("--output_dir", type=str, default="./results/sink_detection", help="Directory to save output results")
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--use_chat_template", action='store_true', help="Whether to use chat template for full text construction")
    parser.add_argument("--generate_heatmap", action='store_true', help="Whether to generate entropy growth heatmap for samples with secondary sink")
    parser.add_argument("--steer_threshold", type=float, default=-1, help="Threshold for steering measure")
    args = parser.parse_args()


    # python3 ./src/hidden_state_base.py --model_short_name deepseek-14b --file_path ./outputs/aime24/vllm/output_n_1_deepseek-14b.jsonl --output_dir ./results/sink_detection --gpu_id 1 --use_chat_template

    model_name = MODEL_DICT[args.model_short_name]
    file_path = args.file_path
    use_chat_template = args.use_chat_template
    output_dir = args.output_dir

    logger.info(f"Using model: {model_name}")
    logger.info(f"Response file path: {file_path}")
    logger.info(f"Using chat template: {use_chat_template}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GPU ID: {args.gpu_id}")
    logger.info(f"Entropy growth heatmap: {args.generate_heatmap}")


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="sdpa",
        cache_dir="/data/models",
    )
    model.eval()

    secondary_token_collector = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]

    if args.sample_num is None:
        args.sample_num = len(lines)

    for sample_index in tqdm(range(args.sample_num)):
        sample = lines[sample_index]
        prompt = sample['prompt']
        response = sample['output']
        full_text = get_full_text(prompt, response, model_name) if not use_chat_template else get_full_text_chat(prompt, response, model_name, tokenizer)


        tokenized_ids = tokenizer(
            [full_text],
            # padding="longest",
            return_tensors="pt",
            add_special_tokens=(not use_chat_template),
        ).to("cuda")

        # kept first 5000 tokens for testing
        if "32" in args.model_short_name or "gpt" in args.model_short_name:
            tokenized_ids = {k: v[:, :9000] for k, v in tokenized_ids.items()}
            logging.warning("token length is capped to 9000 for testing.")

        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        response_len = tokenized_ids['input_ids'].size(1) - prompt_len

        if args.steer_threshold != -1:
            logger.info("Applying steering measure...")
            logger.info(f"Loading BOS token activation from ./steer_vector/bos_token_activation_{args.model_short_name}.pt")
            prefill_steering_meansure(model, f'./steer_vector/bos_token_activation_{args.model_short_name}.pt', target_layer_idx=22, steer_threshold=args.steer_threshold)

        with torch.no_grad():
            model_output = model(**tokenized_ids, output_hidden_states=True, return_dict=True, use_cache=False, collect_target=["attn_output"])

        hidden_states = model_output.hidden_states[1:]  # NOTE: exclude the embedding layer

        sink_map = hidden_state_norm_detection(
            hidden_states=hidden_states,
            bos_sink_pos=[0,1,2] if use_chat_template else [0,1],
            tokenized_ids=tokenized_ids,
        )

        sample_map = {}
        sample_map['candidate_sinks'] = sink_map
        sample_map['sample_index'] = sample_index
        sample_map['prompt_len'] = prompt_len
        sample_map['response_len'] = response_len
        sample_map['full_text_len'] = tokenized_ids['input_ids'].size(1)
        secondary_token_collector.append(sample_map)


        sink_map, has_secondary = sink_labeling_machine(sink_map, model.config.num_hidden_layers)


        # =================================================================================================================
        # # Hidden State Entropy Growth Detection (Optional)
        # =================================================================================================================
        if has_secondary and args.generate_heatmap:
            enable_with_prompt = True
            entropy_growth, entropy_growth_with_bos = hidden_state_entropy_growth_detection(
                hidden_states=hidden_states,
                window_size=100,
                response_len=response_len,
                prompt_len=prompt_len,
                enable_with_prompt=enable_with_prompt,
            )
            # NOTE: You can also save entropy_growth_with_prompt if needed
            output_file_path = os.path.join(
                args.output_dir,
                args.model_short_name,
                f"entropy_growth_{args.model_short_name}_sample{sample_index}.pt")
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            torch.save(entropy_growth, output_file_path)
            if enable_with_prompt:
                torch.save(entropy_growth_with_bos, output_file_path.replace("entropy_growth", "entropy_growth_with_prompt"))
            logger.info(f"Entropy growth data saved to {output_file_path}")


        # clean up cache
        torch.cuda.empty_cache()

    if args.use_chat_template:
        output_file_path = os.path.join(
        args.output_dir,
        args.model_short_name,
        f"sink_detection_{args.model_short_name}_use_chat_template.jsonl")
    else:
        output_file_path = os.path.join(        
            args.output_dir,
            args.model_short_name, 
            f"sink_detection_{args.model_short_name}.jsonl")
        
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w') as f_out:
        for item in secondary_token_collector:
            f_out.write(json.dumps(item) + '\n')
    logger.info(f"Sink detection results saved to {output_file_path}")