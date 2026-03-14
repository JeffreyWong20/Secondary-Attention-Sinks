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
from model import apply_light_patching_model

def main(args):
    # python3 ./src/patch_base.py --model_short_name deepseek-14b --file_path ./outputs/aime24/vllm/output_n_1_deepseek-14b.jsonl --output_dir ./results/sink_detection --gpu_id 1 --use_chat_template
    model_name = MODEL_DICT[args.model_short_name]
    file_path = args.file_path
    use_chat_template = args.use_chat_template
    output_dir = args.output_dir

    logger.info(f"Using model: {model_name}")
    logger.info(f"Response file path: {file_path}")
    logger.info(f"Using chat template: {use_chat_template}")
    logger.info(f"Output directory: {output_dir}")

    logger.info(f"Setting CUDA_VISIBLE_DEVICES to {args.gpu_id}")
    logger.info(f"Collecting from target layers: {args.target_layers}" if args.target_layers is not None else "Collecting from all layers")
    logger.info(f"Collector targets: {args.collector_targets}")
    logger.info(f"Sample index: {args.sample_index}" if args.sample_index!=-1 else "Sample index: All samples")
    logger.info(f"Sample num: {args.sample_num}" if args.sample_num is not None else "Sample num: All samples")


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

    if args.target_layers is not None:
        collector = {layer_idx: defaultdict(list) for layer_idx in args.target_layers}
    else:
        collector = {layer_idx: defaultdict(list) for layer_idx in range(model.config.num_hidden_layers)}
        
    for sample_index in tqdm(range(args.sample_num)):
        if args.sample_index != -1 and sample_index != args.sample_index:
            continue
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

        if "32" in args.model_short_name:
            tokenized_ids = {k: v[:, :9000] for k, v in tokenized_ids.items()}
            logging.warning("token length is capped to 9000 for testing.")

        handles = apply_light_patching_model(collector = collector, model=model, model_short_name=args.model_short_name, target_layers=args.target_layers, collector_targets=args.collector_targets)

        with torch.no_grad():
            model_output = model(**tokenized_ids, output_hidden_states=True, return_dict=True, use_cache=False)
        for handle in handles:
            handle.remove()
        
        if 'residual' in args.collector_targets:
            for layer_idx in collector:
                collector[layer_idx]['residual'].append(model_output.hidden_states[layer_idx].detach().cpu())

        # clean up cache
        torch.cuda.empty_cache()



    # =========== Process collected data ==============
    # if no sink info path provided, directly save the collected data without filtering
    if args.sink_info_path is None:
        collector_targets_str = "-".join(args.collector_targets)
        target_layers_str = "all" if args.target_layers is None else "-".join(map(str, args.target_layers))
        if args.use_chat_template:
            output_file_path = os.path.join(
                args.output_dir,
                args.model_short_name,
                f"{collector_targets_str}_layer_{target_layers_str}_{args.model_short_name}_chat_template_sample{args.sample_index}.pt"
            )
        else:
            output_file_path = os.path.join(
                args.output_dir,
                args.model_short_name,
                f"{collector_targets_str}_layer_{target_layers_str}_{args.model_short_name}_sample{args.sample_index}.pt"
        )
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        torch.save(collector, output_file_path)
        logger.info(f"Saved {args.collector_targets} of {args.target_layers} tensors to {output_file_path}")
        return
    
    # =========== If sink info path provided, filter the collected data based on the sink info and then save =============
    # Filter and organize the collected data based on sink info
    logger.warning(f"Sink info path provided: {args.sink_info_path}, processing collected data with sink info.")
    with open(args.sink_info_path, 'r') as f:
        sink_info_lines = f.readlines()

    save_dict_all_layers = {}
    breakpoint()
    for layer_idx in collector.keys():
        save_dict = {
            'bos': {
                "down": [],
                "mlp_input": [],
                "attn_output": [],
                "residual": [],
            },
            "secondary": {
                "down": [],
                "mlp_input": [],
                "attn_output": [],
                "residual": [],
                "token_id": [],
                "sample_index": [],
                'position': []
            },
            "other": {
                "down": [],
                "mlp_input": [],
                "attn_output": [],
                "residual": [],
                "token_id": [],
                "sample_index": [],
                'position': []
            }
        }
        layer_collector = collector[layer_idx]

        unique_sample_id_token_ids_pos = set()

        for sample_index in tqdm(range(args.sample_num)):
            if args.sample_index != -1 and sample_index != args.sample_index:
                continue

            # Instead of appending, directly assign the BOS token data (BOS token should be the same across all samples)
            # save_dict["bos"]["down"] = layer_collector['down'][sample_index].squeeze()[0]
            # save_dict["bos"]["mlp_input"] = layer_collector['mlp_input'][sample_index].squeeze()[0]
            # save_dict["bos"]["attn_output"] = layer_collector['attn_output'][sample_index].squeeze()[0]
            save_dict["bos"]["residual"] = layer_collector['residual'][sample_index].squeeze()[0]


            sink_line = sink_info_lines[sample_index]
            sink_data = json.loads(sink_line)
            if sink_data['sample_index'] != sample_index:
                raise ValueError("Sample index mismatch.")
            candidate_sinks = sink_data['candidate_sinks']

            exclude_positions = [0, 1, 2]
            pos_list = [int(k) for k in candidate_sinks.keys() if int(k) not in exclude_positions]
            token_id_list = [info['token_id'] for pos, info in candidate_sinks.items() if int(pos) not in exclude_positions]
            save_dict["secondary"]['token_id'] += token_id_list
            save_dict["secondary"]["sample_index"] += [sample_index] * len(token_id_list)
            save_dict["secondary"]['position'] += pos_list
            for state in args.collector_targets:
                save_dict["secondary"][state].append(layer_collector[state][sample_index].squeeze()[pos_list])


            # add other token with same token_ids 
            prompt, response = lines[sample_index]['prompt'], lines[sample_index]['output']
            full_text = get_full_text(prompt, response, model_name) if not use_chat_template else get_full_text_chat(prompt, response, model_name, tokenizer)

            tokenized_ids = tokenizer(
                [full_text],
                # padding="longest",
                return_tensors="pt",
                add_special_tokens=(not use_chat_template),
            ).to("cuda")

            same_token_id_other_data_pos_list = [pos for pos, tid in enumerate(tokenized_ids['input_ids'][0]) if (tid in token_id_list) & (pos not in pos_list)]
            same_token_id_other_data_token_id_list = tokenized_ids['input_ids'][0][same_token_id_other_data_pos_list]
            save_dict["other"]['token_id'] += same_token_id_other_data_token_id_list
            save_dict["other"]["sample_index"] += [sample_index] * len(same_token_id_other_data_pos_list)
            save_dict["other"]['position'] += same_token_id_other_data_pos_list
            for state in args.collector_targets:
                save_dict["other"][state].append(layer_collector[state][sample_index].squeeze()[same_token_id_other_data_pos_list])



        for state in args.collector_targets:
            save_dict["secondary"][state] = torch.cat(save_dict["secondary"][state], dim=0)
            save_dict["other"][state] = torch.cat(save_dict["other"][state], dim=0)

        logger.info(f"collected {len(save_dict['secondary']['attn_output'])} secondary")
        logger.info(f"collected {len(save_dict['other']['attn_output'])} other")

        # save_dict_path = f"creation_layer_info_dict_deepseek-14b_n_{n_gen}.pt"
        # torch.save(save_dict, save_dict_path)

        unique_sample_id_token_ids_pos = set()
        filtered_data = {
            'down': [],
            'mlp_input': [],
            'attn_output': [],
            'residual': [],
            'token_id': [],
            'sample_index': [],
            'position': []
        }

        for i in range(len(save_dict['secondary']['token_id'])):
            token_id = save_dict['secondary']['token_id'][i]
            sample_index = save_dict['secondary']['sample_index'][i]
            position = save_dict['secondary']['position'][i]

            key_tuple = (token_id, position)
            if key_tuple in unique_sample_id_token_ids_pos:
                continue

            unique_sample_id_token_ids_pos.add(key_tuple)

            # Collect only the first occurrence of each unique combination
            filtered_data['token_id'].append(token_id)
            filtered_data['sample_index'].append(sample_index)
            filtered_data['position'].append(position)
            for state in args.collector_targets:
                filtered_data[state].append(save_dict['secondary'][state][i])

        for state in args.collector_targets:
            filtered_data[state] = torch.stack(filtered_data[state])
        save_dict['secondary'] = filtered_data

        logger.info(f"collected {len(save_dict['secondary']['attn_output'])} secondary")

        save_dict_all_layers[layer_idx] = save_dict
        logger.info(f"Saved data for layer {layer_idx} to save_dict_all_layers.")

    collector_targets_str = "-".join(args.collector_targets)
    target_layers_str = "all" if args.target_layers is None else "-".join(map(str, args.target_layers))
    output_file_path = os.path.join(
        args.output_dir,
        args.model_short_name,
        f"save_dict_{collector_targets_str}_layer_{target_layers_str}_{args.model_short_name}_sample{args.sample_index}.pt"
    )
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    torch.save(save_dict_all_layers, output_file_path)
    logger.info(f"Saved {args.collector_targets} of {args.target_layers} tensors to {output_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process model response entropy")
    parser.add_argument("--model_short_name", type=str, default="qwen2.5-7b", help="Short name of the model")
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU ID to use")
    parser.add_argument("--file_path", type=str, default="./outputs/output_n_1_deepseek_14b.jsonl", help="Path to the response sequences file")
    parser.add_argument("--output_dir", type=str, default="./results/sink_detection", help="Directory to save output results")
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--use_chat_template", action='store_true', help="Whether to use chat template for full text construction")
    parser.add_argument("--sample_index", type=int, default=-1, help="Only collect for this sample index")
    parser.add_argument("--target_layers", type=int, default=None, nargs='+', help="Target layers to collect sink info from. If None, collect from all layers")
    parser.add_argument(
        "--collector_targets",
        type=str,
        nargs='+',
        default=["residual", "k", "v"],
        choices=["attn_output", "k", "v", "mlp_input", "residual", "down"],
        help="Targets to collect from the model layers. Allowed: attn_output, k, v, mlp_input, residual, down. E.g., --collector_targets attn_output k v down"
    )
    parser.add_argument("--sink_info_path", type=str, default=None, help="Path to sink info jsonl file")
    args = parser.parse_args()
    main(args)