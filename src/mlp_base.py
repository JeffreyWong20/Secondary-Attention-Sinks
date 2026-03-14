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
from model import apply_light_patching_model, apply_heavy_patching_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process model response entropy")
    parser.add_argument("--model_short_name", type=str, default="qwen2.5-7b", help="Short name of the model")
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU ID to use")
    parser.add_argument("--dataset", type=str, default="aime24", help="Dataset name")
    # file to read input samples
    parser.add_argument("--file_path", type=str, default="./outputs/output_n_1_deepseek_14b.jsonl", help="Path to the response sequences file")
    parser.add_argument("--output_dir", type=str, default="./results/sink_detection", help="Directory to save output results")
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--use_chat_template", action='store_true', help="Whether to use chat template for full text construction")
    # file to read sink info
    parser.add_argument("--sample_index", type=int, default=4, help="Sample index for plotting")
    parser.add_argument("--sink_info_path", type=str, default=None, help="Path to sink info file for selecting target layers")
    parser.add_argument(
        "--collect_target",
        type=str,
        nargs="+",
        default=["attn_output", "mlp"],
        choices=["q", "k", "v", "cos", "sin", "roped_q", "roped_k", "attn_weights", "attn_output", "mlp"],
        help="Target tensors to collect during heavy patching"
    )
    args = parser.parse_args()


    # python3 ./src/patch_base.py --model_short_name deepseek-14b --file_path ./outputs/aime24/vllm/output_n_1_deepseek-14b.jsonl --output_dir ./results/sink_detection --gpu_id 1 --use_chat_template

    model_name = MODEL_DICT[args.model_short_name]
    file_path = args.file_path
    use_chat_template = args.use_chat_template
    output_dir = args.output_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.sink_info_path is None:
        if args.use_chat_template:
            args.sink_info_path = f"./results/sink_detection/{args.dataset}/{args.model_short_name}/sink_detection_{args.model_short_name}_use_chat_template.jsonl"
        else:
            args.sink_info_path = f"./results/sink_detection/{args.dataset}/{args.model_short_name}/sink_detection_{args.model_short_name}.jsonl"

    logger.info(f"Using model: {model_name}")
    logger.info(f"Response file path: {file_path}")
    logger.info(f"Using chat template: {use_chat_template}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Heavy patching for model: {args.model_short_name}")
    logger.info(f"Reading sink info from: {args.sink_info_path}")
    logger.info(f"Sample index for processing: {args.sample_index}")
    logger.info(f"Collecting targets: {args.collect_target}")

    if "attn_weights" in args.collect_target:
        attn_mode = "eager"
    else:
        attn_mode = "sdpa"
    logger.info(f"Attention mode: {attn_mode}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=attn_mode,
        cache_dir="/data/models",
    )
    model.eval()

    cliff_token_collector = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]

    if args.sample_num is None:
        args.sample_num = len(lines)

    cosine_similarities_dict = {}
    for sample_index in tqdm(range(args.sample_num)):
        # if sample_index != args.sample_index:
        #     continue
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
        
        collector = defaultdict(list)


        # locate target layers from sink info
        sink_pos_creation_life = set()
        target_layers = None
        with open(args.sink_info_path, 'r') as f:
            sink_info_lines = f.readlines()
            for sink_line in sink_info_lines:
                sink_data = json.loads(sink_line)
                if sink_data['sample_index'] == sample_index:
                    candidate_sinks = sink_data['candidate_sinks']
                    for pos, info in candidate_sinks.items():
                        # Extract the creation layers of the sinks
                        sink_pos_creation_life.add((int(pos), info['layer'][0], len(info['layer'])))

        # we are only interested in cliffs currently
        logger.info(f"Identified sink positions and their creation layers and life levels: {sink_pos_creation_life}")
        target_layers = sorted(list(set(layer for _, layer, _ in sink_pos_creation_life)))
        logger.info(f"Target layers for heavy patching: {target_layers}")
        
        apply_heavy_patching_model(collector = collector, model=model, model_short_name=args.model_short_name, target_layers=target_layers, patch_mlp=("mlp" in args.collect_target))

        with torch.no_grad():
            model_output = model(**tokenized_ids, output_hidden_states=True, return_dict=True, use_cache=False, collect_target=args.collect_target)


        collector['hidden_states'] = model_output.hidden_states[1:]  # Exclude embedding layer hidden states
        meta_collector = {'state': collector, 'target_layers': target_layers}

        bos_sink = 0
        bos_sink_list = [0, 1, 2]
        for pos, layer, _ in sink_pos_creation_life:
            if pos in bos_sink_list:
                continue
            for state_name, state_tensor in meta_collector['state'][layer].items():
                
                try:
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        state_tensor[0][:, pos, :],
                        state_tensor[0][:, bos_sink, :],
                        dim=-1
                    ).item()
                except Exception as e:
                    breakpoint()
                if state_name not in cosine_similarities_dict:
                    cosine_similarities_dict[state_name] = []
                cosine_similarities_dict[state_name].append(cosine_sim)

        # clean up cache
        torch.cuda.empty_cache()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for state_name, cosine_sims in cosine_similarities_dict.items():
        if state_name == "attn_output":
            continue

        if state_name == "mlp_input":
            state_name = r"$x_t$"
        elif state_name == "down":
            state_name = r"$y_t^{down}$"
        elif state_name == "up":
            state_name = r"$y_t^{up}$"
        elif state_name == "gate":
            state_name = r"$y_t^{gate}$"
        elif state_name == "act":
            state_name = r"$y_t^{silu}$"
        elif state_name == "elementwise":
            state_name = r"$y_t^{silu} \odot y_t^{up}$"
        plt.plot(
            range(len(cosine_sims)),
            cosine_sims,
            marker='o',
            label=state_name
        )
    # plt.title(f"Cosine Similarities at Sink Positions - Sample {sample_index} - {args.model_short_name}")
    plt.xlabel("Sink Instance Index", fontsize=16)
    plt.ylabel("Cosine Similarity with BOS Sink", fontsize=16)
    plt.legend(fontsize='large')
    plt.grid()
    plot_dir = os.path.join(
        args.output_dir,
        args.model_short_name,
    )
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir,
        f"cosine_similarities_sample{sample_index}_{args.model_short_name}.pdf"
    )
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved cosine similarity plot to {plot_path}")
    plt.close()
    # output_file_path = os.path.join(
    #     args.output_dir,
    #     args.model_short_name, 
    #     f"heavy_patch_{args.model_short_name}_sample{args.sample_index}.pt")
    # os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # torch.save(collector, output_file_path)
    # logger.info(f"Saved patched MLP tensors to {output_file_path}")