import transformers
import json
import matplotlib.pyplot as plt
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_full_text, get_full_text_chat, MODEL_DICT


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model_name = "Qwen/Qwen2-Math-7B"
file_path = "./outputs/aime24/vllm/output_n_1_deepseek-14b.jsonl"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="eager",
    cache_dir="/data/models",
)
model.eval()


from transformers.models.qwen2.modeling_qwen2 import FlashAttentionKwargs, Cache, Unpack, apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
from typing import Optional, Callable
from collections import defaultdict


def patch_qwen_attention(self, collector, attn_implementation="sdpa"):
    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        collect_target: Optional[str] = None,
        attn_weights_pos: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        if "q" in collect_target:
            collector["q"].append(query_states.detach().float().cpu())
        if "k" in collect_target:
            collector["k"].append(key_states.detach().float().cpu())
        if "v" in collect_target:
            collector["v"].append(value_states.detach().float().cpu())
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if "cos" in collect_target:
            collector["cos"] = cos.detach().float().cpu()
        if "sin" in collect_target:
            collector["sin"] = sin.detach().float().cpu()
        if "roped_q" in collect_target:
            collector["roped_q"].append(query_states.detach().float().cpu())
        if "roped_k" in collect_target:
            collector["roped_k"].append(key_states.detach().float().cpu())
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    
        if "attn_weights" in collect_target and attn_implementation == "eager":
            attention_interface = eager_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )
        if "attn_weights" in collect_target and attn_implementation == "eager":
            if attn_weights_pos is None:
                collector["attn_weights"].append(attn_weights.detach().float().cpu())
            else:
                collector["attn_weights"].append(attn_weights[0, :, :, attn_weights_pos].detach().float().cpu())
        del attn_weights
        attn_weights = None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if "attn_output" in collect_target:
            collector["attn_output"].append(attn_output.detach().float().cpu())
        return attn_output, attn_weights
    self.forward = patched_forward

def patch_qwen_mlp(self, collector):
    # ['mlp_input', 'up', 'gate', 'act', 'elementwise', 'down']
    def patched_forward(x, collect_target = ['mlp_input', 'down']):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        act = self.act_fn(gate)
        elementwise = act * up
        down = self.down_proj(elementwise)
        if "mlp_input" in collect_target:
            collector["mlp_input"].append(x.detach().float().cpu())
        if "up" in collect_target:
            collector["up"].append(up.detach().float().cpu())
        if "gate" in collect_target:
            collector["gate"].append(gate.detach().float().cpu())
        if "act" in collect_target:
            collector["act"].append(act.detach().float().cpu())
        if "elementwise" in collect_target:
            collector["elementwise"].append(elementwise.detach().float().cpu())
        if "down" in collect_target:
            collector["down"].append(down.detach().float().cpu())
        return down
    self.forward = patched_forward

    # collect attention output 
from collections import defaultdict
def attn_output_hook(self, input, output, collector):
    attn_output, attn_weights = output
    collector['attn_output'].append(attn_output.detach().cpu())
    return output

def mlp_input_hook(self, input, output, collector):
    hidden_states = input[0]
    collector['mlp_input'].append(hidden_states.detach().cpu())
    collector['down'].append(output.detach().cpu())
    return output


for target_layer_idx, _ in enumerate(model.model.layers):
    collector = defaultdict(list)
    # heavy patching approach
    def patch_model(model, collector):
        for idx, layer in enumerate(model.model.layers):
            if idx != target_layer_idx:
                patch_qwen_attention(layer.self_attn, collector, attn_implementation="sdpa")
            else:
                patch_qwen_attention(layer.self_attn, collector, attn_implementation="eager")
            # patch_qwen_mlp(layer.mlp, collector)
        return model
    model = patch_model(model, collector)

    from tqdm import tqdm
    cosine_similarities_dict = {}
    use_chat_template = True
    # sample_num = 300
    # n_gen = 10

    sample_num = 30
    n_gen = 1

    sink_info_path = f"./results/sink_detection/deepseek-14b/sink_detection_n_{n_gen}_deepseek-14b.jsonl"
    file_path = f"./outputs/aime24/vllm/output_n_{n_gen}_deepseek-14b.jsonl"

    cliff_token_collector = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        sample_num = min(len(lines), sample_num)

    with open(sink_info_path, 'r') as f:
        sink_info_lines = f.readlines()

    for sample_index in tqdm(range(sample_num)):
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

        # keep the first 9k tokens
        tokenized_ids = {k: v[:, :10000] for k, v in tokenized_ids.items()}
        
        sink_data = json.loads(sink_info_lines[sample_index])
        if sink_data['sample_index'] != sample_index:
            raise ValueError("Sample index mismatch.")

        candidate_sinks = sink_data['candidate_sinks']
        exclude_positions = [0, 1, 2]
        pos_list = [int(k) for k in candidate_sinks.keys() if (int(k) not in exclude_positions) and (int(k) < tokenized_ids['input_ids'].shape[1])]
        pos_list = [0] + pos_list  # always include BOS position
        with torch.no_grad():
            model_output = model(**tokenized_ids, output_hidden_states=True, return_dict=True, use_cache=False, collect_target=['attn_weights'], attn_weights_pos=pos_list)
        
        torch.cuda.empty_cache()

    save_dict = {
        'bos': {
            "attn_weights": [], # [num_head, seq_length]
        },
        "cliff": {
            "attn_weights": [],
            "token_id": [],
            "sample_index": [],
            'position': [],
            'life_time': [],
        },
    }


    unique_sample_id_token_ids_pos = set()
    sink_count = 0
    for sample_index in range(sample_num):
        sink_line = sink_info_lines[sample_index]
        sink_data = json.loads(sink_line)
        if sink_data['sample_index'] != sample_index:
            raise ValueError("Sample index mismatch.")
        candidate_sinks = sink_data['candidate_sinks']

        exclude_positions = [0, 1, 2]
        pos_list = [int(k) for k in candidate_sinks.keys() if (int(k) not in exclude_positions) and (int(k) < tokenized_ids['input_ids'].shape[1])]
        token_id_list = [info['token_id'] for pos, info in candidate_sinks.items() if (int(pos) not in exclude_positions) and (int(pos) < tokenized_ids['input_ids'].shape[1])]

        for idx, pos in enumerate(pos_list):
            # if (sample_index, candidate_sinks[str(pos)]['token_id'], pos) in unique_sample_id_token_ids_pos:
            if (candidate_sinks[str(pos)]['token_id'], pos) in unique_sample_id_token_ids_pos:
                continue
            sink_count += 1
            unique_sample_id_token_ids_pos.add((candidate_sinks[str(pos)]['token_id'], pos))
            save_dict["cliff"]['token_id'].append(candidate_sinks[str(pos)]['token_id'])
            save_dict["cliff"]["sample_index"].append(sample_index)
            save_dict["cliff"]['position'].append(pos)
            save_dict['bos']['attn_weights'].append(collector['attn_weights'][sample_index][:, :, 0]) # BOS token at position 0
            save_dict["cliff"]["attn_weights"].append(collector['attn_weights'][sample_index][:, :, idx+1]) # +1 because of BOS token at position 0
            save_dict["cliff"]['life_time'].append(len(candidate_sinks[str(pos)]['layer']))

    print(f"collected {sink_count} cliff")
    # create directory if not exists
    os.makedirs(f"./deepseek-14b", exist_ok=True)  
    torch.save(save_dict, f"./deepseek-14b/creation_layer_info_dict_deepseek-14b_n_{n_gen}_attn_{target_layer_idx}.pt")
    print(f"Saved attention weights and metadata for target layer {target_layer_idx} to deepseek-14b/creation_layer_info_dict_deepseek-14b_n_{n_gen}_attn_{target_layer_idx}.pt")




