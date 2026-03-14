from transformers.models.qwen2.modeling_qwen2 import FlashAttentionKwargs, Cache, Unpack, apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
from typing import Optional, Callable
from collections import defaultdict
import torch



def light_qwen_patching_model(collector, model, model_short_name=None, target_layers=None, collect_targets=[]):
    def attn_output_hook(self, input, output, collector):
        attn_output, attn_weights = output
        collector['attn_output'].append(attn_output.detach().cpu())
        return output

    def mlp_input_hook(self, input, output, collector):
        hidden_states = input[0]
        collector['mlp_input'].append(hidden_states.detach().cpu())
        return output

    def mlp_down_output_hook(self, input, output, collector):
        down_tensor = output
        collector['down'].append(down_tensor.detach().cpu())
        return output

    def k_output_hook(self, input, output, collector):
        k_tensor = output[0]
        collector['k'].append(k_tensor.detach().cpu())
        return output

    def v_output_hook(self, input, output, collector):
        v_tensor = output[0]
        collector['v'].append(v_tensor.detach().cpu())
        return output

    handles = []

    for idx, layer in enumerate(model.model.layers):
        if target_layers is not None and idx not in target_layers:
            continue

        if 'attn_output' in collect_targets:
            handles.append(layer.self_attn.register_forward_hook(
                lambda module, input, output, c=collector[idx]: attn_output_hook(module, input, output, c)
            ))
        if 'mlp_input' in collect_targets:
            handles.append(layer.mlp.register_forward_hook(
                lambda module, input, output, c=collector[idx]: mlp_input_hook(module, input, output, c)
            ))
        if 'k' in collect_targets:
            handles.append(layer.self_attn.k_proj.register_forward_hook(
                lambda module, input, output, c=collector[idx]: k_output_hook(module, input, output, c)
            ))
        if 'v' in collect_targets:
            handles.append(layer.self_attn.v_proj.register_forward_hook(
            lambda module, input, output, c=collector[idx]: v_output_hook(module, input, output, c)
        ))
        if 'down' in collect_targets:
            handles.append(layer.mlp.down_proj.register_forward_hook(
                lambda module, input, output, c=collector[idx]: mlp_down_output_hook(module, input, output, c)
            ))

    return handles



def patch_qwen_attention(self, collector):
    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        collect_target: Optional[str] = None,
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

        if "attn_weights" in collect_target:
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
        if "attn_weights" in collect_target:
            collector["attn_weights"].append(attn_weights.detach().float().cpu())
        del attn_weights
        attn_weights = None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if "attn_output" in collect_target:
            collector["attn_output"].append(attn_output.detach().float().cpu())
        return attn_output, attn_weights
    self.forward = patched_forward

def patch_qwen_mlp(self, collector):
    def patched_forward(x, collect_target = ['mlp_input', 'up', 'gate', 'act', 'elementwise', 'down']):
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


def heavy_qwen_patching_model(collector, model, model_short_name=None, target_layers=None, patch_mlp=False):
    for idx, layer in enumerate(model.model.layers):
        if target_layers is not None and idx not in target_layers:
            continue
        collector[idx] = defaultdict(list)
        patch_qwen_attention(layer.self_attn, collector[idx])
        if patch_mlp:
            patch_qwen_mlp(layer.mlp, collector[idx])