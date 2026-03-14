from .phi import light_phi_patching_model
from .qwen import light_qwen_patching_model, patch_qwen_attention, patch_qwen_mlp, heavy_qwen_patching_model

def apply_light_patching_model(collector, model, model_short_name=None, target_layers=None, collector_targets=[]):
    if "phi" in model_short_name:
        return light_phi_patching_model(collector, model, model_short_name, target_layers=target_layers, collect_targets=collector_targets)
    elif "qwen" in model_short_name or 'llama' in model_short_name or 'qwq' in model_short_name or "deepseek-14b" in model_short_name:
        return light_qwen_patching_model(collector, model, model_short_name, target_layers=target_layers, collect_targets=collector_targets)
    else:
        raise ValueError(f"Patching model not implemented for {model_short_name}")
    

def apply_heavy_patching_model(collector, model, model_short_name=None, target_layers=None, patch_mlp=False):
    if "qwen" in model_short_name or 'qwq' in model_short_name or "deepseek-14b" in model_short_name:
        return heavy_qwen_patching_model(collector, model, model_short_name, target_layers, patch_mlp)
    else:
        raise ValueError(f"Heavy patching not implemented for {model_short_name}")