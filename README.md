# Secondary Attention Sinks



Analyze and visualize **secondary attention sinks** in open LLMs by:

- generating math responses,
- detecting sink tokens from hidden-state similarity to BOS,
- collecting layer-level activations,
- and running PCA / directional analyses.

This repo contains scripts and notebooks used for sink detection and interpretation experiments.

---

## Project Structure

```text
Secondary_Attention_Sinks/
├── src/
│   ├── run_math_vllm.py       # Generation with vLLM
│   ├── hidden_state_base.py   # Sink detection from hidden states
│   ├── patch_base.py          # Layer/state collection with hooks
│   ├── mlp_base.py            # MLP-direction similarity analysis
│   └── utils.py               # Model registry + utility functions
├── data/                      # Input datasets (jsonl)
├── outputs/                   # Model generations
├── results/                   # Sink detection + analysis outputs
├── 01.hidden_k_v_norm.ipynb
├── 02.hierarchical_levels.ipynb
├── 03.a.print_sink_direction.ipynb
├── 03.b.pca_and_clustering.ipynb
├── 04.pca.ipynb
└── 05.sink_score.ipynb
```

---

## Environment Setup

```bash
conda create -n sink python=3.11 -y
conda activate sink
pip install -r requirements.txt
```

### Optional HF cache setup

```bash
export HF_HOME=/data/models
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

---

## End-to-End Pipeline

### 1) Generate responses with vLLM

```bash
CUDA_VISIBLE_DEVICES=0 python3 ./src/run_math_vllm.py \
	--dataset_path ./data/aime24.jsonl \
	--save_path ./outputs/aime24/vllm/output_n_10_deepseek-14b.jsonl \
	--model_short_name deepseek-14b \
	--max_length 16384 \
	--eval_batch_size 1 \
	--n_gen 10 \
	--temperature 0 \
	--use_chat_template
```

Notes:

- Supported dataset names are inferred from filename: `aime24`, `gsm8k`, `math`.
- You can also pass `--model_path` directly instead of `--model_short_name`.

### 2) Detect secondary sinks from hidden states

```bash
model_short_name=deepseek-14b
python3 ./src/hidden_state_base.py \
	--model_short_name $model_short_name \
	--file_path ./outputs/aime24/vllm/output_n_10_deepseek-14b.jsonl \
	--output_dir ./results/sink_detection \
	--gpu_id 0 \
	--sample_num 30 \
	--use_chat_template
```

Output file:

```text
./results/sink_detection/<model_short_name>/sink_detection_<model_short_name>_use_chat_template.jsonl
```

### 3) (Optional) Collect per-layer states for selected samples

```bash
model_short_name=deepseek-14b
sample_index=4

python3 ./src/patch_base.py \
	--model_short_name $model_short_name \
	--file_path ./outputs/aime24/vllm/output_n_10_deepseek-14b.jsonl \
	--output_dir ./results/sink_detection \
	--gpu_id 0 \
	--sample_index $sample_index \
	--collector_targets residual k v \
	--use_chat_template
```

### 4) Analyze MLP directional similarity around sink layers

```bash
model_short_name=deepseek-14b
dataset=aime24

python3 ./src/mlp_base.py \
	--model_short_name $model_short_name \
	--dataset $dataset \
	--file_path ./outputs/${dataset}/vllm/output_n_10_deepseek-14b.jsonl \
	--output_dir ./results/sink_detection \
	--gpu_id 0 \
	--sink_info_path ./results/sink_detection/${model_short_name}/sink_detection_${model_short_name}_use_chat_template.jsonl \
	--sample_num 30 \
	--use_chat_template
```

---

## Notebook Guide

- `01.hidden_k_v_norm.ipynb`: Visualize key/value and hidden-state norm/cosine behavior for sink analysis.
- `02.hierarchical_levels.ipynb`: Compare sink-layer hierarchy behavior across model settings.
- `03.a.print_sink_direction.ipynb`: Collect and export token groups for directional analysis.
- `03.b.pca_and_clustering.ipynb`: PCA + clustering exploration on collected states.
- `04.pca.ipynb`: Detailed PCA-based direction and scale analysis.
- `05.sink_score.ipynb`: Compute sink-related scoring summaries.

Terminology used in notebooks:

- **cliff**: identified secondary sink tokens.
- **other**: tokens with the same token IDs as cliff tokens but not identified as sinks.

---

## Supported Model Short Names

Model aliases are defined in `src/utils.py` (`MODEL_DICT`) and include families such as:

- DeepSeek-R1 Distill (`deepseek-1.5b`, `deepseek-7b`, `deepseek-14b`, ...)
- Qwen / Qwen2 / Qwen2.5 / Qwen3 variants
- Llama variants
- Phi variants
- other research backbones listed in the mapping

If needed, bypass aliases by passing `--model_path <hf_repo_id>`.

---

## Practical Tips

- Keep `--use_chat_template` consistent between generation and downstream analysis.
- Long contexts can be memory-heavy; use smaller `--sample_num` first.
- Some scripts cap sequence length for larger models (for stability/OOM avoidance).
- Prefer one analysis stage at a time and clear CUDA memory between runs.

---

## Cleanup

```bash
conda deactivate
conda remove --name sink --all
```
