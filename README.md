# When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework

This repository contains the implementation of the research paper "When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework". The framework evaluates and compares single-model inference versus divide-and-conquer approaches for long context tasks across various language models.

## Overview

The framework implements two inference strategies:
- **Single Model Inference**: Direct processing of long contexts by language models
- **Divide and Conquer**: Breaking long contexts into chunks, processing them separately, and aggregating results

## Features

- Support for multiple language models (OpenAI GPT, Meta LLaMA, Qwen, Mistral)
- Multiple evaluation tasks (key-value retrieval, math reasoning, QA, summarization)
- Comprehensive evaluation metrics and scoring
- Parallel processing for efficient experimentation
- Context lengths from 1K to 128K tokens

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cd src
cp config.json.template config.json
```

4. Add your API keys to `src/config.json`:
```json
{
    "keys": {
        "TOGETHER_API_KEY": "your_together_api_key_here",
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "GEMINI_API_KEY": "your_gemini_api_key_here"
    }
}
```

## Usage

### Single Model Inference

Run inference with a single model processing the entire context:

```bash
cd src
python inference_sing.py --model MODEL_NAME --task TASK --len CONTEXT_LENGTH --save_dir ../outputs --n_proc 16
```

### Divide and Conquer Inference

Run inference using the divide-and-conquer approach:

```bash
cd src
python inference_dc.py --model MODEL_NAME --task TASK --len CONTEXT_LENGTH --ctx CONTEXT_WINDOW --save_dir ../outputs --n_proc 16
```

### Parameters

- `--model`: Model identifier (see supported models below)
- `--task`: Task type (see supported tasks below)
- `--len`: Input context length (1000, 2000, 4000, 8000, 15000, 30000, 60000, 120000)
- `--ctx`: Context window size for divide-and-conquer (only for `inference_dc.py`)
- `--save_dir`: Output directory for results
- `--n_proc`: Number of parallel processes

### Supported Models (Easy to add your own models)

| Model Key | Full Model Name |
|-----------|-----------------|
| `gpt4-1106` | gpt-4-1106-preview |
| `gpt4o` | gpt-4o-2024-08-06 |
| `gpt4omini` | gpt-4o-mini-2024-07-18 |
| `llama70b` | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo |
| `llama3b` | meta-llama/Llama-3.2-3B-Instruct-Turbo |
| `qwen72b` | Qwen/Qwen2.5-72B-Instruct-Turbo |
| `mistral7b` | mistralai/Mistral-7B-Instruct-v0.2 |

### Supported Tasks (Easy to add your own tasks)

| Task | Description | Max Output Length |
|------|-------------|-------------------|
| `kv` | Key-value retrieval from JSON data | 128 tokens |
| `math` | Mathematical reasoning (finding min/max numbers) | 128 tokens |
| `sum` | Text summarization | 2048 tokens |
| `qaib` | Question answering on books | 128 tokens |
| `char` | Character identification in dialogues | 128 tokens |
| `qalb` | Multiple choice question answering | 128 tokens |

## Data

The datasets are under `data/`, coming from synthetic, infinity-bench, long-bench, etc.

### Data Format

All data files are in JSONL format with the following structure:
```json
{
    "id": 0,
    "context": "...",
    "input": "...", 
    "answer": "..."
}
```

## Evaluation

### Computing Scores

After running inference, evaluate the results:

```bash
cd src
python compute_scores.py --input_file ../outputs/RESULT_FILE.jsonl
```

### Scoring Metrics

- **Key-Value Retrieval**: Exact match
- **Mathematical Tasks**: Exact numerical match
- **QA Tasks**: F1 score with answer normalization
- **Summarization**: ROUGE scores

Details can be found on `score_xx.py`

## Example Workflows

### 1. Compare Single vs Divide-and-Conquer on Key-Value Task

```bash
# Single model inference
python inference_sing.py --model gpt4o --task kv --len 8000 --save_dir ../outputs

# Divide and conquer inference
python inference_dc.py --model gpt4o --task kv --len 8000 --ctx 4000 --save_dir ../outputs

# Evaluate results
python compute_scores.py --input_file ../outputs/kv-gpt4o-len8000.jsonl
python compute_scores.py --input_file ../outputs/kv-gpt4o-len8000-ctx4000.jsonl
```

### 2. Batch Evaluation Across Models

```bash
# Run experiments for multiple models
for model in gpt4o llama70b qwen72b; do
    python inference_sing.py --model $model --task kv --len 8000 --save_dir ../outputs
    python inference_dc.py --model $model --task kv --len 8000 --ctx 4000 --save_dir ../outputs
done
```

## Output Format

Results are saved as JSONL files with the following structure:
```json
{
    "id": 0,
    "answer": "ground_truth_answer",
    "pred": "model_prediction",
    "context": "truncated_context_preview..."
}
```

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/                    # Evaluation datasets
│   ├── kv_1000.jsonl       # Key-value retrieval (1K context)
│   ├── kv_8000.jsonl       # Key-value retrieval (8K context)
│   └── ...
├── outputs/                 # Experiment results
├── src/                     # Source code
│   ├── config.json          # Configuration file (add your API keys)
│   ├── inference_sing.py    # Single model inference
│   ├── inference_dc.py      # Divide and conquer inference
│   ├── compute_scores.py    # Evaluation metrics
│   ├── data.py             # Data processing utilities
│   ├── score_sing.py       # Scoring for single inference
│   └── score_dc.py         # Scoring for divide-conquer
└── .gitignore
```

## Requirements

- Python 3.8+
- OpenAI API access (for GPT models)
- Together AI API access (for LLaMA, Qwen, Mistral models)
- Google AI API access (for Gemini models)

## API Costs

Please note that running experiments on large contexts and multiple models can incur significant API costs. Start with smaller context lengths and fewer samples for initial testing.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{divideconqueragents,
    title={When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework},
    author={Zhen (Zach) Xu, Shang Zhu, Jue Wang, Junlin Wang, Ben Athiwaratkun, Chi Wang, James Zou, Ce Zhang},
    journal={arxiv},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

1. **API Rate Limits**: Reduce `n_proc` parameter or add delays between requests
2. **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed
3. **API Key Errors**: Verify your API keys are correctly set in `config.json`