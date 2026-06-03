# Parameter-Efficient Math Reasoning Fine-Tuning

## Overview
This repository provides a parameter-efficient supervised fine-tuning (SFT) pipeline designed to teach base causal language models structured mathematical reasoning using QLoRA. It supports bitsandbytes 4-bit quantization, Hugging Face SFTTrainer sequence packing, custom TensorBoard logging, and periodic validation checkpoint evaluation to track alignment accuracy.

## Project Structure
* `utils.py`: Setup utilities for base model/tokenizer loading, dataset standardization, and mathematical answer extraction.
* `train.py`: Main SFT/QLoRA training loop with training configuration and Custom MathEvalCallback.
* `evaluate.py`: Evaluation script to run batch inference on base/adapter models, parsing mathematical values and saving results.
* `merge_peft.py`: Merge script to combine PEFT adapter weights with base model weight tensors to export a standalone model.
* `run_train.ps1`: PowerShell script to run CPU debugging runs and standard GPU production configurations.
* `requirements.txt`: Package dependencies.
* `course-project-ideas.md`: Course syllabus and project ideas reference.

## Setup and Installation
To set up the environment and install dependencies, run:
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate # On Windows use: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Running the Application
To run the training pipeline, configure the PowerShell script.

### CPU Debugging Mode
For testing and pipeline verification on CPU with a tiny model (`Qwen/Qwen2.5-0.5B`):
```powershell
.\run_train.ps1 -Mode cpu-debug
```

### GPU Production Mode
For 4-bit quantized QLoRA mathematical reasoning SFT fine-tuning with a 7B model (`Qwen/Qwen2.5-7B`) on GPU:
```powershell
.\run_train.ps1 -Mode gpu-production
```

## Testing and Evaluation
To evaluate a trained adapter model and measure its final math reasoning accuracy on the validation subset, run:
```bash
python evaluate.py --model_name "Qwen/Qwen2.5-0.5B" --adapter_path "./results_cpu" --dataset_path "../constitutional-data-flywheel-sft-curation/output/curated_sft.jsonl" --use_cpu
```
This prints the total validation accuracy score and writes detailed prediction lists (showing ground truth vs generated answers) to `evaluation_results.json`.
