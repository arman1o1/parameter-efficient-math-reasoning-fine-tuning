import re
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset, load_dataset

def setup_model_and_tokenizer(model_name_or_path, load_in_4bit=False, use_cpu=False):
    """Load and configure model and tokenizer for SFT/QLoRA training or inference."""
    print(f"Loading tokenizer for {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # Configure padding token and side
    # padding_side='right' is standard for training to prevent issues with masking and packing
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model {model_name_or_path}...")
    
    # Configure device map and quantization
    if use_cpu:
        device_map = None
        torch_dtype = torch.float32
        bnb_config = None
    else:
        device_map = "auto"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        if load_in_4bit:
            print("Configuring 4-bit BitsAndBytes quantization (QLoRA)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype
            )
        else:
            bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # For PEFT training
    if not use_cpu:
        model.config.use_cache = False
        
    return model, tokenizer

def load_and_prepare_dataset(file_path):
    """
    Load dataset from a local JSONL file or Hugging Face hub (GSM8K).
    Returns a Hugging Face Dataset object.
    """
    if file_path and os.path.exists(file_path):
        print(f"Loading local dataset from {file_path}...")
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        dataset = Dataset.from_list(data)
    else:
        print("Dataset path not found or empty. Loading default GSM8K dataset from Hugging Face...")
        dataset_dict = load_dataset("gsm8k", "main")
        # Rename columns to standard conversational format or keep question/answer
        dataset = dataset_dict["train"]
        
    # Standardize data to messages format if not already
    # SFTTrainer handles standard chat templates natively if 'messages' or standard columns exist
    def format_dataset(example):
        # Already has messages structure (Conversational format)
        if "messages" in example:
            return {"messages": example["messages"]}
        
        # From GSM8K raw schema: question, answer
        if "question" in example and "answer" in example:
            return {
                "messages": [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["answer"]}
                ]
            }
        
        # Default prompt/completion schema
        if "prompt" in example and "completion" in example:
            return {
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]}
                ]
            }
            
        raise ValueError(f"Unrecognized data schema in example: {example.keys()}")
        
    formatted_dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)
    return formatted_dataset

def extract_numerical_answer(text):
    """
    Robust extraction of mathematical values.
    First tries to parse canonical GSM8K format '#### <value>'.
    Falls back to the last numerical expression found in the text.
    """
    if not text:
        return None
    
    # Try GSM8K marker first
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
            
    # Fallback to last number in the generated text
    # Matches integers and decimals, ignoring commas
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
            
    return None
