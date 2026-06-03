import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA Adapters into Base Model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B", help="Path or HF ID of base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory to save merged model")
    args = parser.parse_args()

    print(f"Loading tokenizer from base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading base model: {args.base_model}...")
    # Load base model in float16/bfloat16. Quantization must be disabled for merging adapters!
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    print(f"Loading Peft wrapper with adapter: {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging PEFT adapter layers into base model...")
    # Merge weights and unload the adapter
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}...")
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Merged model saved successfully!")

if __name__ == "__main__":
    main()
