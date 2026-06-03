import argparse
import os
import json
import torch
from peft import PeftModel
from tqdm import tqdm
from utils import setup_model_and_tokenizer, load_and_prepare_dataset, extract_numerical_answer

def main():
    parser = argparse.ArgumentParser(description="Evaluate Math Reasoning Model Accuracy")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter weights (optional)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to local curated dataset (JSONL)")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of items to evaluate (for debugging)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save results")
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        load_in_4bit=False,
        use_cpu=args.use_cpu
    )
    
    # If adapter is specified, load PEFT adapter weights
    if args.adapter_path:
        print(f"Loading LoRA adapter weights from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        
    model.eval()
    
    # Configure padding side to left for auto-regressive generation
    tokenizer.padding_side = "left"

    # Load evaluation dataset
    dataset = load_and_prepare_dataset(args.dataset_path)
    
    # Use validation split if using full dataset
    if "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        # If it's a flat dataset list, take the test/validation subset or full
        eval_dataset = dataset
        
    if args.limit > 0:
        print(f"Limiting evaluation to {args.limit} samples...")
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.limit)))
        
    print(f"Starting evaluation on {len(eval_dataset)} samples...")
    
    correct = 0
    results = []
    
    for item in tqdm(eval_dataset):
        messages = item["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
        
        # Format using the model's native template
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        pred_ans = extract_numerical_answer(gen_text)
        true_ans = extract_numerical_answer(assistant_msg)
        
        is_correct = False
        if pred_ans is not None and true_ans is not None and abs(pred_ans - true_ans) < 1e-2:
            correct += 1
            is_correct = True
            
        results.append({
            "problem": user_msg,
            "ground_truth_raw": assistant_msg,
            "generated_raw": gen_text,
            "pred_answer": pred_ans,
            "true_answer": true_ans,
            "correct": is_correct
        })
        
    accuracy = correct / len(eval_dataset) if len(eval_dataset) > 0 else 0.0
    print(f"\nEvaluation Results:")
    print(f"Total Samples: {len(eval_dataset)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Save results
    output_data = {
        "metadata": {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "total_samples": len(eval_dataset),
            "correct": correct,
            "accuracy": accuracy
        },
        "results": results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
