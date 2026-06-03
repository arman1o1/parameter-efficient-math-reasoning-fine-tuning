import argparse
import os
import torch
from transformers import set_seed, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from utils import setup_model_and_tokenizer, load_and_prepare_dataset, extract_numerical_answer

class MathEvalCallback(TrainerCallback):
    """Custom callback to periodically evaluate mathematical accuracy on a validation subset."""
    def __init__(self, eval_dataset, tokenizer, eval_limit=20, eval_steps=50):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_limit = eval_limit
        self.eval_steps = eval_steps
        self.trainer = None
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n[MathEvalCallback] Running validation evaluation at step {state.global_step}...")
            model.eval()
            
            # Temporarily set padding side to left for auto-regressive generation
            prev_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            
            correct = 0
            total = min(len(self.eval_dataset), self.eval_limit)
            
            # Subsample validation dataset to limit CPU/GPU overhead
            eval_subset = self.eval_dataset.select(range(total))
            
            for item in eval_subset:
                messages = item["messages"]
                user_msg = next(m["content"] for m in messages if m["role"] == "user")
                assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
                
                # Format prompt using the model's native chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False
                    )
                
                # Decode only the generated response
                gen_ids = outputs[0][inputs.input_ids.shape[1]:]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                pred_val = extract_numerical_answer(gen_text)
                true_val = extract_numerical_answer(assistant_msg)
                
                if pred_val is not None and true_val is not None and abs(pred_val - true_val) < 1e-2:
                      correct += 1
                      
            accuracy = correct / total if total > 0 else 0.0
            print(f"[MathEvalCallback] Step {state.global_step}: Math Validation Accuracy = {accuracy * 100:.2f}% ({correct}/{total})")
            
            # Restore tokenizer padding side
            self.tokenizer.padding_side = prev_padding_side
            
            # Log metrics (automatically captured by TensorBoard / WandB if configured)
            if self.trainer is not None:
                self.trainer.log({"eval_math_accuracy": accuracy})
            
            # Revert model to training mode
            model.train()

def main():
    parser = argparse.ArgumentParser(description="Parameter-Efficient Math Reasoning Fine-Tuning Pipeline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model identifier")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to local curated dataset (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save trained model adapters")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if > 0)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--use_packing", action="store_true", help="Enable sequence packing for efficiency")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable bitsandbytes 4-bit quantization")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluate math accuracy every N steps")
    parser.add_argument("--eval_limit", type=int, default=10, help="Maximum validation samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Enable QLoRA only if GPU is available and use_cpu is False
    load_in_4bit = args.load_in_4bit and not args.use_cpu and torch.cuda.is_available()
    
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        load_in_4bit=load_in_4bit,
        use_cpu=args.use_cpu
    )
    
    # 2. Setup PEFT Model
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
        
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 3. Load datasets
    full_dataset = load_and_prepare_dataset(args.dataset_path)
    
    # Split into train/validation
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(eval_dataset)}")
    
    # Define conversational formatting function for SFTTrainer
    def formatting_prompts_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    # 4. Set up SFTConfig & SFTTrainer
    # Setup TensorBoard log directory
    logging_dir = os.path.join(args.output_dir, "logs")
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=1,
        eval_strategy="no",  # We run custom evaluation via callback
        save_strategy="steps",
        save_steps=20,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        use_cpu=args.use_cpu,
        packing=args.use_packing,
        max_length=512,
        report_to="tensorboard",
        remove_unused_columns=False
    )
    
    # Instantiate custom evaluation callback
    math_eval_callback = MathEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        eval_limit=args.eval_limit,
        eval_steps=args.eval_steps
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=training_args,
        callbacks=[math_eval_callback]
    )
    
    # Assign trainer to callback for logging
    math_eval_callback.trainer = trainer
    
    # Start training
    print("Starting SFT training...")
    trainer.train()
    
    # Save adapter model
    print(f"Saving final adapter model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
