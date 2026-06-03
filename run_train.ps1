# PowerShell runner script for Project 4 Fine-Tuning Pipeline

param(
    [string]$Mode = "cpu-debug" # "cpu-debug" or "gpu-production"
)

$DatasetPath = "../constitutional-data-flywheel-sft-curation/output/curated_sft.jsonl"
$BaseModelCPU = "Qwen/Qwen2.5-0.5B" # Tiny model for CPU verification
$BaseModelGPU = "Qwen/Qwen2.5-7B"   # Target base model for mathematical alignment

if ($Mode -eq "cpu-debug") {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "STARTING CPU DEBUGGING PIPELINE" -ForegroundColor Cyan
    Write-Host "This verification step runs on CPU using Qwen2.5-0.5B" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    # Run a tiny 5-step SFT pipeline to verify correctness of train.py, dataset loading, and callbacks
    python train.py `
        --model_name $BaseModelCPU `
        --dataset_path $DatasetPath `
        --output_dir "./results_cpu" `
        --epochs 1 `
        --max_steps 5 `
        --batch_size 1 `
        --grad_accum 1 `
        --learning_rate 1e-4 `
        --eval_steps 2 `
        --eval_limit 2 `
        --use_cpu
        
    Write-Host "`nCPU Debugging Run completed successfully. To evaluate this adapter:" -ForegroundColor Green
    Write-Host "python evaluate.py --model_name $BaseModelCPU --adapter_path './results_cpu' --dataset_path $DatasetPath --limit 3 --use_cpu" -ForegroundColor Yellow

} elseif ($Mode -eq "gpu-production") {
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "STARTING GPU QLoRA PRODUCTION PIPELINE" -ForegroundColor Green
    Write-Host "Requires CUDA GPU environment." -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green

    # Run QLoRA 4-bit SFT fine-tuning with cosine learning rate scheduler and evaluation
    python train.py `
        --model_name $BaseModelGPU `
        --dataset_path $DatasetPath `
        --output_dir "./results_gpu" `
        --epochs 3 `
        --batch_size 2 `
        --grad_accum 8 `
        --learning_rate 2e-4 `
        --lora_r 16 `
        --lora_alpha 32 `
        --load_in_4bit `
        --eval_steps 50 `
        --eval_limit 25
        
    Write-Host "`nGPU Production SFT Training complete. Merging PEFT adapter back to base model..." -ForegroundColor Green
    python merge_peft.py `
        --base_model $BaseModelGPU `
        --adapter_path "./results_gpu" `
        --output_path "./results_gpu_merged"
        
    Write-Host "`nTo evaluate the merged model:" -ForegroundColor Green
    Write-Host "python evaluate.py --model_name './results_gpu_merged' --dataset_path $DatasetPath" -ForegroundColor Yellow
} else {
    Write-Error "Invalid mode parameter. Choose 'cpu-debug' or 'gpu-production'."
}
