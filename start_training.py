import os
import subprocess

# Set environment variables to prevent CUDA OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

print("[BioMed MiniGPT VLM] Starting Stage 2 training with memory optimization...")
print(f"[BioMed MiniGPT VLM] PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

# Start training
result = subprocess.run([
    r'D:\HP\Mniconda3\envs\BioMed\python.exe',
    r'D:\SDURA\MiniGPT-4\train_biomedical.py',
    '--cfg-path', r'D:\SDURA\MiniGPT-4\train_configs\minigpt4_vicuna7b_biomedical_stage2.yaml'
])

print(f"[BioMed MiniGPT VLM] Training exited with code: {result.returncode}")
