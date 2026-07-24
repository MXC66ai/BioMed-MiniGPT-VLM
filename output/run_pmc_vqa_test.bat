@echo off
cd /d D:\SDURA\MiniGPT-4
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
"D:\HP\Mniconda3\envs\BioMed\python.exe" -u eval_biomedical.py --cfg-path eval_config.yaml --dataset pmc_vqa --split test --batch-size 4 --max-new-tokens 100 --output-dir output/biomedical_eval > output\biomedical_eval\logs\pmc_vqa_test.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> output\biomedical_eval\logs\pmc_vqa_test.log
