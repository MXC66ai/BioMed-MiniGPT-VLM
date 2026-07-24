@echo off
cd /d D:\SDURA\MiniGPT-4
"D:\HP\Mniconda3\envs\BioMed\python.exe" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 > output\biomedical_eval\logs\torch_install.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> output\biomedical_eval\logs\torch_install.log
