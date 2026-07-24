@echo off
powershell -NoExit -Command "Get-Content -Wait -Tail 30 'D:\SDURA\MiniGPT-4\output\biomedical_eval\logs\pmc_vqa_test.log'"
