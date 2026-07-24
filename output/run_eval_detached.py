"""Detached evaluation runner: runs eval_biomedical.py for a dataset and appends EXIT_CODE to the log."""
import os
import subprocess
import sys

dataset = sys.argv[1]
max_samples = sys.argv[2] if len(sys.argv) > 2 else "0"
batch_size = sys.argv[3] if len(sys.argv) > 3 else "4"
log_path = rf"D:\SDURA\MiniGPT-4\output\biomedical_eval\logs\{dataset}_test.log"

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

cmd = [
    r"D:\HP\Mniconda3\envs\BioMed\python.exe", "-u", "eval_biomedical.py",
    "--cfg-path", "eval_config.yaml",
    "--dataset", dataset,
    "--split", "test",
    "--batch-size", batch_size,
    "--max-new-tokens", "100",
    "--output-dir", "output/biomedical_eval",
]
if max_samples != "0":
    cmd += ["--max-samples", max_samples]

with open(log_path, "wb") as log:
    proc = subprocess.Popen(
        cmd,
        cwd=r"D:\SDURA\MiniGPT-4",
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    code = proc.wait()

with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"\nEXIT_CODE={code}\n")
