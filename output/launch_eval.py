"""Launch run_eval_detached.py fully detached (no console window, survives window-close events)."""
import subprocess
import sys

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

dataset = sys.argv[1]
extra = sys.argv[2:]

proc = subprocess.Popen(
    [
        r"D:\HP\Mniconda3\envs\BioMed\python.exe",
        r"D:\SDURA\MiniGPT-4\output\biomedical_eval\logs\run_eval_detached.py",
        dataset,
    ] + extra,
    creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
    close_fds=True,
)
print(f"LAUNCHED {dataset} runner PID={proc.pid}")
