import os
import subprocess

bash_script = """#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /content
uv run --with 'torch,pandas,numpy,scipy' python colab_main_train.py > colab_main_log.txt 2>&1
"""

with open("/content/colab_main_job.sh", "w") as f:
    f.write(bash_script)

os.chmod("/content/colab_main_job.sh", 0o755)
subprocess.Popen(["nohup", "/content/colab_main_job.sh"], cwd="/content",
                 stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'),
                 preexec_fn=os.setpgrp)
print("Colab main branch training started!")
