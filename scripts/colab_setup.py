import os
import subprocess

print("Setting up Colab environment...")
subprocess.run("sudo apt-get update && sudo apt-get install -y git-lfs", shell=True, check=True)
if not os.path.exists("IO-VNBD"):
    print("Cloning IO-VNBD dataset...")
    subprocess.run("git clone https://github.com/onyekpeu/IO-VNBD", shell=True, check=True)

print("Pulling LFS data for M (Driver B)...")
pull_cmd = "cd IO-VNBD && git lfs install && git lfs pull --include='Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/*.csv'"
subprocess.run(pull_cmd, shell=True, check=True)
print("Setup complete! Data is ready.")
