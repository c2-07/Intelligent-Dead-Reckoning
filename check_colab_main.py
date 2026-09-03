import subprocess
print("--- COLAB MAIN BRANCH LOG ---")
print(subprocess.getoutput("tail -n 30 /content/colab_main_log.txt"))
