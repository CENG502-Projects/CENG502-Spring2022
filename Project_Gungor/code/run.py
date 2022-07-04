import subprocess
import numpy as np
from datetime import datetime

pyfile = "/home/user502/dev/dive/BalancedMetaSoftmax-Classification/main.py"
config = "/home/user502/dev/dive/BalancedMetaSoftmax-Classification/config/CIFAR100_LT/dive_bbn_recipe.yaml"
teacher_model = "/home/user502/dev/dive/BalancedMetaSoftmax-Classification/logs/CIFAR100_LT/models/resnet32_balanced_softmax_imba100/final_model_checkpoint.pth"
teacher_config = "/home/user502/dev/dive/BalancedMetaSoftmax-Classification/config/CIFAR100_LT/balanced_softmax_imba100.yaml"

for weight in np.arange(0.05, 1, 0.05):
  file_name = f"/home/user502/dev/dive/BalancedMetaSoftmax-Classification/logs/CIFAR100_LT/models/dive_imba100_bbn_recipe/log_{weight}.txt"
  with open(file_name, "w+") as log_file:
    now = datetime.now()
    cmd = ["python3", pyfile, "--cfg", config, "--weight", str(weight), "--teacher_model_path", teacher_model, "--teacher_model_config", teacher_config]
    print(f"Running {cmd} at {now.strftime('%H:%M:%S')}")
    subprocess.run(cmd, stdout=log_file, stderr=log_file)
    now = datetime.now()
    print(f"Finished at {now.strftime('%H:%M:%S')}\n")

