import torch




if torch.cuda.is_available(): 
    device = torch.device("cuda")
    print("Running on the GPU")
else: 
    device = torch.device("cpu")
    print("Running on the CPU")

import subprocess

try:
    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    nvidia_driver_version = nvidia_smi_output.decode().strip()
    print("NVIDIA driver version: ", nvidia_driver_version)
except Exception as e:
    print("Could not determine NVIDIA driver version. Error: ", str(e))
cuda_version = torch.version.cuda
print("CUDA version:", cuda_version)

pytorch_version = torch.__version__
print("PyTorch version:", pytorch_version)
import torch
print(torch.cuda.is_available())
