import torch

print("PyTorch:", torch.__version__)
print("HIP/ROCm version:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")


for i in range(torch.cuda.device_count()):
    try:
        x = torch.ones(4, device=f"cuda:{i}")
        print(i, torch.cuda.get_device_name(i), x, x.device)
    except Exception as e:
        print(f"Device {i} failed: {type(e).__name__}: {e}")