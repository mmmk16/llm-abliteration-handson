"""0章 手順 3 の確認: PyTorch が MPS (Metal) を使えるか"""
import torch

print("torch", torch.__version__)
print("mps available:", torch.backends.mps.is_available())

x = torch.randn(1024, 1024, device="mps")
y = x @ x
torch.mps.synchronize()
print("matmul on mps ok:", y.shape, y.dtype)
print("allocated:", torch.mps.current_allocated_memory() / 1e6, "MB")
