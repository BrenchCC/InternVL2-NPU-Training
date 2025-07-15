import torch
import torch_npu
print(torch.npu.is_available())

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)
 
print(z)