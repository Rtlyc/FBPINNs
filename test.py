import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.memory_summary())