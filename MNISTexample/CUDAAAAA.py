import torch
print(torch.cuda.is_available())  # True means CUDA is enabled.
print(torch.cuda.current_device())  # Prints the GPU device index.
print(torch.cuda.get_device_name(0))  # Prints the name of your GPU.
