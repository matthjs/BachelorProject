import pynvml
import torch.cuda


def get_gpu_memory_usage():
    """
    Assumes only one GPU is available.
    Uses the NVIDIA Management Library to return GPU memory usage information.
    :return: Returns the total amount of VRAM available on the GPU and the VRAM usage.
    """
    torch.cuda.synchronize()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you have only one GPU
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total  # in bytes
    used_memory = mem_info.used  # in bytes
    return total_memory / (1024 ** 3), used_memory / (1024 ** 3)  # Convert bytes to gigabytes