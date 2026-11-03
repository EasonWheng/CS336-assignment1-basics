# get_batch

import numpy.typing as npt
import numpy as np
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    接收 NumPy 数组 x（存 token ID 的整数数组）、batch_size、context_length 以及 PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），
    返回一对张量：采样得到的输入序列，以及对应的下一 token 目标。
    两个张量的形状都应为 (batch_size,context_length)，包含 token ID，并放在指定设备上。

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)
    x = torch.as_tensor(dataset[idx], device=device).long()
    y = torch.as_tensor(dataset[idx + 1], device=device).long()
    return x, y
