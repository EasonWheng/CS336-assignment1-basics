# save_checkpoint, load_checkpoint
"""
检查点应包含恢复训练所需的全部状态。至少需要能恢复模型权重。如果使用有状态优化器（如
AdamW），还需要保存优化器状态（例如 AdamW 的矩估计）。最后，要恢复学习率调度，还需要知
道停止时的迭代次数。PyTorch 让保存这些都很容易：每个 nn.Module 都有 state_dict() 方法，返回
包含全部可学习权重的字典，之后可以用配套的 load_state_dict() 恢复；torch.optim.Optimizer 同
理。此外，torch.save(obj, dest) 可以把对象（例如一个字典，值里既有张量也有整数这类普通
Python 对象）转储到文件（路径）或类文件对象，之后用 torch.load(src) 载回内存。
"""
import os
from typing import IO, BinaryIO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    把模型、优化器和迭代次数中的全部状态转储到类文件对象 out。
    可以用模型和优化器的state_dict 方法取得相关状态，用 torch.save(obj, out) 把 obj 写入 out（PyTorch 这里支持路径或类文件对象）。
    常见做法是让 obj 为字典，但只要之后能加载，格式随意。

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    dict_to_save = {
        "model": model.state_dict(),        # 注意这里存的时候一定要state_dict，读取的时候也要用loadstatedict才完成一次存取读取
        "optimizer": optimizer.state_dict(),    # 如果直接存model读取的时候是会报错的TYPEERROR
        "iteration": iteration,
    }
    torch.save(dict_to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    从 src（路径或类文件对象）加载检查点，并从中恢复模型和优化器状态。函数应返回检查点中保存的迭代次数。
    可以用 torch.load(src) 取回 save_checkpoint 中保存的内容，用模型和优化器的 load_state_dict 方法把它们恢复到之前的状态。

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    dict_loaded = torch.load(src)
    model.load_state_dict(dict_loaded["model"])
    optimizer.load_state_dict(dict_loaded["optimizer"])
    return dict_loaded["iteration"]
