import os
import torch
import numpy as np
import argparse
import yaml

from cs336_basics.model import TransformerLM, build_ablation_model
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, learning_rate_schedule
from cs336_basics.data import get_batch
from cs336_basics.serialization import save_checkpoint, load_checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def plot_losses(train_history, val_history, log_interval, eval_interval,
                out_path="loss_curve.png", start_iteration=0):
    train_steps = [start_iteration + (i + 1) * log_interval
                   for i in range(len(train_history))]
    val_steps = [start_iteration + (i + 1) * eval_interval
                 for i in range(len(val_history))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_steps, train_history, label="train loss")
    ax.plot(val_steps, val_history, label="val loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("TinyStories training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


REQUIRED_GROUPS = ("model", "optimizer", "data", "run", "checkpoint")

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"configs/tinystories.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ablation", type=str, default=None, choices=["layer_norm_ablation",
                        "pre_norm_ablation", "NoPE", "swiglu_ablation"])
    return parser.parse_args(argv)


def load_config(path, args):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"配置文件 {path} 不是映射结构")

    missing = [group for group in REQUIRED_GROUPS if group not in raw]
    if missing:
        raise ValueError(f"配置文件 {path} 缺少分组: {', '.join(missing)}")

    merged = {}
    sources = {}
    duplicates = {}
    for group in REQUIRED_GROUPS:
        section = raw[group]
        if not isinstance(section, dict):
            raise ValueError(f"配置分组 {group} 不是映射结构")
        for key, value in section.items():
            if key in merged:
                duplicates.setdefault(key, [sources[key]]).append(group)
            merged[key] = value
            sources[key] = group

    if duplicates:
        detail = "; ".join(f"{key} 同时出现在 {' 和 '.join(groups)}"
                           for key, groups in duplicates.items())
        raise ValueError(f"配置文件 {path} 存在跨分组重名键: {detail}")

    config = argparse.Namespace(**merged)

    if args.resume is not None:
        config.resume = args.resume
    if args.device is not None:
        config.device = args.device
    config.verbose = bool(args.verbose) or bool(getattr(config, "verbose", False))

    return config


def validate_config(config):
    errors = []

    def is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    positive_ints = [
        ("vocab_size", "model.vocab_size"),
        ("context_length", "model.context_length"),
        ("d_model", "model.d_model"),
        ("num_layers", "model.num_layers"),
        ("num_heads", "model.num_heads"),
        ("d_ff", "model.d_ff"),
        ("total_steps", "optimizer.total_steps"),
        ("batch_size", "data.batch_size"),
        ("eval_batches", "run.eval_batches"),
        ("log_interval", "run.log_interval"),
        ("eval_interval", "run.eval_interval"),
        ("checkpoint_interval", "checkpoint.checkpoint_interval"),
    ]
    for attr, label in positive_ints:
        value = getattr(config, attr, None)
        if not (is_int(value) and value > 0):
            errors.append(f"{label} 必须是正整数, 当前为 {value!r}")

    value = getattr(config, "warmup_steps", None)
    if not (is_int(value) and value >= 0):
        errors.append(f"optimizer.warmup_steps 必须是非负整数, 当前为 {value!r}")

    value = getattr(config, "seed", None)
    if not (is_int(value) and value >= 0):
        errors.append(f"run.seed 必须是非负整数, 当前为 {value!r}")

    positive_numbers = [
        ("max_lr", "optimizer.max_lr"),
        ("eps", "optimizer.eps"),
        ("rope_theta", "model.rope_theta"),
        ("grad_clip", "optimizer.grad_clip"),
    ]
    for attr, label in positive_numbers:
        value = getattr(config, attr, None)
        if not (is_number(value) and value > 0):
            errors.append(f"{label} 必须是正数, 当前为 {value!r}")

    nonnegative_numbers = [
        ("min_lr", "optimizer.min_lr"),
        ("weight_decay", "optimizer.weight_decay"),
    ]
    for attr, label in nonnegative_numbers:
        value = getattr(config, attr, None)
        if not (is_number(value) and value >= 0):
            errors.append(f"{label} 必须是非负数, 当前为 {value!r}")

    betas = getattr(config, "betas", None)
    if not (isinstance(betas, (list, tuple)) and len(betas) == 2
            and all(is_number(b) and 0.0 <= b < 1.0 for b in betas)):
        errors.append(f"optimizer.betas 必须是两个 [0, 1) 区间的数, 当前为 {betas!r}")

    device = getattr(config, "device", None)
    if device not in ("cpu", "cuda"):
        errors.append(f"data.device 必须是 'cpu' 或 'cuda', 当前为 {device!r}")
    elif device == "cuda":
        try:
            import torch
        except ImportError:
            errors.append("data.device 为 'cuda', 但当前环境无法导入 torch")
        else:
            if not torch.cuda.is_available():
                errors.append("data.device 为 'cuda', 但 torch.cuda.is_available() 返回 False")

    for attr, label in (("train_data", "data.train_data"), ("val_data", "data.val_data")):
        value = getattr(config, attr, None)
        if not isinstance(value, str) or not value:
            errors.append(f"{label} 必须是非空路径字符串, 当前为 {value!r}")
        elif not os.path.isfile(value):
            errors.append(f"{label} 指向的文件不存在: {value}")

    value = getattr(config, "checkpoint_dir", None)
    if not isinstance(value, str) or not value:
        errors.append(f"checkpoint.checkpoint_dir 必须是非空路径字符串, 当前为 {value!r}")

    resume = getattr(config, "resume", None)
    if resume is not None:
        if not isinstance(resume, str) or not resume:
            errors.append(f"checkpoint.resume 必须是路径字符串或 null, 当前为 {resume!r}")
        elif not os.path.isfile(resume):
            errors.append(f"checkpoint.resume 指向的检查点不存在: {resume}")

    d_model = getattr(config, "d_model", None)
    num_heads = getattr(config, "num_heads", None)
    if is_int(d_model) and is_int(num_heads) and num_heads > 0 and d_model > 0:
        if d_model % num_heads != 0:
            errors.append(f"model.d_model ({d_model}) 必须能被 model.num_heads ({num_heads}) 整除")

    max_lr = getattr(config, "max_lr", None)
    min_lr = getattr(config, "min_lr", None)
    if is_number(max_lr) and is_number(min_lr) and min_lr > max_lr:
        errors.append(f"optimizer.min_lr ({min_lr}) 不能大于 optimizer.max_lr ({max_lr})")

    warmup_steps = getattr(config, "warmup_steps", None)
    total_steps = getattr(config, "total_steps", None)
    if is_int(warmup_steps) and is_int(total_steps) and warmup_steps >= total_steps:
        errors.append(f"optimizer.warmup_steps ({warmup_steps}) 必须小于 optimizer.total_steps ({total_steps})")

    if errors:
        raise ValueError("配置校验失败:\n" + "\n".join(f"  - {msg}" for msg in errors))

    return config


def load_datasets(config):
    train_dataset = np.load(config.train_data, mmap_mode="r")
    val_dataset = np.load(config.val_data, mmap_mode="r")
    return train_dataset, val_dataset


def build_model(config):
    model = TransformerLM(config.d_model,
                          config.num_heads,
                          config.d_ff,
                          config.vocab_size,
                          config.context_length,
                          config.num_layers,
                          config.rope_theta,
                          config.device)
    return model


def build_optimizer(model, config):
    optimizer = AdamW(model.parameters(),
                      config.max_lr,
                      config.betas,
                      config.eps,
                      config.weight_decay)
    return optimizer


def save_checkpoint_if_due(model, optimizer, iteration, config):
    path = os.path.join(config.checkpoint_dir, f"checkpoint_{iteration}.pt")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    save_checkpoint(model, optimizer, iteration, path)

    return path


def load_checkpoint_if_needed(model, optimizer, config):
    if config.resume is None:
        return 0
    path = config.resume
    return load_checkpoint(path, model, optimizer)


def evaluate(model, val_data, config):
    batch_size = config.batch_size
    context_length = config.context_length
    device = config.device

    model.eval()
    with torch.no_grad():
        loss = 0
        for _ in range(config.eval_batches):
            input_tensor, target_tensor = get_batch(val_data, batch_size, context_length, device)
            logits = model(input_tensor)
            loss += cross_entropy(logits, target_tensor).item()
        average_val_loss = loss / config.eval_batches
    model.train()

    return average_val_loss


def train(config,
          ablation_type=None):
    run = None
    if getattr(config, "wandb_enabled", False):
        run = wandb.init(
            project=getattr(config, "wandb_project", "cs336-a1"),
            name=getattr(config, "run_name", None),
            config=vars(config),
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_dataset, val_dataset = load_datasets(config)
    model = (
        build_model(config) if ablation_type is None
        else build_ablation_model(ablation_type, config)
    )
    model.compile()

    optimizer = build_optimizer(model, config)
    iteration = load_checkpoint_if_needed(model, optimizer, config) # 如果有的话这里会恢复model
    device = config.device

    train_loss_acc = 0
    last_train_loss = None
    train_loss_history = []
    last_val_loss = None
    val_loss_history = []
    paths = []
    iteration_i = 0

    model.train()
    for i in range(iteration, config.total_steps):
        lr = learning_rate_schedule(i, config.max_lr, config.min_lr, config.warmup_steps, config.total_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        input_seq, target_seq = get_batch(train_dataset, config.batch_size, config.context_length, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_seq)
        loss = cross_entropy(logits, target_seq)
        loss.backward()     # 计算梯度
        gradient_clipping(model.parameters(), config.grad_clip)
        optimizer.step()
        last_train_loss = loss.detach()
        train_loss_acc += last_train_loss

        iteration_i += 1

        if  (i + 1) % config.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, config)
            print(f"[ val/验证 ] iteration {i + 1} | val_loss = {val_loss:.4f}", flush=True)
            val_loss_history.append(val_loss)
            last_val_loss = val_loss
            if run is not None:
                wandb.log({"val/loss": val_loss}, step=i + 1)

        if (i + 1) % config.checkpoint_interval == 0:
            paths.append(save_checkpoint_if_due(model, optimizer, i + 1, config))

        if (i + 1) % config.log_interval == 0:
            average_train_loss = (train_loss_acc / iteration_i).item()
            train_loss_history.append(average_train_loss)
            # 打印日志
            if run is not None:
                wandb.log(
                    {"train/loss": average_train_loss, "lr": lr, "iteration": i + 1},
                    step=i + 1,
                )
            print(f"[ train/训练 ] 该 log 区间(iteration {i + 1})的平均训练 train_loss: {average_train_loss}", flush=True)
            train_loss_acc, iteration_i = 0, 0

    path = save_checkpoint_if_due(model, optimizer, config.total_steps, config)
    if run is not None:
        wandb.finish()

    last_train_loss = last_train_loss.item() if last_train_loss is not None else None

    return {
        "iteration": config.total_steps,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "train_history": train_loss_history,
        "val_history": val_loss_history,
        "last_checkpoint": path,
        "paths": paths
    }


def main():
    args = parse_args()
    config = load_config(args.config, args)
    validate_config(config)

    fnl_log = train(config, args.ablation)
    plot_losses(
        fnl_log["train_history"],
        fnl_log["val_history"],
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        out_path="loss_curve.png",
    )

    return 0


if __name__ == "__main__":
    main()
