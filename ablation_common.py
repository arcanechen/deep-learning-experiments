"""控制变量实验的原子存档、随机状态与训练步骤。"""
import contextlib
import csv
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from data import ROOT, preprocess

PLAN_PATH = ROOT / 'reports' / 'ablation_plan.json'
OUT = ROOT / 'reports' / 'ablations'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def exclusive_lock(path):
    """进程退出时由系统释放，避免重复启动两个训练队列。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def rng_state(generator):
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def epoch(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    amp = device.type == 'cuda'
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = attempts = updates = 0
    # post hook 只在真正调用 optimizer.step 后触发，AMP 跳过更新时不会触发。
    def did_step(*_):
        nonlocal updates
        updates += 1
    hook = optimizer.register_step_post_hook(did_step) if training else None
    try:
        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images, labels = preprocess(images, device), labels.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Non-finite loss; stopping without committing this epoch')
                if training:
                    attempts += 1
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                size = len(labels)
                loss_sum += loss.detach() * size
                correct += (logits.argmax(1) == labels).sum()
                count += size
    finally:
        if hook:
            hook.remove()
    return {'loss': loss_sum.item() / count, 'accuracy': correct.item() / count,
            'batches': len(loader), 'attempted_updates': attempts, 'actual_updates': updates}


def write_csv(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def run_name(config_id, seed):
    return f'fashion_ablation_{config_id}_amp_s{seed}'
