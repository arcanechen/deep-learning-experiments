"""只使用训练集和验证集。正式测试另运行 evaluate.py。"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from data import ROOT, DATASET_NAMES, prepare_split, preprocess
from models import MODEL_NAMES, build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = preprocess(images, device)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            size = labels.size(0)
            total_loss += loss.detach() * size
            total_correct += (logits.argmax(1) == labels).sum()
            count += size
    return {'loss': total_loss.item() / count, 'accuracy': total_correct.item() / count}


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ('loss', 'accuracy')):
        for part in ('train', 'val'):
            ax.plot([r['epoch'] for r in history], [r[f'{part}_{metric}'] for r in history],
                    marker='o', markersize=3, label=part)
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASET_NAMES, default='fashion_mnist')
    parser.add_argument('--model', choices=MODEL_NAMES, default='lenet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--amp', action='store_true', help='CUDA automatic mixed precision with gradient scaling')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check environment or explicitly use --device cpu.')
    if args.amp and args.device != 'cuda':
        parser.error('--amp requires CUDA')
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(args.device)
    run_name = args.run_name or f'{args.dataset}_{args.model}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}'
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        parser.error('run-name must be a directory name, not a path')
    run_dir = ROOT / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args)
    config.update({'optimizer': 'Adam', 'weight_decay': 0.0, 'input_size': 32,
                   'deterministic': True, 'precision': 'amp_float16' if args.amp else 'float32',
                   'selection': 'minimum validation cross-entropy',
                   'python': platform.python_version(), 'torch': torch.__version__,
                   'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None})
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    source_dir = run_dir / 'source'
    source_dir.mkdir()
    for name in ('train.py', 'data.py', 'models.py', 'evaluate.py',
                 'prepare_hwdb.py', 'download_hwdb.py', 'hwdb_selection.json'):
        shutil.copy2(ROOT / name, source_dir / name)
    train_data, val_data, split_info = prepare_split(args.dataset)
    (run_dir / 'data_info.json').write_text(json.dumps(split_info, indent=2), encoding='utf-8')
    pin = device.type == 'cuda'
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    model = build_model(args.model).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (run_dir / 'model.txt').write_text(str(model) + f'\nTrainable parameters: {params}\n', encoding='utf-8')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    history, best_loss, best_epoch = [], float('inf'), 0
    if pin:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    print(f'RUN_DIR={run_dir}\nParameters={params:,}; train={len(train_data)}; val={len(val_data)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, args.amp)
        val_metrics = run_epoch(model, val_loader, device, amp=args.amp)
        if pin:
            torch.cuda.synchronize()
        record = {'epoch': epoch,
                  **{f'train_{k}': v for k, v in train_metrics.items()},
                  **{f'val_{k}': v for k, v in val_metrics.items()},
                  'seconds': time.perf_counter() - epoch_start}
        history.append(record)
        with (run_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(history)
        if val_metrics['loss'] < best_loss:
            best_loss, best_epoch = val_metrics['loss'], epoch
            torch.save({'model': args.model, 'dataset': args.dataset, 'state_dict': model.state_dict(),
                        'epoch': epoch, 'val_loss': best_loss, 'amp': args.amp}, run_dir / 'best.pt')
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4%} val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4%} seconds={record['seconds']:.1f}", flush=True)
    summary = {'best_epoch': best_epoch, 'best_val_loss': best_loss,
               'best_val_accuracy': history[best_epoch - 1]['val_accuracy'],
               'parameters': params, 'training_seconds': time.perf_counter() - start,
               'peak_allocated_mb': torch.cuda.max_memory_allocated() / 2**20 if pin else None,
               'test_evaluated': False}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_history(history, run_dir / 'learning_curves.png')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
