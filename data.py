"""固定分层划分；测试集独立加载。像素固定映射到 [-1, 1]。"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DATASET_NAMES = ("mnist", "fashion_mnist", "hwdb10")


def hwdb_metadata():
    path = DATA_ROOT / 'hwdb10' / 'metadata.json'
    if not path.exists():
        raise FileNotFoundError('Run download_hwdb.py and prepare_hwdb.py before using hwdb10.')
    info = json.loads(path.read_text(encoding='utf-8'))
    manifest = path.parent / 'samples.jsonl'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != info['manifest_sha256']:
        raise ValueError('HWDB sample manifest has changed')
    return info


def load_hwdb_partition(split, info):
    path = DATA_ROOT / 'hwdb10' / f'{split}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['array_sha256'][split]:
        raise ValueError(f'HWDB data integrity error: {path}')
    with np.load(path, allow_pickle=False) as arrays:
        images = torch.from_numpy(arrays['images'].copy()).unsqueeze(1)
        labels = torch.from_numpy(arrays['labels'].copy())
    if len(labels) != info['sample_counts'][split]:
        raise ValueError('HWDB sample count mismatch')
    return TensorDataset(images, labels)


def load_raw(name, train=True):
    cls = {"mnist": MNIST, "fashion_mnist": FashionMNIST}[name]
    if name == "fashion_mnist":
        # 数据集作者的官方 GitHub 镜像；torchvision 仍校验原始 MD5。
        cls.mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    return cls(root=str(DATA_ROOT), train=train, download=True)


def stratified_indices(labels, seed=2026, val_fraction=0.1):
    """各类别分别抽取验证样本，划分种子与模型训练种子相互独立。"""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for category in np.unique(labels):
        ids = np.flatnonzero(labels == category)
        rng.shuffle(ids)
        count = max(1, round(len(ids) * val_fraction))
        val_ids.extend(ids[:count].tolist())
        train_ids.extend(ids[count:].tolist())
    return np.array(sorted(train_ids)), np.array(sorted(val_ids))


def prepare_split(name):
    if name == 'hwdb10':
        info = hwdb_metadata()
        info = {**info, 'train_size': info['sample_counts']['train'],
                'val_size': info['sample_counts']['val'],
                'split_sha256': info['manifest_sha256']}
        return load_hwdb_partition('train', info), load_hwdb_partition('val', info), info
    raw = load_raw(name, train=True)
    train_ids, val_ids = stratified_indices(raw.targets.numpy())
    split_file = DATA_ROOT / "splits" / f"{name}_seed2026.npz"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    if split_file.exists():
        with np.load(split_file) as old:
            if not (np.array_equal(old['train'], train_ids) and np.array_equal(old['val'], val_ids)):
                raise ValueError("Saved split disagrees with current data; refusing to change it silently.")
    else:
        np.savez(split_file, train=train_ids, val=val_ids)
    fingerprint = hashlib.sha256(train_ids.tobytes() + val_ids.tobytes()).hexdigest()
    info = {"dataset": name, "split_seed": 2026, "split_sha256": fingerprint,
            "train_size": len(train_ids), "val_size": len(val_ids),
            "preprocessing": "uint8 / 255; bilinear 32x32 antialias; (x - 0.5) / 0.5",
            "classes": list(raw.classes)}
    split_file.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return (TensorDataset(raw.data[train_ids].unsqueeze(1), raw.targets[train_ids]),
            TensorDataset(raw.data[val_ids].unsqueeze(1), raw.targets[val_ids]), info)


def prepare_test(name):
    if name == 'hwdb10':
        info = hwdb_metadata()
        return load_hwdb_partition('test', info), info['classes']
    raw = load_raw(name, train=False)
    return TensorDataset(raw.data.unsqueeze(1), raw.targets), list(raw.classes)


def preprocess(images, device):
    # 放到 GPU 后做批量预处理，减少 Python 逐张处理的开销。
    x = images.to(device, non_blocking=True).float().div_(255)
    x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False, antialias=True)
    return x.sub_(0.5).div_(0.5)
