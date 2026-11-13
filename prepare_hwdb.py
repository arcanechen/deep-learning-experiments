"""从已校验的官方 GNT 构建固定十类子集，严格按书写人隔离。"""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
import struct
import zipfile
import hashlib

import numpy as np
from PIL import Image

from download_hwdb import DEST, FILES, ROOT, sha256_file


def iter_gnt(blob):
    """数值字段为小端，标签的两字节按原 GB 编码次序解码。"""
    offset, index = 0, 0
    while offset < len(blob):
        if len(blob) - offset < 10:
            raise ValueError(f'Truncated GNT header at {offset}')
        size, tag, width, height = struct.unpack_from('<I2sHH', blob, offset)
        if width == 0 or height == 0 or size != 10 + width * height:
            raise ValueError(f'Invalid GNT record at {offset}: {size}, {width}, {height}')
        if offset + size > len(blob):
            raise ValueError(f'Truncated GNT pixels at {offset}')
        yield index, offset, tag, width, height, blob[offset + 10:offset + size]
        offset += size
        index += 1


def writer_id(member):
    match = re.fullmatch(r'(\d+)(?:[-_].*)?', PurePosixPath(member).stem)
    if not match:
        raise ValueError(f'Unrecognized writer filename: {member}')
    return match.group(1)


def split_writers(train_writers, test_writers, seed, fraction):
    train_writers, test_writers = set(train_writers), set(test_writers)
    if train_writers & test_writers:
        raise ValueError('Official train/test writer overlap')
    ids = np.array(sorted(train_writers))
    shuffled = np.random.default_rng(seed).permutation(ids)
    count = max(1, round(len(ids) * fraction))
    val = set(shuffled[:count].tolist())
    return {'train': train_writers - val, 'val': val, 'test': test_writers}


def fit_image(image):
    width, height = image.size
    size = max(width, height)
    margin = max(1, round(size * 0.1))
    canvas = Image.new('L', (size + 2 * margin, size + 2 * margin), color=255)
    canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
    return canvas.resize((64, 64), Image.Resampling.BILINEAR)


def build():
    config_path = ROOT / 'hwdb_selection.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    classes = config['classes']
    if len(classes) != 10 or len(set(classes)) != 10:
        raise ValueError('Exactly 10 unique characters required')
    tags = {char.encode('gb2312'): (index, char) for index, char in enumerate(classes)}
    output = ROOT / 'data' / 'hwdb10'
    if output.exists():
        raise FileExistsError('hwdb10 already exists; keep its fixed split instead of overwriting.')
    sources = json.loads((DEST / 'sources.json').read_text(encoding='utf-8'))
    by_name = {s['file']: s for s in sources}
    writer_members = {'train': [], 'test': []}
    for name in FILES:
        source = by_name[name]
        if not source['zip_crc_verified'] or sha256_file(DEST / name) != source['sha256']:
            raise ValueError(f'Archive hash mismatch: {name}')
        partition = 'test' if name == 'Gnt1.1Test.zip' else 'train'
        with zipfile.ZipFile(DEST / name) as archive:
            for member in sorted(archive.namelist()):
                if member.lower().endswith('.gnt'):
                    writer_members[partition].append((name, member, writer_id(member)))
    train_writers = {x[2] for x in writer_members['train']}
    test_writers = {x[2] for x in writer_members['test']}
    if len(train_writers) != config['official_train_writers_expected'] or len(test_writers) != config['official_test_writers_expected']:
        raise ValueError(f'Unexpected writer counts: {len(train_writers)}, {len(test_writers)}')
    groups = split_writers(train_writers, test_writers, config['split_seed'], config['validation_writer_fraction'])
    staging = output.with_name('hwdb10_build_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    staging.mkdir(parents=True, exist_ok=False)
    arrays, labels, rows = defaultdict(list), defaultdict(list), []
    parsed_counts = Counter()
    raw_duplicates = defaultdict(list)
    for name in FILES:
        print(f'PARSING {name}', flush=True)
        with zipfile.ZipFile(DEST / name) as archive:
            for member in sorted(archive.namelist()):
                if not member.lower().endswith('.gnt'):
                    continue
                writer = writer_id(member)
                split = next(k for k, v in groups.items() if writer in v)
                blob = archive.read(member)  # 再次由 zipfile 验证该成员 CRC
                for index, offset, tag, width, height, pixels in iter_gnt(blob):
                    parsed_counts[name] += 1
                    if tag not in tags:
                        continue
                    label, char = tags[tag]
                    image = Image.frombytes('L', (width, height), pixels)
                    # 图片路径完全自行构造，不把归档成员路径用于解压目标。
                    filename = f'images/{split}/{label:02d}/{writer}_{index:05d}.png'
                    path = staging / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        raise ValueError(f'Duplicate sample identity: {filename}')
                    image.save(path)
                    pixel_hash = hashlib.sha256(struct.pack('<HH', width, height) + pixels).hexdigest()
                    raw_duplicates[pixel_hash].append((split, writer))
                    rows.append({'split': split, 'label': label, 'character': char,
                                 'writer_id': writer, 'official_partition': 'test' if name == 'Gnt1.1Test.zip' else 'train',
                                 'archive': name, 'member': member, 'record_index': index,
                                 'record_offset': offset, 'gb2312_hex': tag.hex(),
                                 'width': width, 'height': height, 'pixel_sha256': pixel_hash,
                                 'png_sha256': sha256_file(path), 'image': filename,
                                 'array_index': len(labels[split])})
                    arrays[split].append(np.asarray(fit_image(image), dtype=np.uint8))
                    labels[split].append(label)
        print(f'{name}: scanned {parsed_counts[name]:,} records; selected so far {len(rows)}', flush=True)
    counts = {s: dict(Counter(labels[s])) for s in groups}
    for split in groups:
        if set(counts[split]) != set(range(10)):
            raise ValueError(f'Missing classes in {split}: {counts[split]}')
        np.savez_compressed(staging / f'{split}.npz',
                            images=np.stack(arrays[split]), labels=np.array(labels[split], dtype=np.int64))
    manifest_path = staging / 'samples.jsonl'
    manifest_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    duplicate_groups = [v for v in raw_duplicates.values() if len(v) > 1]
    cross_split = [v for v in duplicate_groups if len({x[0] for x in v}) > 1]
    if cross_split:
        raise ValueError(f'Identical raw images across splits; review before using: {cross_split[:5]}')
    metadata = {'dataset': 'hwdb10', 'display_name': config['dataset'], 'version': config['version'],
                'classes': classes, 'selection': config, 'selection_sha256': sha256_file(config_path),
                'split_seed': config['split_seed'], 'writer_ids': {k: sorted(v) for k, v in groups.items()},
                'writer_counts': {k: len(v) for k, v in groups.items()},
                'sample_counts': {k: len(v) for k, v in labels.items()},
                'class_counts': counts, 'parsed_record_counts': dict(parsed_counts),
                'duplicate_raw_image_groups': duplicate_groups, 'cross_split_identical_raw_images': len(cross_split),
                'sources': sources, 'manifest_sha256': sha256_file(manifest_path),
                'array_sha256': {k: sha256_file(staging / f'{k}.npz') for k in groups},
                'prepared_images': '64x64; aspect preserved; centered; white padding; original PNGs retained',
                'training_preprocessing': '64x64 uint8 -> /255 -> bilinear 32x32 antialias -> (x-0.5)/0.5',
                'preparation_library_versions': {'numpy': np.__version__, 'pillow': Image.__version__},
                'course_subset_equivalence': False}
    (staging / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    # 为可复现性保留数据准备脚本和选类配置。
    import shutil
    (staging / 'source').mkdir()
    for filename in ('download_hwdb.py', 'prepare_hwdb.py', 'hwdb_selection.json'):
        shutil.copy2(ROOT / filename, staging / 'source' / filename)
    staging.rename(output)
    make_preview(output, classes, rows)
    write_report(output, metadata)
    print(json.dumps({k: metadata[k] for k in ('classes', 'writer_counts', 'sample_counts', 'class_counts', 'cross_split_identical_raw_images')}, ensure_ascii=False, indent=2), flush=True)
    print(f'READY: {output}', flush=True)


def write_report(output, metadata):
    lines = ['# CASIA-HWDB1.1 自选十类子集', '',
             '本子集使用中科院自动化所官方离线单字图像构建。用户已选择采用官方数据自行选类；它不等同于课件的原十类子集，实际样本数按解析结果记录，不人为凑成 2382/601。', '',
             '## 固定类别与实际数量', '',
             '| 标签 | 汉字 | 训练 | 验证 | 测试 |', '|---|---|---:|---:|---:|']
    for label, char in enumerate(metadata['classes']):
        values = [metadata['class_counts'][s][label] for s in ('train', 'val', 'test')]
        lines.append(f'| {label} | {char} | {values[0]} | {values[1]} | {values[2]} |')
    counts = metadata['sample_counts']
    lines.extend([f'| 合计 | | {counts["train"]} | {counts["val"]} | {counts["test"]} |', '',
                  '## 划分与来源', '',
                  '- 类别选择在查看模型效果前固定，配置保存在 `hwdb_selection.json`。',
                  '- 官方 240 位训练书写人按固定种子 2026 划为 216 位训练、24 位验证；官方 60 位测试书写人保持独立。',
                  '- 不按单张图片随机划分汉字数据，避免相同书写人的风格同时出现在训练、验证、测试中。',
                  '- 所有三个压缩包均完成全成员 CRC 检查与本地 SHA-256 记录。SHA-256 是本地完整性指纹，不是发布方提供的签名。',
                  '- `data/hwdb10/samples.jsonl` 为每张图片记录标签、书写人、归档文件、GNT 记录序号/偏移、像素与 PNG 哈希。',
                  '- 原始灰度 PNG 保存在 `data/hwdb10/images`；预处理缓存为三个独立 NPZ 文件。',
                  '- `data/hwdb10/metadata.json` 保存实际类别数、样本数、书写人清单和文件指纹。', '',
                  '## 图像处理', '',
                  '保留官方灰度与白底黑字。原图保持长宽比，放在白色正方形画布中央，四周增加约 10% 边距后缩放为 64×64。原始尺寸图片单独保留。现有五模型的共同训练入口会再缩放至 32×32，并按固定公式映射到 [-1,1]。', '',
                  '这与课件示例直接缩放到 64×64 有差异，应在最终报告中注明。将来如果研究 32/64 分辨率，应作为单独变量，不能混用后直接比较。', '',
                  '## 复现与使用', '', '```powershell',
                  "& 'D:\\Anaconda\\envs\\ai-lab\\python.exe' download_hwdb.py",
                  "& 'D:\\Anaconda\\envs\\ai-lab\\python.exe' prepare_hwdb.py",
                  "& 'D:\\Anaconda\\envs\\ai-lab\\python.exe' verify_hwdb.py",
                  "& 'D:\\Anaconda\\envs\\ai-lab\\python.exe' train.py --dataset hwdb10 --model lenet --epochs 10 --seed 42",
                  '```', '', '已有子集不会被准备脚本覆盖，以保持实验划分固定。下载脚本可续传和复核已有原始包。', '',
                  '独立核验结果见 `data/hwdb10/verification.json`（运行 verify_hwdb.py 后生成）。数据准备完成不代表已完成五模型正式训练。', '',
                  '## 学习重点', '',
                  '1. 标签 0～9 只是模型使用的索引，必须保留与汉字的映射。',
                  '2. 对书写人泛化的评估需要按人划分；单纯随机打乱图片可能低估遇到新笔迹时的误差。',
                  '3. 测试集只用来评价已确定的实验设置，不能根据测试集表现换类、换划分或调参。',
                  '4. 原始图片、模型输入缓存和样本清单分别承担可追溯、效率和复现的作用。', '',
                  '[官方下载页](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)；[GNT 格式说明](https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html)。', '',
                  '引用：C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang. CASIA online and offline Chinese handwriting databases. ICDAR, 2011, pp. 37–41.'])
    (ROOT / 'docs' / 'HWDB十类子集说明.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def make_preview(output, classes, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    train_rows = [r for r in rows if r['split'] == 'train']
    fig, axes = plt.subplots(10, 5, figsize=(8, 14), constrained_layout=True)
    for label, char in enumerate(classes):
        sample_rows = [r for r in train_rows if r['label'] == label][:5]
        for ax, row in zip(axes[label], sample_rows):
            with Image.open(output / row['image']) as img:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'{label}: {char} / {row["writer_id"]}', fontsize=10)
            ax.axis('off')
    fig.suptitle('CASIA-HWDB1.1 自选十类：训练集每类前五个样本', fontsize=14)
    fig.savefig(output / 'class_preview.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    build()
