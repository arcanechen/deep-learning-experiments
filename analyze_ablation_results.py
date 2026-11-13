"""全部控制实验核验完成后，生成逐种子差值、训练曲线和计算量分析素材。"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from controlled_cnn import ControlledCNN

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports' / 'ablations'


def macs(depth, width):
    model = ControlledCNN(depth, width).eval()
    total = 0
    handles = []
    def count(layer, inputs, output):
        nonlocal total
        if isinstance(layer, nn.Conv2d):
            total += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, nn.Linear):
            total += output.numel() * layer.in_features
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    with torch.inference_mode():
        model(torch.zeros(1, 1, 32, 32))
    for handle in handles:
        handle.remove()
    return total


def main():
    torch.set_num_threads(2)
    check = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    if check['verified_runs'] != 33 or check['missing']:
        raise ValueError('Complete and verify all 33 runs before interpreting results')
    aggregate = json.loads((OUT / 'aggregate.json').read_text(encoding='utf-8'))
    with (OUT / 'individual_runs.csv').open(encoding='utf-8-sig') as f:
        individual = list(csv.DictReader(f))
    by_seed = {(r['config'], int(r['seed'])): r for r in individual}
    seeds = (42, 43, 44)
    lines = ['# 控制变量实验：进一步核对与分析素材', '',
             '全部数值来自已核验的运行记录。逐种子差值用于观察一致性；三次训练不足以支持稳健的显著性检验。', '',
             '## 逐种子准确率', '', '| 配置 | seed 42 | seed 43 | seed 44 | 平均值 ± 样本 SD |', '|---|---:|---:|---:|---:|']
    for r in aggregate:
        values = [float(by_seed[r['id'], seed]['accuracy']) * 100 for seed in seeds]
        np.testing.assert_allclose(np.mean(values), 100*r['accuracy_mean'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.std(values, ddof=1), 100*r['accuracy_std'], rtol=0, atol=1e-10)
        lines.append(f'| {r["id"]} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | '
                     f'{100*r["accuracy_mean"]:.2f} ± {100*r["accuracy_std"]:.2f}% |')
    comparisons = []
    for group, reference in [('depth', 'd2_w32_b128'), ('width', 'd2_w32_b128'),
                             ('batch', 'd2_w32_b128'), ('matched_parameters', 'd2_w27_b128')]:
        lines += ['', f'## {group}：相对 {reference} 的差值', '',
                  '| 配置 | seed 42 | seed 43 | seed 44 | 平均差 |', '|---|---:|---:|---:|---:|']
        for r in aggregate:
            if group not in r['groups'] or r['id'] == reference:
                continue
            difference = [(float(by_seed[r['id'], seed]['accuracy']) -
                           float(by_seed[reference, seed]['accuracy'])) * 100 for seed in seeds]
            comparisons.append({'group': group, 'config': r['id'], 'reference': reference,
                                'paired_seed_differences_pp': difference,
                                'mean_difference_pp': float(np.mean(difference))})
            lines.append(f'| {r["id"]} | {difference[0]:+.2f} | {difference[1]:+.2f} | {difference[2]:+.2f} | {np.mean(difference):+.2f} |')
    lines += ['', '差值单位为百分点。同种子不意味着不同结构的初始化参数逐项相同；只是沿用预先配对的种子编号。', '',
              '## 参数、计算量与实测成本', '',
              '| 配置 | 卷积层数 | 参数量 | 单图 Conv/Linear MACs | 平均训练+验证秒数 | 平均峰值分配显存 MiB |',
              '|---|---:|---:|---:|---:|---:|']
    profiles = []
    for r in aggregate:
        operations = macs(r['depth_per_stage'], r['width'])
        profiles.append({'config': r['id'], 'parameters': r['parameters'], 'conv_linear_macs': operations})
        lines.append(f'| {r["id"]} | {r["conv_layers"]} | {r["parameters"]:,} | {operations:,} | '
                     f'{r["compute_seconds_mean"]:.1f} | {r["peak_allocated_mb_mean"]:.1f} |')
    lines += ['', 'MACs 只统计 32×32 单图前向的卷积和全连接乘加，不包含 BN、激活、池化、反向传播、优化器或检查点写盘；不能用它直接推算运行秒数。', '',
              '## 实际更新与 AMP 跳过次数', '',
              '| 配置 | 计划更新 | seed 42 跳过 | seed 43 跳过 | seed 44 跳过 |', '|---|---:|---:|---:|---:|']
    for r in aggregate:
        skipped = [int(by_seed[r['id'], s]['attempted_updates']) - int(by_seed[r['id'], s]['actual_updates']) for s in seeds]
        assert all(value >= 0 for value in skipped)
        lines.append(f'| {r["id"]} | {r["planned_optimizer_steps"]} | {skipped[0]} | {skipped[1]} | {skipped[2]} |')
    lines += ['', '跳过更新说明 GradScaler 没有调用该 batch 的 optimizer.step。已记录的跳过次数属于本轮训练条件，不应隐藏，也不能单独归因于模型容量。', '',
              '## 训练与验证曲线', '',
              '下图展示共同参照配置 d2_w32_b128 的三个种子。星号标记最低验证损失的轮次。训练准确率是在一轮内持续更新参数时累计，验证准确率是在轮末计算，两者不是严格同一状态下的泛化差距。', '',
              '![共同参照配置的训练曲线](reference_learning_curves.png)']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for column, seed in enumerate(seeds):
        folder = ROOT / 'runs' / f'fashion_ablation_d2_w32_b128_amp_s{seed}'
        with (folder / 'history.csv').open(encoding='utf-8-sig') as f:
            history = list(csv.DictReader(f))
        epochs = [int(h['epoch']) for h in history]
        best = min(history, key=lambda h: float(h['val_loss']))
        for row, metric in enumerate(('loss', 'accuracy')):
            ax = axes[row, column]
            for part in ('train', 'val'):
                ax.plot(epochs, [float(h[f'{part}_{metric}']) for h in history], label=part)
            ax.scatter([int(best['epoch'])], [float(best[f'val_{metric}'])], marker='*', s=100, color='black', label='selected')
            ax.set(xlabel='Epoch', ylabel=metric, title=f'seed {seed}' if row == 0 else '')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Reference configuration: d2_w32_b128; selection by validation loss')
    fig.savefig(OUT / 'reference_learning_curves.png', dpi=150)
    plt.close(fig)
    (OUT / 'detailed_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'detailed_analysis.json').write_text(json.dumps({'seed_differences': comparisons, 'profiles': profiles}, indent=2), encoding='utf-8')
    print('Wrote detailed_analysis.md, detailed_analysis.json and reference_learning_curves.png')


if __name__ == '__main__':
    main()
