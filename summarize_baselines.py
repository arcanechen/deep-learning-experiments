"""从实际预测和日志生成基础实验汇总，同时核对指标和可比性。"""
import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_baselines import ROOT, EPOCHS, MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('fp32', 'amp'), default='fp32')
    args = parser.parse_args()
    suffix = '_amp' if args.precision == 'amp' else ''
    records = []
    error_sections = ['# 基础实验错误分析', '',
                      '以下由独立测试预测计算，用于描述固定实验结果。每个组合列出最多的三种误判方向，错误率以该真实类别的测试样本数为分母。原因仍需结合原图判断。', '']
    fingerprints = {}
    missing = []
    for dataset in EPOCHS:
        for model in MODELS:
            path = ROOT / 'runs' / f'{dataset}_{model}_baseline{suffix}_s42'
            if not (path / 'test_metrics.json').exists():
                missing.append(f'{dataset}/{model}')
                continue
            read_json = lambda name: json.loads((path / name).read_text(encoding='utf-8'))
            config, summary, metrics, data = [read_json(name) for name in
                  ('config.json', 'summary.json', 'test_metrics.json', 'data_info.json')]
            with (path / 'test_predictions.csv').open(encoding='utf-8') as f:
                predictions = list(csv.DictReader(f))
            truth = np.array([int(r['true_label']) for r in predictions])
            pred = np.array([int(r['predicted_label']) for r in predictions])
            cm = np.zeros((10, 10), dtype=int)
            np.add.at(cm, (truth, pred), 1)
            stored_cm = np.loadtxt(path / 'confusion_matrix.csv', delimiter=',', dtype=int)
            np.testing.assert_array_equal(cm, stored_cm)
            categories = data['classes']
            off_diagonal = cm.copy()
            np.fill_diagonal(off_diagonal, 0)
            top_errors = np.argsort(off_diagonal.ravel())[::-1][:3]
            error_sections.extend([f'## {dataset} / {model}', '',
                                   '| 真实类别 | 预测类别 | 错误张数 | 占真实类别比例 |',
                                   '|---|---|---:|---:|'])
            for flat_index in top_errors:
                actual, predicted = np.unravel_index(flat_index, cm.shape)
                count = int(cm[actual, predicted])
                if count:
                    error_sections.append(f'| {categories[actual]} | {categories[predicted]} | {count} | {count / cm[actual].sum():.2%} |')
            error_sections.extend(['', f'图片与完整混淆矩阵：`{path}`。', ''])
            assert len(predictions) == metrics['test_size']
            assert np.isclose((truth == pred).mean(), metrics['accuracy'])
            denominator = cm.sum(axis=0) + cm.sum(axis=1)
            f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(10), where=denominator != 0).mean()
            assert np.isclose(f1, metrics['macro_f1'])
            with (path / 'history.csv').open(encoding='utf-8') as f:
                history = list(csv.DictReader(f))
            best_epoch = int(min(history, key=lambda r: float(r['val_loss']))['epoch'])
            assert best_epoch == summary['best_epoch'] == metrics['checkpoint_epoch']
            assert len(history) == EPOCHS[dataset] == config['epochs']
            assert config['seed'] == 42 and config['batch_size'] == 128 and config['lr'] == 0.001
            assert config['optimizer'] == 'Adam' and config['input_size'] == 32
            assert config['precision'] == ('amp_float16' if args.precision == 'amp' else 'float32')
            assert np.isclose(float(history[best_epoch - 1]['val_loss']), summary['best_val_loss'])
            assert np.isclose(float(history[best_epoch - 1]['val_accuracy']), summary['best_val_accuracy'])
            old = fingerprints.setdefault(dataset, data['split_sha256'])
            assert old == data['split_sha256'], 'Different data splits within one dataset'
            records.append({'dataset': dataset, 'model': model, 'epochs': config['epochs'],
                            'best_epoch': best_epoch, 'parameters': summary['parameters'],
                            'train_size': data['train_size'], 'val_size': data['val_size'],
                            'test_size': metrics['test_size'], 'val_accuracy': summary['best_val_accuracy'],
                            'best_val_loss': summary['best_val_loss'],
                            'last_train_accuracy': float(history[-1]['train_accuracy']),
                            'last_val_accuracy': float(history[-1]['val_accuracy']),
                            'last_val_loss': float(history[-1]['val_loss']),
                            'test_accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
                            'training_seconds': summary['training_seconds'],
                            'peak_allocated_mb': summary['peak_allocated_mb'],
                            'run_dir': str(path)})
    out = ROOT / 'reports' / args.precision
    out.mkdir(parents=True, exist_ok=True)
    if not records:
        raise RuntimeError('No complete baseline results')
    with (out / 'baseline_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (out / 'baseline_verification.json').write_text(json.dumps(
        {'verified_runs': len(records), 'missing_runs': missing,
         'checks': ['predictions -> accuracy', 'predictions -> Macro-F1', 'confusion matrices',
                    'validation loss -> selected epoch', 'fixed recipe', 'same data split within each dataset']},
        indent=2), encoding='utf-8')
    names = {'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST', 'hwdb10': 'HWDB10 (self-selected)'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, dataset in zip(axes, EPOCHS):
        selected = [r for r in records if r['dataset'] == dataset]
        values = [r['test_accuracy'] * 100 for r in selected]
        bars = ax.bar([r['model'] for r in selected], values, color='#2F6B9A')
        ax.set(title=names[dataset], ylabel='Test accuracy (%)', ylim=(0, 105))
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('Fixed-recipe baseline comparison (one training seed)', fontsize=14)
    fig.savefig(out / 'baseline_accuracy.png', dpi=160)
    plt.close(fig)
    lines = ['# 实验一：基础模型对比', '',
             f'已完成 {len(records)}/15 个基础组合，各一个随机种子。全部数值来自本机真实训练与独立测试。', '',
             '![三个数据集的测试准确率](baseline_accuracy.png)', '',
             '指标表见 [CSV](baseline_summary.csv)，主要误判方向见 [错误分析](baseline_error_analysis.md)。', '',
             '## 实验协议', '',
             f'- 所有模型从随机初始化训练，固定 seed=42、Adam、lr=0.001、batch=128、精度 {args.precision}，使用 32×32 灰度输入。AMP 表示 FP16 自动混合精度及梯度缩放，参数与优化器状态仍保持 FP32；验证与测试也使用同一 autocast 设置。',
             '- MNIST 和 Fashion-MNIST 训练 10 轮；HWDB10 训练 50 轮。同一数据集内部预算相同，跨数据集的训练耗时不可直接比较。',
             '- 以最低验证交叉熵选权重，再对相应测试集评估一次；没有依据测试结果调整本轮配置。',
             '- MNIST 分层抽样逐类取整后为 54001/5999/10000；Fashion-MNIST 为 54000/6000/10000；HWDB10 为 2159/239/597。',
             '- HWDB10 为官方数据自行选择的十类，按书写人划分，不是课件原始十类子集。', '',
             '## 实测结果', '',
             '| 数据集 | 模型 | 参数量 | 选中轮数 | 验证准确率 | 测试准确率 | Macro-F1 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["dataset"]} | {r["model"]} | {r["parameters"]:,} | {r["best_epoch"]} | {r["val_accuracy"]:.2%} | {r["test_accuracy"]:.2%} | {r["macro_f1"]:.4f} |')
    lines.extend(['', '## 训练过程：最佳轮次与最后一轮', '',
                  '| 数据集 | 模型 | 最低验证损失 | 最后一轮验证损失 | 最后一轮训练准确率 | 最后一轮验证准确率 |',
                  '|---|---|---:|---:|---:|---:|'])
    for r in records:
        lines.append(f'| {r["dataset"]} | {r["model"]} | {r["best_val_loss"]:.4f} | {r["last_val_loss"]:.4f} | {r["last_train_accuracy"]:.2%} | {r["last_val_accuracy"]:.2%} |')
    lines.extend(['', '最后一轮不一定是选中轮次。训练准确率是在一轮内参数持续更新时累计的，验证准确率使用轮末固定权重；对于 Dropout/BN 模型还存在训练/评估模式差异，所以这个差距不能单独视为纯粹的泛化差距。验证损失回升需要结合完整曲线判断。'])
    stats_path = ROOT / 'reports' / 'model_stats.json'
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        lines.extend(['', '## 参数与计算量', '',
                      '| 模型 | 参数量 | 单张图像 Conv/Linear MACs |', '|---|---:|---:|'])
        for item in stats['models']:
            lines.append(f'| {item["model"]} | {item["parameters"]:,} | {item["conv_linear_macs_per_image"]:,} |')
        lines.extend(['', '这些数值来自实际适配模型的 32×32 输入，不是原始 ImageNet 模型的论文数值。MACs 只统计卷积和全连接，不包含 BN、激活、池化、反向传播及优化器，也不等于实测耗时。', '',
                      'GoogLeNet 参数少于 AlexNet，但在本实验的输入与下采样设置下乘加次数更多。卷积核会在多个空间位置重复使用，因此参数量、计算量、显存和速度需要分别观察。'])
    mnist_records = [r for r in records if r['dataset'] == 'mnist']
    if mnist_records:
        lines.extend(['', '## 本机成本记录（MNIST，同为 10 轮）', '',
                      '| 模型 | 总耗时（秒） | PyTorch 峰值已分配显存（MiB） |', '|---|---:|---:|'])
        for r in mnist_records:
            lines.append(f'| {r["model"]} | {r["training_seconds"]:.1f} | {r["peak_allocated_mb"]:.1f} |')
        lines.extend(['', '耗时包括训练、验证和运行中的日志/最佳权重保存，不包括数据下载；每项只测一次，可能受系统负载影响。显存只统计 PyTorch 已分配内存，不代表整张 GPU 的占用。完整三套数据的成本记录见 CSV。'])
    lines.extend(['', '## 初步观察', ''])
    for dataset in EPOCHS:
        subset = [r for r in records if r['dataset'] == dataset]
        if subset:
            best_score = max(r['test_accuracy'] for r in subset)
            winners = [r['model'] for r in subset if r['test_accuracy'] == best_score]
            qualifier = '并列最高' if len(winners) > 1 else '最高'
            lines.append(f'- {dataset}：本轮已测模型中，{"、".join(winners)} 的测试准确率{qualifier}，为 {best_score:.2%}。这描述本轮结果，不用于重新选模型或调整参数。')
        indexed = {r['model']: r for r in subset}
        if 'mlp' in indexed and 'lenet' in indexed:
            difference = (indexed['lenet']['test_accuracy'] - indexed['mlp']['test_accuracy']) * 100
            lines.append(f'  LeNet 相对 MLP 的测试准确率差为 {difference:+.2f} 个百分点。')
    lines.extend(['', '## 解释边界', '',
                  '这是统一初始配方的比较，不是每个网络经过充分调参后的最优性能。小数据集、大模型和相同学习率的组合可能欠拟合、过拟合或优化不稳定，需要结合训练与验证曲线解释。主对比采用 AMP 以适应笔记本的算力与显存，早期 FP32 运行保留作独立参考，不能与 AMP 行混成同一精度对比。', '',
                  '不能由不同架构的排序单独推出网络越深越好、参数越多越好等因果结论；这些问题需要同一模型家族中的控制变量实验。当前只有一个训练种子，尚不能说明小幅差异稳定显著。', '',
                  'AMP 主对比中的 Fashion-MNIST / GoogLeNet 曾在完成前中断，原始记录已保留，随后按同一配置从头完成。重跑与中断记录的早期数值存在差异，原因尚未定位，因此不宣称跨进程逐位复现。详情见 [运行记录](../../docs/运行中断与复现记录.md)；未依据测试成绩在两个运行间择优。', '',
                  'HWDB10 只有 597 张测试图片，且按书写人分组。分类准确率的差异应结合错误数、类别混淆、书写人和重复训练分析。不同数据集的任务难度不同，不能把其准确率直接当作模型优劣排名。', '',
                  '## 结果核验与原始记录', '',
                  '汇总程序从逐图预测重新计算准确率、Macro-F1 与混淆矩阵，核对最佳轮数及同一数据集的划分指纹。原始目录列在 baseline_summary.csv 的 run_dir 列，每次运行保留配置、源码副本、训练曲线、权重和错误样例。', '',
                  '## 后续实验', '',
                  '- 在同一 CNN 家族中改变深度和宽度，研究深度与参数量。',
                  '- 固定网络改变 batch size，同时记录更新次数和时间。',
                  '- 对关键结论进行多个随机种子复跑，报告均值与标准差。'])
    if missing:
        lines.extend(['', '尚未完成的基础组合：' + '、'.join(missing)])
    (out / 'baseline_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'baseline_error_analysis.md').write_text('\n'.join(error_sections) + '\n', encoding='utf-8')
    print(json.dumps({'verified_runs': len(records), 'missing_runs': missing}, indent=2))
    for r in records:
        print(f'{r["dataset"]:15s} {r["model"]:10s} accuracy={r["test_accuracy"]:.4%} F1={r["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
