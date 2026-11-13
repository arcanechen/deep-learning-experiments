"""核对全部预测后，按预设配置计算跨种子均值和样本标准差。"""
import csv
import hashlib
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ablation_common import ROOT, OUT, PLAN_PATH, run_name, atomic_json, write_csv


def main():
    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    records, missing, splits = [], [], set()
    for c in plan['configurations']:
        for seed in plan['training_seeds']:
            folder = ROOT / 'runs' / run_name(c['id'], seed)
            if not (folder / 'test_metrics.json').exists():
                missing.append(folder.name)
                continue
            read = lambda f: json.loads((folder / f).read_text(encoding='utf-8'))
            config, summary, metrics, data = [read(f) for f in ('config.json', 'summary.json', 'test_metrics.json', 'data_info.json')]
            assert config['plan_sha256'] == hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
            assert all(config[k] == v for k, v in c.items()) and config['seed'] == seed
            assert config['epochs'] == plan['epochs'] and config['lr'] == plan['lr']
            assert config['precision'] == plan['precision'] and config['optimizer'] == plan['optimizer']
            assert config['dataset'] == plan['dataset'] and config['weight_decay'] == plan['weight_decay']
            with (folder / 'history.csv').open(encoding='utf-8-sig') as stream:
                history = list(csv.DictReader(stream))
            assert len(history) == plan['epochs']
            best = min(history, key=lambda r: float(r['val_loss']))
            assert int(best['epoch']) == summary['best_epoch'] == metrics['checkpoint_epoch']
            assert np.isclose(float(best['val_loss']), summary['best_val_loss'])
            assert summary['attempted_updates'] == c['planned_optimizer_steps']
            assert summary['actual_updates'] == sum(int(r['actual_updates']) for r in history)
            assert 0 <= summary['actual_updates'] <= summary['attempted_updates']
            assert metrics['checkpoint_sha256'] == hashlib.sha256((folder / 'best.pt').read_bytes()).hexdigest()
            with (folder / 'test_predictions.csv').open(encoding='utf-8-sig') as stream:
                predictions = list(csv.DictReader(stream))
            assert [int(r['test_index']) for r in predictions] == list(range(10000))
            truth = np.array([int(r['true_label']) for r in predictions])
            pred = np.array([int(r['predicted_label']) for r in predictions])
            cm = np.zeros((10, 10), dtype=int)
            np.add.at(cm, (truth, pred), 1)
            np.testing.assert_array_equal(cm, np.loadtxt(folder / 'confusion_matrix.csv', delimiter=',', dtype=int))
            accuracy = (truth == pred).mean()
            f1 = (2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))).mean()
            assert np.isclose(accuracy, metrics['accuracy']) and np.isclose(f1, metrics['macro_f1'])
            assert metrics['test_size'] == len(truth)
            splits.add(data['split_sha256'])
            records.append({'config': c['id'], 'seed': seed, 'conv_layers': c['conv_layers'],
                            'width': c['width'], 'batch_size': c['batch_size'], 'parameters': c['parameters'],
                            'best_epoch': summary['best_epoch'], 'accuracy': accuracy, 'macro_f1': f1,
                            'compute_seconds': summary['compute_seconds'], 'peak_allocated_mb': summary['peak_allocated_mb'],
                            'actual_updates': summary['actual_updates'], 'attempted_updates': summary['attempted_updates']})
    assert len(splits) <= 1
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'verification.json', {'verified_runs': len(records), 'missing': missing,
                'checks': ['frozen plan', 'common split', 'validation selection', 'epoch budget',
                           'actual updates', 'checkpoint hash', 'predictions -> accuracy/F1/confusion matrix']})
    if not records:
        print('No completed test results yet; progress is saved separately.')
        return
    write_csv(OUT / 'individual_runs.csv', records)
    aggregate = []
    for c in plan['configurations']:
        rows = [r for r in records if r['config'] == c['id']]
        if len(rows) != len(plan['training_seeds']):
            continue
        row = {**c, 'n_seeds': len(rows)}
        for metric in ('accuracy', 'macro_f1', 'compute_seconds', 'peak_allocated_mb', 'actual_updates'):
            values = [r[metric] for r in rows]
            row[metric + '_mean'] = float(np.mean(values))
            row[metric + '_std'] = float(np.std(values, ddof=1))
        aggregate.append(row)
    atomic_json(OUT / 'aggregate.json', aggregate)
    lines = ['# 实验一：控制变量与重复训练', '',
             f'已核验 {len(records)}/33 次独立运行；{len(aggregate)}/11 种配置已具备全部三个种子。', '',
             'Fashion-MNIST；固定划分 54000/6000/10000；Adam，lr=0.001，无权重衰减，无数据增强；20 epochs；AMP；seed=42/43/44。模型按最低验证损失选取。全部配置预先固定，先完成全部训练，再统一测试。', '',
             '以下“±”为三个训练种子的样本标准差，不是置信区间；只反映训练随机性，不包含重新抽取训练/测试数据的变化。', '']
    groups = [('depth', '深度：固定 W=32、batch=128'), ('width', '宽度/参数量：固定 6 个卷积层、batch=128'),
              ('batch', '批量大小：固定 6 个卷积层、W=32'), ('matched_parameters', '近似匹配参数量：同时改变深度与宽度')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, groups):
        rows = [r for r in aggregate if key in r['groups']]
        lines += [f'## {title}', '', '| 配置 | 参数量 | 测试准确率（均值±SD） | Macro-F1（均值±SD） | 计算秒数（均值） | 实际更新次数（均值） |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f'| {r["id"]} | {r["parameters"]:,} | {100*r["accuracy_mean"]:.2f} ± {100*r["accuracy_std"]:.2f}% | '
                         f'{r["macro_f1_mean"]:.4f} ± {r["macro_f1_std"]:.4f} | {r["compute_seconds_mean"]:.1f} | {r["actual_updates_mean"]:.1f} |')
        lines.append('')
        ax.errorbar(range(len(rows)), [100*r['accuracy_mean'] for r in rows],
                    yerr=[100*r['accuracy_std'] for r in rows], fmt='o', capsize=5)
        lower = min([100*(r['accuracy_mean']-r['accuracy_std']) for r in rows], default=80)
        upper = max([100*(r['accuracy_mean']+r['accuracy_std']) for r in rows], default=100)
        ax.set(xticks=range(len(rows)), xticklabels=[r['id'] for r in rows],
               title=key, ylabel='Test accuracy (%)', ylim=(max(0, min(80, lower-2)), max(100, upper+1)))
        ax.tick_params(axis='x', rotation=25)
        ax.grid(alpha=.2)
    fig.suptitle('Three seeds: mean and sample standard deviation; fixed training recipe')
    fig.savefig(OUT / 'comparison.png', dpi=150)
    plt.close(fig)
    lines += ['', '![控制变量对比](comparison.png)', '', '## 解释边界', '',
              '- 固定宽度增加深度时，参数量也会增加；近似匹配参数组可作补充，但其中宽度又同时变化，不能声称完全隔离了所有因素。',
              '- 固定 epoch 改变 batch 时，看到的样本数相同，更新次数不同。BN 的批统计也受 batch 影响。',
              '- 实际更新次数由优化器步骤钩子统计；若 AMP 检测到非有限梯度，会跳过相应更新，不能只拿 batch 数代替。',
              '- 耗时只含成功完成轮次的训练与验证，不含初始化、检查点写盘和被中断的部分轮次。峰值显存为 PyTorch 已分配内存。与上一阶段含存盘的计时口径不同，不直接横比。',
              '- 同一个配置在多个研究组出现时复用同三次运行，不能当成新的独立重复。',
              '- 三个种子不足以证明很小的差异具有统计显著性；这些结论只适用于当前数据划分、模型家族与训练配方。', '',
              '逐次指标见 individual_runs.csv；核验记录见 verification.json；原始权重、日志与预测在 runs/fashion_ablation_*。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'VERIFIED {len(records)}/33; complete configurations {len(aggregate)}/11')
    if len(records) == 33 and len(aggregate) == 11 and not missing:
        subprocess.run([sys.executable, str(ROOT / 'analyze_ablation_results.py')], cwd=ROOT, check=True)
        # 可编辑报告可能包含学生后续补充，自动队列不覆盖已存在的报告。
        if not (ROOT / 'reports' / '实验一报告_可编辑稿.md').exists():
            subprocess.run([sys.executable, str(ROOT / 'write_experiment_report.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
