"""从持久化的轮次状态刷新进度；无需加载权重。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from ablation_common import PLAN_PATH, OUT, ROOT, atomic_json, run_name


def refresh(phase=None):
    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    rows = []
    for seed in plan['training_seeds']:
        for c in plan['configurations']:
            folder = ROOT / 'runs' / run_name(c['id'], seed)
            progress = folder / 'progress.json'
            row = json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else {'epoch': 0}
            if (folder / 'summary.json').exists():
                complete = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
                row.update({'epoch': plan['epochs'], 'best_epoch': complete['best_epoch'],
                            'best_val_loss': complete['best_val_loss']})
            rows.append({'config': c['id'], 'seed': seed, **row,
                         'trained': (folder / 'summary.json').exists(),
                         'tested': (folder / 'test_metrics.json').exists()})
    previous = OUT / 'progress.json'
    old_phase = json.loads(previous.read_text(encoding='utf-8')).get('phase', 'prepared') if previous.exists() else 'prepared'
    state = {'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
             'phase': phase or old_phase, 'planned_runs': len(rows),
             'trained_runs': sum(r['trained'] for r in rows),
             'tested_runs': sum(r['tested'] for r in rows), 'runs': rows}
    atomic_json(previous, state)
    lines = ['# 实验一进度与续跑', '', f'更新时间：{state["updated_at"]}', '',
             f'阶段：{state["phase"]}；训练完成 {state["trained_runs"]}/{len(rows)}，测试完成 {state["tested_runs"]}/{len(rows)}。', '',
             '已完成上一阶段：五模型 × 三数据集 15 组 AMP 基础对比，见 reports/amp/baseline_summary.md。', '',
             '本阶段固定 Fashion-MNIST、11 配置 × 3 种子、20 epochs，研究深度、宽度/参数量、batch size 及近似匹配参数量。', '',
             '## 中断时的恢复命令（已完成时无需执行）', '', '在 PowerShell 中执行（自动跳过完成项、恢复未完成项）：', '',
             '```powershell', "Set-Location 'D:\\深度学习作业'",
             "& 'D:\\Anaconda\\envs\\ai-lab\\python.exe' -u run_ablations.py", '```', '',
             '正在运行时不要重复启动；队列锁会阻止并行重复。查看 reports/ablations/queue.log 与本文件。', '',
             '每轮原子保存 last.pt，包含模型、Adam、GradScaler、随机状态、DataLoader 状态、历史和最佳模型。中途退出最多需要重跑当前未保存轮次。', '',
             '本文件中的轮次仅表示已持久化的完整轮次，不能单凭文件时间判断进程是否仍在运行。', '',
             'Codex 账户额度可由聊天中的专用工具按需读取；本机训练脚本不查询账户。最近一次快照见 reports/ablations/usage_snapshot.json。训练每轮持续存盘，不依赖等待额度快用完才保存。', '',
             '| 配置 | 种子 | 已保存轮次 | 训练完成 | 测试完成 |', '|---|---:|---:|---|---|']
    lines += [f'| {r["config"]} | {r["seed"]} | {r["epoch"]}/20 | {r["trained"]} | {r["tested"]} |' for r in rows]
    lines += ['', '## 文件入口', '', '- 预先固定方案：reports/ablation_plan.json',
              '- 实际模型：controlled_cnn.py；训练：train_ablation.py；队列：run_ablations.py',
              '- 续训与梯度检查：tests/test_ablation_runtime.py',
              '- 结果：reports/ablations/summary.md（汇总生成后）',
              '- 入门学习：docs/第零课_从Python到PyTorch.md', '',
              '会话交接与后续检查：docs/下一次继续任务说明.md。', '',
              ('后续：本阶段训练与测试已完成，无需重复启动队列。阅读 reports/实验一报告_可编辑稿.md 和 reports/ablations/结果解读.md，补充学生信息、个人学习总结及教师要求的提交格式。'
               if state['phase'] == 'complete' else
               '后续：完成全部配置训练，再统一独立测试、计算三个种子的均值与样本标准差；不得根据测试结果更改本轮配置。')]
    text = '\n'.join(lines) + '\n'
    for path in (ROOT / '进度与续跑.md', OUT / 'progress.md'):
        temp = path.with_suffix('.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    return state


if __name__ == '__main__':
    print(json.dumps(refresh(), ensure_ascii=False, indent=2))
