# 实验一进度与续跑

更新时间：2026-10-31T12:52:35.885048+08:00

阶段：complete；训练完成 33/33，测试完成 33/33。

已完成上一阶段：五模型 × 三数据集 15 组 AMP 基础对比，见 reports/amp/baseline_summary.md。

本阶段固定 Fashion-MNIST、11 配置 × 3 种子、20 epochs，研究深度、宽度/参数量、batch size 及近似匹配参数量。

## 中断时的恢复命令（已完成时无需执行）

在 PowerShell 中执行（自动跳过完成项、恢复未完成项）：

```powershell
Set-Location 'D:\深度学习作业'
& 'D:\Anaconda\envs\ai-lab\python.exe' -u run_ablations.py
```

正在运行时不要重复启动；队列锁会阻止并行重复。查看 reports/ablations/queue.log 与本文件。

每轮原子保存 last.pt，包含模型、Adam、GradScaler、随机状态、DataLoader 状态、历史和最佳模型。中途退出最多需要重跑当前未保存轮次。

本文件中的轮次仅表示已持久化的完整轮次，不能单凭文件时间判断进程是否仍在运行。

Codex 账户额度可由聊天中的专用工具按需读取；本机训练脚本不查询账户。最近一次快照见 reports/ablations/usage_snapshot.json。训练每轮持续存盘，不依赖等待额度快用完才保存。

| 配置 | 种子 | 已保存轮次 | 训练完成 | 测试完成 |
|---|---:|---:|---|---|
| d1_w32_b128 | 42 | 20/20 | True | True |
| d1_w48_b128 | 42 | 20/20 | True | True |
| d2_w16_b128 | 42 | 20/20 | True | True |
| d2_w27_b128 | 42 | 20/20 | True | True |
| d2_w32_b32 | 42 | 20/20 | True | True |
| d2_w32_b64 | 42 | 20/20 | True | True |
| d2_w32_b128 | 42 | 20/20 | True | True |
| d2_w32_b256 | 42 | 20/20 | True | True |
| d2_w64_b128 | 42 | 20/20 | True | True |
| d3_w21_b128 | 42 | 20/20 | True | True |
| d3_w32_b128 | 42 | 20/20 | True | True |
| d1_w32_b128 | 43 | 20/20 | True | True |
| d1_w48_b128 | 43 | 20/20 | True | True |
| d2_w16_b128 | 43 | 20/20 | True | True |
| d2_w27_b128 | 43 | 20/20 | True | True |
| d2_w32_b32 | 43 | 20/20 | True | True |
| d2_w32_b64 | 43 | 20/20 | True | True |
| d2_w32_b128 | 43 | 20/20 | True | True |
| d2_w32_b256 | 43 | 20/20 | True | True |
| d2_w64_b128 | 43 | 20/20 | True | True |
| d3_w21_b128 | 43 | 20/20 | True | True |
| d3_w32_b128 | 43 | 20/20 | True | True |
| d1_w32_b128 | 44 | 20/20 | True | True |
| d1_w48_b128 | 44 | 20/20 | True | True |
| d2_w16_b128 | 44 | 20/20 | True | True |
| d2_w27_b128 | 44 | 20/20 | True | True |
| d2_w32_b32 | 44 | 20/20 | True | True |
| d2_w32_b64 | 44 | 20/20 | True | True |
| d2_w32_b128 | 44 | 20/20 | True | True |
| d2_w32_b256 | 44 | 20/20 | True | True |
| d2_w64_b128 | 44 | 20/20 | True | True |
| d3_w21_b128 | 44 | 20/20 | True | True |
| d3_w32_b128 | 44 | 20/20 | True | True |

## 文件入口

- 预先固定方案：reports/ablation_plan.json
- 实际模型：controlled_cnn.py；训练：train_ablation.py；队列：run_ablations.py
- 续训与梯度检查：tests/test_ablation_runtime.py
- 结果：reports/ablations/summary.md（汇总生成后）
- 入门学习：docs/第零课_从Python到PyTorch.md

会话交接与后续检查：docs/下一次继续任务说明.md。

后续：本阶段训练与测试已完成，无需重复启动队列。阅读 reports/实验一报告_可编辑稿.md 和 reports/ablations/结果解读.md，补充学生信息、个人学习总结及教师要求的提交格式。
