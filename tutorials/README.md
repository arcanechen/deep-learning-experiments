# CNN 与 RNN 训练教程配套练习

教程日期：2026 年 11 月 2 日。面向会 Python、刚开始使用 PyTorch 的学习者。

## 文档

- PDF：`../deliverables/CNN与RNN训练及独立实验手把手教程.pdf`
- 可编辑 Word：同名 `.docx`。
- 完整文字与源码：`CNN与RNN独立实验手把手教程_含源码.md`。

按教程第 1–2 节认识张量和梯度，第 3–7 节学习 CNN，第 8–11 节学习循环网络，第 12–17 节学习独立实验设计、排错与复现。

## 运行

本练习依赖当前项目已有数据、模型定义和 `ai-lab` 环境，结果仅用于学习，不是正式报告的新成绩。

```powershell
Set-Location 'D:\深度学习作业'
$py = 'D:\Anaconda\envs\ai-lab\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $py tutorials\train_practice.py --task cnn --device cpu --one-batch-steps 40
& $py tutorials\train_practice.py --task cnn --device cpu --epochs 2
& $py tutorials\train_practice.py --task rnn --device cuda --epochs 2
& $py tutorials\train_practice.py --task gru --device cuda --epochs 2
& $py tutorials\train_practice.py --task lstm --device cuda --epochs 2
& $py tutorials\check_practice.py
```

默认训练子集 512、验证子集 128、batch 32。调试时可以加 `--train-limit 64 --val-limit 32`。没有 CUDA 时改用 `--device cpu`。

每次在 `runs/任务_时间戳/` 创建新目录，保存 `config.json`、`history.json`、`best.pt`、`example_prediction.npz`。不覆盖正式实验，不计算测试集成绩。单批模式测量同一个训练 batch，不能理解成独立验证。练习脚本不支持精确断点恢复。

`check_practice.py` 复算最近一次 GRU 练习所保存的一个预测批次，打印摄氏度 MAE/RMSE 和逐预测距离 RMSE；一个批次通常不等于整个验证集。

## 已执行的验证

- CNN：CPU、64/32 样本、2 轮；另测单批模式 3 次更新。
- GRU：CUDA、64/32 窗口、2 轮。
- RNN 与 LSTM：CUDA、32/16 窗口、各 1 轮。
- 四种模型均成功保存、重新加载最佳权重并产生预测；这不等于完整断点续训验证。
- 误差复算脚本已运行。小规模验证证明流程可运行，不证明收敛或泛化性能。

正式实验及其冻结训练代码保持原有状态。后续新研究应另建实验目录并明确记录配置。
