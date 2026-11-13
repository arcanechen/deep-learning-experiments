# CASIA-HWDB1.1 自选十类子集

本子集使用中科院自动化所官方离线单字图像构建。用户已选择采用官方数据自行选类；它不等同于课件的原十类子集，实际样本数按解析结果记录，不人为凑成 2382/601。

## 固定类别与实际数量

| 标签 | 汉字 | 训练 | 验证 | 测试 |
|---|---|---:|---:|---:|
| 0 | 日 | 216 | 24 | 60 |
| 1 | 月 | 216 | 23 | 61 |
| 2 | 山 | 216 | 24 | 60 |
| 3 | 水 | 216 | 24 | 59 |
| 4 | 火 | 216 | 24 | 60 |
| 5 | 木 | 216 | 24 | 60 |
| 6 | 土 | 216 | 24 | 59 |
| 7 | 田 | 216 | 24 | 58 |
| 8 | 人 | 216 | 24 | 60 |
| 9 | 中 | 215 | 24 | 60 |
| 合计 | | 2159 | 239 | 597 |

## 划分与来源

- 类别选择在查看模型效果前固定，配置保存在 `hwdb_selection.json`。
- 官方 240 位训练书写人按固定种子 2026 划为 216 位训练、24 位验证；官方 60 位测试书写人保持独立。
- 不按单张图片随机划分汉字数据，避免相同书写人的风格同时出现在训练、验证、测试中。
- 所有三个压缩包均完成全成员 CRC 检查与本地 SHA-256 记录。SHA-256 是本地完整性指纹，不是发布方提供的签名。
- `data/hwdb10/samples.jsonl` 为每张图片记录标签、书写人、归档文件、GNT 记录序号/偏移、像素与 PNG 哈希。
- 原始灰度 PNG 保存在 `data/hwdb10/images`；预处理缓存为三个独立 NPZ 文件。
- `data/hwdb10/metadata.json` 保存实际类别数、样本数、书写人清单和文件指纹。

## 图像处理

保留官方灰度与白底黑字。原图保持长宽比，放在白色正方形画布中央，四周增加约 10% 边距后缩放为 64×64。原始尺寸图片单独保留。现有五模型的共同训练入口会再缩放至 32×32，并按固定公式映射到 [-1,1]。

这与课件示例直接缩放到 64×64 有差异，应在最终报告中注明。将来如果研究 32/64 分辨率，应作为单独变量，不能混用后直接比较。

## 复现与使用

```powershell
& 'D:\Anaconda\envs\ai-lab\python.exe' download_hwdb.py
& 'D:\Anaconda\envs\ai-lab\python.exe' prepare_hwdb.py
& 'D:\Anaconda\envs\ai-lab\python.exe' verify_hwdb.py
& 'D:\Anaconda\envs\ai-lab\python.exe' train.py --dataset hwdb10 --model lenet --epochs 10 --seed 42
```

已有子集不会被准备脚本覆盖，以保持实验划分固定。下载脚本可续传和复核已有原始包。

独立核验结果见 `data/hwdb10/verification.json`（运行 verify_hwdb.py 后生成）。数据准备完成不代表已完成五模型正式训练。

## 学习重点

1. 标签 0～9 只是模型使用的索引，必须保留与汉字的映射。
2. 对书写人泛化的评估需要按人划分；单纯随机打乱图片可能低估遇到新笔迹时的误差。
3. 测试集只用来评价已确定的实验设置，不能根据测试集表现换类、换划分或调参。
4. 原始图片、模型输入缓存和样本清单分别承担可追溯、效率和复现的作用。

[官方下载页](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)；[GNT 格式说明](https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html)。

引用：C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang. CASIA online and offline Chinese handwriting databases. ICDAR, 2011, pp. 37–41.
