# HWDB 官方数据来源核验

核验日期：2026-10-30。三个官方压缩包现已全部下载，通过全成员 ZIP CRC 校验，并记录本地 SHA-256。用户已决定使用官方数据自行选类，已生成十类子集，见 [十类子集说明](HWDB十类子集说明.md)。

## 一手来源

发布者：中国科学院自动化研究所模式识别国家重点实验室（CASIA/NLPR）。

- [数据库首页](https://nlpr.ia.ac.cn/databases/handwriting/home.html)
- [官方下载页](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)
- [离线数据库及 GNT 格式](https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html)
- [学术使用说明](https://nlpr.ia.ac.cn/databases/handwriting/Application_form.html)

官网说明：自 2020 年起，学术研究下载不再需要申请表；研究成果应适当引用原数据库论文。

## 应选择的文件

图像分类需要离线单字图像（GNT），不要误用页面顶部的 512 维手工特征数据，也不要误用在线笔画轨迹（POT）。

| 文件 | 直接下载链接 | 本机 HEAD 核验 |
|---|---|---|
| Gnt1.1TrainPart1.zip | https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip | HTTP 200，application/zip，919543940 bytes |
| Gnt1.1TrainPart2.zip | https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip | HTTP 200，application/zip，966309492 bytes |
| Gnt1.1Test.zip | https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1Test.zip | HTTP 200，application/zip，479659120 bytes |

总下载量约 2.37 GB（十进制）。上表为最初的 HEAD 结果；后续已完成全包下载、CRC 验证与本地 SHA-256 记录，详情见 `data/hwdb_official/sources.json`。解析扫描了共 1172907 条原始记录，选出 2995 个目标样本。

## 与课件的关系

官方 HWDB1.1 包含 3755 类汉字以及另外的字母、数字、符号，训练与测试采用不重叠的书写人。课件只说选自 HWDB1，使用 10 类、2382 张训练和 601 张测试图像，并未给出完整字符名单和样本清单。

针对 HWDB1、2382、601 等组合检索，未找到可核实为课程同一子集的公开数据包。不能仅凭样本数量接近就判定是同一份数据，也不能为了凑数量任意删样本后声称复现原划分。

本次按用户决定构建“CASIA-HWDB1.1 自选十类子集”。它是有权威来源的实验数据，不等同于课件中的原始子集；最终报告将明确标注这个差异。

## 后续准备原则

1. 记录原始下载地址、文件 SHA-256、数据版本。
2. 明确固定的十个汉字及其标签映射，不能依据测试表现挑选类别。
3. 保留官方训练/测试书写人边界，从训练书写人中划验证集。
4. GNT 解析后记录原文件名（书写人）、样本序号、原字符编码，输出图像与清单。
5. 将实际样本数、图像尺寸处理方法、类别分布写入报告。

推荐引用：C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang. CASIA online and offline Chinese handwriting databases. ICDAR, 2011, pp. 37–41.
