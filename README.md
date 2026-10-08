# NILM 宿舍用电负载识别系统 — 阶段三/四代码管线

> 工程学导论期末项目 · 方案三（NILM 简版）
> 本仓库包含：**模拟数据生成器** + **阶段三（特征提取+分类）** + **阶段四（事件检测+差分匹配）**
> 硬件采集固件（ESP32 / I2S+DMA）在阶段一/二完成后接入，数据格式对齐即可零改动复用本管线。

## ⚡ 快速开始

```bash
pip install -r requirements.txt
python run_pipeline.py            # 完整跑：模拟数据 → 阶段三 → 阶段四
python run_pipeline.py --skip-sim # 已有数据，直接跑阶段三+四
```

产出：

```
figures/   stage3_confusion_matrix.png  混淆矩阵
           stage3_decision_tree.png     决策树结构图（报告用）
           stage3_feature_scatter.png   特征空间散点（THD vs PF）
           stage4_event_timeline.png    事件检测时间轴
results/   stage3_report.md             模型对照评估报告
           feature_table.csv            每段录制的 8 维特征
           stage4_events.csv            事件识别结果表
```

## 🗂 代码结构

```
src/config.py        全局参数（采样率/谐波次数/阈值…改参数只动这里）
src/simulator.py     模拟数据生成器：正弦+谐波+噪声+频率抖动+12bit量化
src/features.py      正弦相关法特征提取（替代 FFT，只用高数上册）
src/train_eval.py    阶段三：按录制段分组划分 + 决策树/kNN/SVM 对照
src/event_detect.py  阶段四：滑窗RMS事件检测 + 复系数差分匹配
run_pipeline.py      一键管线入口
```

## 📐 核心原理速览

**正弦相关法**（替代 FFT）：信号 x[n] 在频率 f 的能量 =
它与 cos/sin 掩波的内积：`A=(2/N)Σx·cos(2πft)`，`B=(2/N)Σx·sin(2πft)`，
幅值 `√(A²+B²)`。只用定积分思想 + 三角函数正交性，高数上册覆盖。

**8 维特征**：`[rms, h1, h3, h5, h7, thd, crest, pf]`
谐波全部取**相对基波的比例**——电网电压 ±10% 波动时比例不变（归一化抗扰）。

**防数据泄漏**：训练/测试按 `recording_id` **整段划分**（GroupShuffleSplit），
同一录制段的相邻样本不会一边一个导致准确率虚高。

**事件检测 + 差分匹配**（解决组合爆炸）：
总电流 = 各设备电流之和 → 突变事件 → 前后稳态段**复数谐波系数相减**
（相关运算是线性的，差分结果即新设备的指纹）→ 与单设备指纹库最近邻匹配。
只需单设备数据，21 种组合采集全省。

## ⚠️ 实现要点（踩过的坑，答辩可能问）

1. **供电方案必须是 12V 交流（AC-AC 变压器）**：SCT013 是交流互感器，
   直流供电下输出为零，50Hz 基波/谐波/PF 全部失效。原方案的 AC-DC
   适配器与整套特征体系原理冲突，已修正。
2. **差分必须在共享相位原点下做**：不同信号段的复系数若各自从 0 开始
   计时，相减不等于设备系数（基波对消不干净、PF 算错）。本实现对
   所有段使用绝对时间轴（`offset` 参数）。
3. **OFF 事件差分要反向**：`after − before = −设备电流`，复系数相位翻 π，
   PF 变负。代码按 RMS 变化方向自动对齐。
4. **事件检测用 RMS 相对变化率**（默认 8%），不用 0.1A 绝对阈值——
   开关电源负载本身波动大，绝对阈值会误报。

## 📊 模拟数据上的当前结果

| 指标 | 结果 | 目标 |
|---|---|---|
| 单设备识别（决策树，按段划分测试集） | 100% | ≥90% |
| 对照组 kNN / SVM | 100% / 100% | - |
| 双设备事件级识别 | 4/4 = 100% | ≥80% |

> 模拟数据仅验证管线正确性；最终成绩以 12V 实验台真实采集数据为准。
> 真实数据放入 `data/real/` 后运行
> `python run_pipeline.py --skip-sim --data-dir data/real` 即可。

## 🔌 接入真实数据的格式约定

CSV 列：`timestamp, voltage, current, label, recording_id`

- 每种设备一个文件，放 `data/real/`
- 每段录制一个独立 `recording_id`（防泄漏划分依赖它）
- 事件验证序列文件名以 `event_` 开头（会被阶段三跳过、阶段四使用）
