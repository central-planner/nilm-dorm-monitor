"""
阶段四：事件检测 + 差分匹配
===========================
解决"组合爆炸"：7 类设备两两组合 21 种，不可能逐个采集。
思路（与开题报告 §7 一致）：

    总电流 i(t) = i_基线(t) + i_新设备(t)        （基尔霍夫电流定律）
    => 检测到突变事件 -> 取事件前后稳态段 -> 差分 Δi(t)
    -> Δi 的特征就是"新打开/关闭的那台设备"的指纹
    -> 只需单设备数据库，无需采集任何组合数据

两个关键实现细节：
  1. 事件检测用**滑窗 RMS 的相对变化率**，而不是报告初稿的 0.1A 绝对阈值——
     开关电源类负载本身波动大，绝对阈值会误报。
  2. 差分在**复数谐波系数**上做，而不是在时域波形上直接相减——
     相关运算是线性的：coeff(i_after - i_before) = coeff(i_after) - coeff(i_before)。
     这样天然对齐同一电压相位基准，避免两段波形相位错位导致差分失真。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .features import estimate_f0, harmonic_coeffs, correlate, FEATURE_ORDER


# ----------------------------------------------------------------------
# 1. 事件检测：滑窗 RMS + 相对阈值
# ----------------------------------------------------------------------
def sliding_rms(x, window=C.EVENT_WINDOW, hop=C.EVENT_HOP):
    """滑窗 RMS，返回 (rms 数组, 对应的中心时刻采样点索引)。"""
    idx = np.arange(0, len(x) - window + 1, hop)
    rms = np.array([np.sqrt(np.mean(x[i:i + window] ** 2)) for i in idx])
    return rms, idx + window // 2


def detect_events(current, fs=C.FS,
                  rel_threshold=C.EVENT_REL_THRESHOLD, min_gap=C.EVENT_MIN_GAP):
    """
    在滑窗 RMS 序列上找突变点。
    阈值 = rel_threshold × 全程最大 RMS（相对量，对负载大小自适应）。
    min_gap 内的重复触发被合并为一个事件。
    返回 [(采样点索引, Δrms), ...]，Δrms>0 表示有设备打开，<0 表示关闭。
    """
    rms, centers = sliding_rms(current)
    delta = np.diff(rms)
    threshold = rel_threshold * rms.max()

    raw = np.where(np.abs(delta) > threshold)[0]
    events, last = [], -min_gap
    for j in raw:
        t = centers[j + 1]
        if t - last >= min_gap:                     # 合并同一事件的连续触发
            events.append((int(t), float(delta[j])))
            last = t
        elif abs(delta[j]) > abs(events[-1][1]):    # 保留变化最大的一帧
            events[-1] = (int(t), float(delta[j]))
    return events


# ----------------------------------------------------------------------
# 2. 差分匹配：事件前后稳态段 -> 复系数差分 -> 与指纹库最近邻
# ----------------------------------------------------------------------
def _segment_signature(current, f0, offset=0):
    """
    一段稳态信号的"复数指纹"：1~10 次谐波复系数 + RMS。
    复系数线性可减，是差分匹配的原子操作。
    offset 是该段在整段录音中的绝对起始采样点——保证所有段的
    复系数共享同一相位原点，跨段相减才有意义。
    """
    coeffs = harmonic_coeffs(current, f0, offset=offset)
    rms = float(np.sqrt(np.mean(current ** 2)))
    return {"coeffs": coeffs, "rms": rms}


def build_device_library(feature_table):
    """
    从阶段三的特征表建"单设备指纹库"：每类设备取特征均值 + 典型 RMS。
    匹配时用 [rms, h3, h5, h7, thd, crest, pf]（h1 恒为 1，无区分度）。
    """
    use = ["rms", "h3", "h5", "h7", "thd", "crest", "pf"]
    lib = {}
    for label, grp in feature_table.groupby("label"):
        lib[label] = grp[use].mean().to_numpy()
    return lib, use


def delta_features(seg_before, seg_after, f0, phi_v1, fs=C.FS):
    """
    由事件前后两段稳态信号计算"变化的那台设备"的特征向量。
    三个关键点：
      1. 谐波复系数相减（线性性）：coeff(i_a - i_b) = coeff(i_a) - coeff(i_b)
      2. **方向对齐**：OFF 事件（i_a - i_b = -设备电流）要把差分反过来，
         否则所有复系数相位翻转 π，功率因数变成负值，匹配必错
      3. **相位基准**：PF 必须以电压基波相位为参照（cos(φ_v1 - φ_i1)），
         不能直接对差分系数取 angle——相关运算的相位原点不是电压相位
    """
    c_b, c_a = seg_before["coeffs"], seg_after["coeffs"]
    d_rms = seg_after["rms"] - seg_before["rms"]
    on = d_rms >= 0
    d_coeffs = {k: (c_a[k] - c_b[k]) if on else (c_b[k] - c_a[k]) for k in c_b}
    d_amps = {k: abs(v) for k, v in d_coeffs.items()}
    h1 = d_amps[1] + 1e-12

    # 差分波形的时域重建仅用于 crest 估计（用谐波合成，避免相位错位问题）
    # 注意合成公式必须与 correlate 的定义一致：x ≈ A·cos(ωt) + B·sin(ωt)
    n = np.arange(int(fs))                        # 重建 1 秒
    wave = np.zeros(len(n))
    for k, c in d_coeffs.items():
        ang = 2 * np.pi * k * f0 * n / fs
        wave += c.real * np.cos(ang) + c.imag * np.sin(ang)
    crest = float(np.max(np.abs(wave)) / (abs(d_rms) + 1e-12))

    return {
        "rms": abs(d_rms),
        "h3": d_amps.get(3, 0.0) / h1,
        "h5": d_amps.get(5, 0.0) / h1,
        "h7": d_amps.get(7, 0.0) / h1,
        "thd": float(np.sqrt(sum(a ** 2 for k, a in d_amps.items() if k >= 2)) / h1),
        "crest": crest,
        "pf": float(np.cos(phi_v1 - np.angle(d_coeffs[1]))),   # 以电压相位为基准
        "sign": 1.0 if on else -1.0,
    }


def match_device(d_feats, library, lib_keys):
    """
    最近邻匹配：对量纲差异大的特征做标准化后算欧氏距离。
    返回 (设备名, 置信度)。置信度 = 1 - d1/(d1+d2)（最近与次近邻的相对差距）。
    """
    names = list(library.keys())
    M = np.array([library[n] for n in names])
    v = np.array([d_feats[k] for k in lib_keys])
    scale = M.std(axis=0) + 1e-9
    d = np.linalg.norm((M - v) / scale, axis=1)
    order = np.argsort(d)
    conf = 1.0 - d[order[0]] / (d[order[0]] + d[order[1]] + 1e-12)
    return names[order[0]], float(conf)


# ----------------------------------------------------------------------
# 3. 阶段四主流程
# ----------------------------------------------------------------------
def run_stage4(feature_table, seq_path=None, truth_path=None, verbose=True):
    seq_path = seq_path or os.path.join(C.DATA_DIR, "simulated", "event_sequence.csv")
    truth_path = truth_path or os.path.join(C.DATA_DIR, "simulated", "event_truth.csv")

    df = pd.read_csv(seq_path)
    current = df["current"].to_numpy()
    voltage = df["voltage"].to_numpy()
    f0 = estimate_f0(voltage)
    if verbose:
        print(f"  [evt] 估计电网频率 f0 = {f0:.2f} Hz")

    events = detect_events(current)
    library, lib_keys = build_device_library(feature_table)

    # 对每个事件：取前后稳态段 -> 差分 -> 匹配
    rows = []
    for t, d_rms in events:
        b0, b1 = max(0, t - C.STEADY_LEN - C.EVENT_WINDOW), max(C.EVENT_WINDOW, t - C.EVENT_WINDOW)
        a0, a1 = min(len(current) - C.STEADY_LEN, t + C.EVENT_WINDOW), min(len(current), t + C.EVENT_WINDOW + C.STEADY_LEN)
        seg_b = _segment_signature(current[b0:b1], f0, offset=b0)
        seg_a = _segment_signature(current[a0:a1], f0, offset=a0)
        # 电压基波相位基准（与电流段共享绝对时间原点，取事件后稳态段即可）
        _, phi_v1, _ = correlate(voltage[a0:a1], f0, offset=a0)
        d_feats = delta_features(seg_b, seg_a, f0, phi_v1)
        device, conf = match_device(d_feats, library, lib_keys)
        rows.append({
            "time_s": round(t / C.FS, 3),
            "type": "ON" if d_feats["sign"] > 0 else "OFF",
            "device": device, "confidence": round(conf, 3),
            "delta_rms_A": round(d_feats["rms"], 3),
        })
        if verbose:
            print(f"  [evt] t={t / C.FS:5.2f}s  {rows[-1]['type']:3s}  "
                  f"-> {device:20s} (置信度 {conf:.2f}, ΔRMS {d_feats['rms']:.2f}A)")

    pred = pd.DataFrame(rows)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    pred.to_csv(os.path.join(C.RESULTS_DIR, "stage4_events.csv"), index=False)

    # ---- 与真值表对比，计算事件级准确率 ----
    acc = None
    if os.path.exists(truth_path):
        truth = pd.read_csv(truth_path)
        hits = 0
        for _, tr in truth.iterrows():
            cand = pred[np.abs(pred["time_s"] - tr["time_s"]) < 0.5]
            if len(cand) and cand.iloc[0]["device"] == tr["device"] \
                    and cand.iloc[0]["type"] == tr["type"]:
                hits += 1
        acc = hits / len(truth)
        if verbose:
            print(f"  [evt] 事件级识别准确率: {hits}/{len(truth)} = {acc:.1%}")

    # ---- 事件时间轴图 ----
    rms, centers = sliding_rms(current)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    t_all = np.arange(len(current)) / C.FS
    axes[0].plot(t_all, current, lw=0.4, color="steelblue")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Event detection & delta matching on composite sequence")
    axes[1].plot(centers / C.FS, rms, lw=1.2, color="darkorange")
    axes[1].set_ylabel("Sliding RMS (A)")
    axes[1].set_xlabel("Time (s)")
    for _, r in pred.iterrows():
        for ax in axes:
            ax.axvline(r["time_s"], color="red", ls="--", lw=0.8, alpha=0.6)
        axes[1].annotate(f"{r['type']}\n{r['device'].split('_')[0]}\n{r['confidence']:.2f}",
                         (r["time_s"], rms.max() * 0.9), ha="center", fontsize=8,
                         color="darkred")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "stage4_event_timeline.png"), dpi=150)
    plt.close(fig)

    return pred, acc
