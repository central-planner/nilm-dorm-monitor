"""
特征提取：正弦相关法（替代 FFT）
================================
为什么不用 FFT？——傅里叶级数在高数下册，本项目只用高数上册的知识。

数学原理（一次谐波幅值的计算）
------------------------------
想知道信号 x[n] 在频率 f 上有多少能量，只需算两个**内积（点积）**：

    A = (2/N) · Σ x[n] · cos(2π·f·n·Ts)
    B = (2/N) · Σ x[n] · sin(2π·f·n·Ts)
    幅值 amp = √(A² + B²)     相位 φ = atan2(B, A)

这正是定积分 ∫x(t)·cos(2πft)dt 的离散化——用的是
"定积分的几何意义 + 三角函数正交性"，高数上册完全覆盖。

与 FFT 相比的两个实际好处：
  1. 直接指定频率，不存在频谱泄漏，无需窗函数校正；
  2. 可以在 48~52Hz 扫频找幅值最大点 → 顺便测出电网真实频率 f0。

特征向量（8 维，全部对电压波动鲁棒）
--------------------------------------
    [rms,           电流有效值（绝对量，区分功率档位）
     h1,            基波幅值（归一化后恒为 1，占位保持向量结构清晰）
     h3, h5, h7,    3/5/7 次谐波相对基波的比例（电压波动时比例不变）
     thd,           总谐波畸变率 = √(Σ_{k=2..10} Hk²) / H1
     crest,         峰均比 = 峰值 / 有效值（脉冲型负载显著偏大）
     pf]            功率因数 = cos(电压基波相位 - 电流基波相位)
"""

import numpy as np

from . import config as C


def correlate(signal, freq, fs=C.FS, offset=0):
    """
    正弦相关法的核心：计算信号在单个频率上的幅值与相位。
    返回 (幅值, 相位, 复系数 A+1j*B)。
    复系数形式在阶段四"差分"中直接可减（相关运算是线性的）。

    参数 offset：该段信号在整段录音中的起始采样点索引。
    掩波用绝对时间轴 (n+offset)/fs 生成——只有所有段共享同一个
    相位原点，不同段的复系数才能直接相减（阶段四差分匹配的前提）。
    单段特征提取时 offset=0 即可（相位差在同一原点下依然自洽）。
    """
    n = offset + np.arange(len(signal))
    ang = 2 * np.pi * freq * n / fs
    a = (2.0 / len(signal)) * np.sum(signal * np.cos(ang))
    b = (2.0 / len(signal)) * np.sum(signal * np.sin(ang))
    return np.hypot(a, b), np.arctan2(b, a), complex(a, b)


def estimate_f0(signal, fs=C.FS,
                f_range=C.F0_SWEEP_RANGE, step=C.F0_SWEEP_STEP):
    """
    扫频估计真实电网频率：在 48~52Hz 内找基波幅值最大的频率点。
    通常对**电压通道**做（更干净），得到的 f0 供电流通道使用。
    """
    freqs = np.arange(f_range[0], f_range[1] + 1e-9, step)
    amps = np.array([correlate(signal, f, fs)[0] for f in freqs])
    return float(freqs[np.argmax(amps)])


def harmonic_coeffs(signal, f0, max_k=C.THD_MAX_HARMONIC, fs=C.FS, offset=0):
    """
    一次性算出 1~max_k 次谐波的复系数（A+1jB）。
    复系数是可线性叠加/相减的——阶段四差分匹配直接复用本函数。
    offset 含义见 correlate()：跨段比较/相减时必须传绝对起始位置。
    返回 dict: {k: complex}
    """
    return {k: correlate(signal, k * f0, fs, offset)[2] for k in range(1, max_k + 1)}


def extract_features(current, voltage, fs=C.FS):
    """
    对一段稳态录制（≈1秒）提取 8 维特征向量。
    返回 (feature_dict, f0)。
    """
    # 1) 从电压通道估计真实电网频率
    f0 = estimate_f0(voltage, fs)

    # 2) 电压基波相位（功率因数的相位基准）
    _, phi_v1, _ = correlate(voltage, f0, fs)

    # 3) 电流各次谐波
    coeffs = harmonic_coeffs(current, f0, fs=fs)
    amps = {k: abs(coeffs[k]) for k in coeffs}
    h1 = amps[1] + 1e-12                       # 防除零
    phi_i1 = np.angle(coeffs[1])

    # 4) 组装特征
    rms = float(np.sqrt(np.mean(current ** 2)))
    feats = {
        "rms": rms,
        "h1": 1.0,                                                # 归一化基准
        "h3": amps.get(3, 0.0) / h1,
        "h5": amps.get(5, 0.0) / h1,
        "h7": amps.get(7, 0.0) / h1,
        "thd": float(np.sqrt(sum(a ** 2 for k, a in amps.items() if k >= 2)) / h1),
        "crest": float(np.max(np.abs(current)) / (rms + 1e-12)),
        "pf": float(np.cos(phi_v1 - phi_i1)),
    }
    return feats, f0


FEATURE_ORDER = ["rms", "h1", "h3", "h5", "h7", "thd", "crest", "pf"]


def features_to_vector(feats):
    """特征 dict -> 固定顺序的 numpy 向量（喂给 sklearn）。"""
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=float)
