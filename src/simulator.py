"""
模拟数据生成器（仅用于管线调试，不作为最终证据）
================================================
在没有硬件的阶段，用数学模型生成"以假乱真"的电流/电压波形，
让特征提取、分类、事件检测整条管线先跑通。
真实数据采集后，只需替换 data/ 目录下的 CSV，其余代码零改动。

信号模型（对每种设备）：
    i(t) = Σ_k  I_k · sin(2π·k·f0·t + φ_k)  +  噪声
    v(t) = Σ_k  V_k · sin(2π·k·f0·t + ψ_k)  +  噪声

按开题报告的要求叠加四种"真实感"效应：
  1. 谐波      —— 每种设备有独特的谐波"指纹"（见 DEVICE_SIGNATURES）
  2. 噪声      —— 高斯测量噪声 + 轻微随机电磁干扰尖峰
  3. 频率抖动  —— 每段录制 f0 在 50±0.3Hz 随机，段内还有慢漂移（模拟电网波动）
  4. 量化      —— 模拟 12-bit ADC：信号经调理电路缩放后按 0~3.3V 量化

⚠️ AC-AC 修正方案：本模拟器生成的是 12V **交流**供电下的波形，
   有真实的 50Hz 基波、谐波和相位（功率因数可计算）。
   若使用直流适配器，互感器输出为零、谐波特征全部失效（见项目报告）。
"""

import os
import numpy as np
import pandas as pd

from . import config as C

# ----------------------------------------------------------------------
# 设备签名库：每种设备的"电流指纹"
#   harmonics: {谐波次数: 相对基波的幅值}
#   pf:        功率因数（基波电流滞后电压的相位 cosφ，感性负载 < 1）
#   rms:       典型工作电流有效值 (A)
#   noise:     测量噪声标准差相对满量程的比例
# 注：E5 用"调光灯"替代原方案 65W 笔记本电源——后者超过 5A 互感器
#     和 3A 适配器量程（方案自查发现的矛盾点）。
# ----------------------------------------------------------------------
DEVICE_SIGNATURES = {
    "E1_incandescent": {   # 白炽灯：纯电阻，几乎只有基波
        "harmonics": {1: 1.00, 3: 0.02, 5: 0.01},
        "pf": 0.99, "rms": 0.83, "noise": 0.004,
    },
    "E2_led_desk_lamp": {  # LED 台灯：开关电源驱动，奇次谐波丰富
        "harmonics": {1: 1.00, 3: 0.35, 5: 0.20, 7: 0.10, 9: 0.05},
        "pf": 0.70, "rms": 0.42, "noise": 0.006,
    },
    "E3_usb_charger": {    # USB 充电器：谐波更丰富 + 脉冲式电流（高峰均比）
        "harmonics": {1: 1.00, 3: 0.50, 5: 0.35, 7: 0.20, 9: 0.12, 11: 0.06},
        "pf": 0.55, "rms": 1.25, "noise": 0.008,
    },
    "E4_ac_fan": {         # 交流风扇：电机类感性负载，谐波少但相位滞后明显
        "harmonics": {1: 1.00, 3: 0.08, 5: 0.05, 7: 0.02},
        "pf": 0.60, "rms": 0.67, "noise": 0.005,
    },
    "E5_dimmer_lamp": {    # 调光灯（相控调压）：波形被"切掉一块"，THD 最高
        "harmonics": {1: 1.00, 3: 0.60, 5: 0.40, 7: 0.25, 9: 0.15},
        "pf": 0.50, "rms": 0.50, "noise": 0.006,
    },
}


def _quantize_adc(signal_physical, volt_per_unit, rng):
    """
    模拟"调理电路 + 12-bit ADC"的完整链路：
        物理量 -> 缩放成电压 -> 抬升 1.65V -> 量化到 0~3.3V -> 反变换回物理量
    这样 CSV 里存的仍是物理单位（V / A），但带有真实的量化台阶。
    """
    v_adc = signal_physical * volt_per_unit + C.ADC_OFFSET
    levels = 2 ** C.ADC_BITS
    v_quant = np.round(v_adc / C.ADC_VREF * (levels - 1)) / (levels - 1) * C.ADC_VREF
    v_quant = np.clip(v_quant, 0.0, C.ADC_VREF)
    return (v_quant - C.ADC_OFFSET) / volt_per_unit


def _jittered_phase(n_samples, f0, rng):
    """
    生成带频率抖动的相位序列 φ(t)。
    - 段间：每段录制 f0 在 50±0.3Hz 内随机（模拟不同时刻的电网频率）
    - 段内：瞬时频率做微小幅度的随机游走（±0.05Hz），模拟慢漂移
    返回累计相位（rad），直接供 sin() 使用。
    """
    drift = np.cumsum(rng.normal(0, 0.002, n_samples))      # 随机游走
    drift -= np.linspace(drift[0], drift[-1], n_samples)    # 去趋势，防止跑偏
    f_inst = f0 + np.clip(drift, -0.05, 0.05)
    phase = 2 * np.pi * np.cumsum(f_inst) / C.FS
    return phase


def generate_voltage(n_samples, f0, rng):
    """
    电压通道：12V 交流，含少量 3 次谐波（真实电网波形并非完美正弦，约 2%）。
    所有挂在线上的设备共享同一个电压相位基准——这是阶段四"差分匹配"的关键。
    """
    phase = _jittered_phase(n_samples, f0, rng)
    v_peak = C.MAINS_VOLTAGE_RMS * np.sqrt(2)
    v = v_peak * (np.sin(phase) + 0.02 * np.sin(3 * phase + rng.uniform(0, 2 * np.pi)))
    v += rng.normal(0, 0.002 * v_peak, n_samples)           # 测量噪声
    return _quantize_adc(v, C.PT_VOLT_PER_VOLT, rng), phase


def generate_current(device_key, n_samples, phase, rng):
    """
    按设备签名生成电流波形。
    参数 phase 来自 generate_voltage —— 同一电压基准下，各次谐波的相对相位
    由功率因数（基波）和谐波相位分布决定。
    """
    sig = DEVICE_SIGNATURES[device_key]
    phi_lag = np.arccos(sig["pf"])          # 基波电流滞后角
    # 由目标 RMS 反推基波峰值：I_rms² ≈ Σ (I_k_peak²/2)
    h = sig["harmonics"]
    i1_peak = sig["rms"] * np.sqrt(2) / np.sqrt(sum(a ** 2 for a in h.values()))

    i = np.zeros(n_samples)
    for k, rel in h.items():
        # 谐波相位：围绕 -k·φ_lag 加少量随机，模拟真实设备的个体差异
        ph = -k * phi_lag + rng.normal(0, 0.1)
        i += i1_peak * rel * np.sin(k * phase + ph)

    # 高斯测量噪声（相对互感器满量程 ±5A）
    i += rng.normal(0, sig["noise"] * 5.0, n_samples)
    # 偶发电磁干扰尖峰（约 0.1% 的采样点）
    spike_mask = rng.random(n_samples) < 0.001
    i[spike_mask] += rng.normal(0, 0.3, spike_mask.sum())

    return _quantize_adc(i, C.CT_VOLT_PER_AMP, rng)


def generate_dataset(out_dir=None, n_per_device=C.N_RECORDINGS_PER_DEVICE, seed=C.RANDOM_SEED):
    """
    生成单设备数据集：5 种设备 × 每种 n_per_device 段录制。
    输出：data/simulated/<label>.csv
    列：timestamp, voltage, current, label, recording_id
    """
    out_dir = out_dir or os.path.join(C.DATA_DIR, "simulated")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    summary = []
    for label in DEVICE_SIGNATURES:
        frames = []
        for r in range(n_per_device):
            f0 = 50.0 + rng.uniform(-0.3, 0.3)              # 段间频率抖动
            v, phase = generate_voltage(C.N_SAMPLES, f0, rng)
            i = generate_current(label, C.N_SAMPLES, phase, rng)
            t = np.arange(C.N_SAMPLES) / C.FS
            frames.append(pd.DataFrame({
                "timestamp": t, "voltage": v, "current": i,
                "label": label, "recording_id": f"{label}_r{r:02d}",
            }))
        df = pd.concat(frames, ignore_index=True)
        path = os.path.join(out_dir, f"{label}.csv")
        df.to_csv(path, index=False)
        summary.append((label, n_per_device, len(df)))
        print(f"  [sim] {label}: {n_per_device} 段 x {C.N_SAMPLES} 点 -> {path}")

    return summary


def generate_event_sequence(out_dir=None, seed=C.RANDOM_SEED + 1):
    """
    生成一段"多设备叠加"序列，用于阶段四事件检测验证。
    时间线（共 9 秒）：
        0~1s   空载基线
        1s     E2 LED 台灯打开
        3s     E4 风扇打开（两设备叠加）
        5s     E2 关闭（剩风扇）
        7s     E4 关闭（回基线）
    电流是各设备电流之和（基尔霍夫电流定律），共享同一电压相位。
    """
    out_dir = out_dir or os.path.join(C.DATA_DIR, "simulated")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_total = 9 * C.FS
    f0 = 50.0 + rng.uniform(-0.3, 0.3)
    v, phase = generate_voltage(n_total, f0, rng)

    i_base = rng.normal(0, 0.004 * 5.0, n_total)            # 基线只有噪声
    i_e2 = generate_current("E2_led_desk_lamp", n_total, phase, rng)
    i_e4 = generate_current("E4_ac_fan", n_total, phase, rng)

    i = i_base.copy()
    i[1 * C.FS:5 * C.FS] += i_e2[1 * C.FS:5 * C.FS]         # E2: 1s~5s
    i[3 * C.FS:7 * C.FS] += i_e4[3 * C.FS:7 * C.FS]         # E4: 3s~7s

    df = pd.DataFrame({
        "timestamp": np.arange(n_total) / C.FS,
        "voltage": v, "current": i,
        "label": "event_sequence", "recording_id": "event_seq_01",
    })
    path = os.path.join(out_dir, "event_sequence.csv")
    df.to_csv(path, index=False)

    # 事件真值表（评估用）
    truth = pd.DataFrame([
        {"time_s": 1.0, "type": "ON",  "device": "E2_led_desk_lamp"},
        {"time_s": 3.0, "type": "ON",  "device": "E4_ac_fan"},
        {"time_s": 5.0, "type": "OFF", "device": "E2_led_desk_lamp"},
        {"time_s": 7.0, "type": "OFF", "device": "E4_ac_fan"},
    ])
    truth_path = os.path.join(out_dir, "event_truth.csv")
    truth.to_csv(truth_path, index=False)
    print(f"  [sim] 事件序列 9s -> {path}（真值表 event_truth.csv）")
    return path, truth_path


if __name__ == "__main__":
    print("生成模拟数据 ...")
    generate_dataset()
    generate_event_sequence()
    print("完成。")
