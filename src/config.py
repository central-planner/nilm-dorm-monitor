"""
全局参数配置
============
所有模块共用的常量都集中在这里，改参数只改这一处。

设计依据（与开题报告 §5.3 一致）：
- 采样率 4 kHz：要分析到 50Hz 的 7 次谐波（350Hz），奈奎斯特要求 ≥700Hz，
  留 5 倍以上余量取 4 kHz。
- 每段 4096 点：4096 / 4000 ≈ 1 秒信号，频率分辨率 ≈ 1 Hz，
  足够区分 50.0Hz 和 50.5Hz 的电网频率抖动。
- 12-bit ADC：ESP32 内置，无需外置芯片。
"""

# ---------- 采样参数 ----------
FS = 4000               # 采样率 (Hz)
N_SAMPLES = 4096        # 每段录制采样点数（≈1秒）
F0_NOMINAL = 50.0       # 电网标称频率 (Hz)

# ---------- ADC / 调理电路参数（用于模拟量化效应） ----------
ADC_BITS = 12           # ESP32 ADC 位数
ADC_VREF = 3.3          # ADC 满量程电压 (V)
ADC_OFFSET = 1.65       # 调理电路抬升的直流中心 (V)

# SCT013-005：5A 交流 -> ±1V 输出，即 0.2 V/A
CT_VOLT_PER_AMP = 0.2
# ZMPT101B：经模块分压后约 0.05 V/V（12V交流峰值≈17V -> ±0.85V，在安全区内）
PT_VOLT_PER_VOLT = 0.05

# ---------- 供电参数（AC-AC 修正方案：12V 交流安全变压器） ----------
MAINS_VOLTAGE_RMS = 12.0    # 实验台供电电压有效值 (V)

# ---------- 数据生成 ----------
N_RECORDINGS_PER_DEVICE = 20    # 每种设备录制段数（>=20 才有统计意义）
RANDOM_SEED = 42

# ---------- 特征提取 ----------
HARMONICS = (1, 3, 5, 7)        # 特征向量中单独保留的谐波次数
THD_MAX_HARMONIC = 10           # THD 计算到的最高谐波次数
F0_SWEEP_RANGE = (48.0, 52.0)   # 正弦相关法扫频范围（找真实电网频率）
F0_SWEEP_STEP = 0.05

# ---------- 事件检测（阶段四） ----------
EVENT_WINDOW = 400          # 滑窗大小（采样点，= 0.1秒）
EVENT_HOP = 100             # 滑窗步长
EVENT_REL_THRESHOLD = 0.08  # RMS 相对变化阈值（占全程最大 RMS 的比例）
EVENT_MIN_GAP = 1000        # 两个事件之间的最小间隔（采样点），防止同一事件重复触发
STEADY_LEN = 2000           # 事件前后取多长的稳态段做差分（采样点，= 0.5秒）

# ---------- 路径 ----------
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
