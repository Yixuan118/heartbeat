import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import convolve
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import warnings

# --- 物理和模拟参数 ---
fs = 100          # 原始采样率 (Hz)
duration = 10     # 信号持续时间 (秒)
n_points_signal = int(duration * fs) # 信号点数

# -- 模拟床垫/介质的物理属性 (主路径) --
distance = 0.15       # 源到接收器的有效传播距离 (米)
velocity_ref = 30     # 参考波速 (m/s)
# 使用较低的 Q 值以增强主路径的衰减和弥散
q_factor = 5          # 品质因子 Q (无量纲) - 控制主路径衰减和弥散 (v5.1: lowered from 10 to 5)
f_ref = fs / 4        # 参考频率 (Hz)

# -- 几何扩散模型 (主路径) --
geometric_spreading_factor = 1.0 / (distance + 1e-6)

# -- 混响参数 (指数衰减噪声模型) --
# 混响相对于主路径峰值到达时间的额外延迟因子
reverb_start_delay_factor = 1.1 # e.g., 1.1 means reverb starts 10% after expected direct arrival
reverb_decay_tau = 0.25       # 指数衰减时间常数 tau (秒) - 控制混响尾巴长度 (v5.1: increased from hypothetical e.g. 0.1)
# 混响噪声幅度 (相对于主路径峰值的比例) - 控制混响强度
reverb_noise_rel_amplitude = 0.6 # (v5.1: adjusted for potentially stronger effect)

# -- 冲激响应计算参数 --
ir_duration_s = 1.0 # 总冲激响应的持续时间 (秒)
n_points_ir = int(ir_duration_s * fs)
ir_time_axis = np.arange(n_points_ir) / fs

# -- 噪声参数 --
noise_level = 0.04     # 噪声标准差，相对于最终信号最大绝对值

# --- 数据加载与路径设置 ---
data_path = './data/BCG.npy'
output_dir = './data'
output_filename = 'BCG_after_bed.npy'
output_filepath = os.path.join(output_dir, output_filename)

if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        exit()

# --- 加载数据 ---
try:
    real_bcg_data = np.load(data_path)
    print(f"Shape of loaded real_bcg_data: {real_bcg_data.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 数据准备 ---
# --- 数据准备（仅使用前 6000 个样本）---
num_samples_to_process = min(6000, real_bcg_data.shape[0])  # 避免超出实际数据长度
bcg_signals = real_bcg_data[:num_samples_to_process]
print(f"Using first {num_samples_to_process} samples for simulation.")
print(f"Shape of selected bcg_signals: {bcg_signals.shape}")

# 保存前 6000 个样本
bcg_signals_filename = os.path.join(output_dir, 'BCG_before_bed.npy')
try:
    np.save(bcg_signals_filename, bcg_signals)
    print(f"Original BCG signals (first 6000 samples) saved to '{bcg_signals_filename}'")
except Exception as e:
    print(f"Error saving BCG signals: {e}")

if bcg_signals.shape[1] != n_points_signal:
    print(f"Warning: Signal length in data ({bcg_signals.shape[1]}) != calculated n_points ({n_points_signal}). Using actual length.")
    n_points_signal = bcg_signals.shape[1]
    duration = n_points_signal / fs
    print(f"Adjusted n_points_signal={n_points_signal}, duration={duration:.2f}s")

simulated_beddot_signals = np.zeros_like(bcg_signals)

# --- 计算主路径的物理频率响应 H_direct(f) ---
travel_time = distance / velocity_ref
freqs = fftfreq(n_points_ir, d=1/fs)
abs_freqs = np.abs(freqs)
abs_freqs[abs_freqs == 0] = 1e-9

attenuation_factor_freq = np.exp(-np.pi * abs_freqs * travel_time / q_factor)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    inv_velocity_f = (1.0 / velocity_ref) * (1.0 + (1.0 / (np.pi * q_factor)) * np.log(abs_freqs / f_ref))
phase_response = -2 * np.pi * freqs * distance * inv_velocity_f
phase_response[freqs == 0] = 0
frequency_response_H_direct = (geometric_spreading_factor *
                               attenuation_factor_freq *
                               np.exp(1j * phase_response))

# --- 计算主路径冲激响应 h_direct(t) ---
h_direct_t_complex = ifft(frequency_response_H_direct)
h_direct_t = np.real(h_direct_t_complex)
# 找到主路径峰值，用于缩放混响幅度
h_direct_peak_amp = np.max(np.abs(h_direct_t))
if h_direct_peak_amp < 1e-9: h_direct_peak_amp = 1.0 # Avoid division by zero

# --- 计算指数衰减噪声混响 h_reverb(t) ---
h_reverb_t = np.zeros(n_points_ir)
# 计算混响开始样本点
expected_delay_samples = int(travel_time * fs)
reverb_start_sample = min(n_points_ir - 1, int(expected_delay_samples * reverb_start_delay_factor))

# 生成白噪声序列
noise = np.random.normal(0, 1, n_points_ir)

# 生成指数衰减包络
envelope = np.zeros(n_points_ir)
# 时间轴从混响开始点计算相对时间 t'
time_since_reverb_start = (np.arange(n_points_ir) - reverb_start_sample) / fs
# 只在混响开始后应用包络
valid_indices = np.where(time_since_reverb_start >= 0)[0]
envelope[valid_indices] = np.exp(-time_since_reverb_start[valid_indices] / reverb_decay_tau)

# 计算混响绝对幅度并应用
reverb_abs_amplitude = reverb_noise_rel_amplitude * h_direct_peak_amp
h_reverb_t = reverb_abs_amplitude * noise * envelope

# --- 组合总冲激响应 h_total(t) ---
h_total_t = h_direct_t + h_reverb_t

# --- 模拟处理循环 ---
print("Simulating BedDot signals using convolution with physical IR + exponential noise reverb...")
for i in tqdm(range(num_samples_to_process), desc="Processing signals"):
    original_signal = bcg_signals[i]

    # 1. 卷积: 应用总冲激响应
    r_t_convolved = convolve(original_signal, h_total_t, mode='same')

    # 2. 添加噪声
    signal_max_abs = np.max(np.abs(r_t_convolved))
    if signal_max_abs < 1e-9: signal_max_abs = 1.0
    actual_noise_std = noise_level * signal_max_abs
    noise_addon = np.random.normal(0, actual_noise_std, n_points_signal) # Renamed to avoid conflict
    noisy_signal = r_t_convolved + noise_addon

    # 3. 归一化 ([0, 1])
    min_val = np.min(noisy_signal)
    max_val = np.max(noisy_signal)
    if max_val > min_val:
        normalized_signal = (noisy_signal - min_val) / (max_val - min_val)
    else:
        normalized_signal = np.zeros_like(noisy_signal)

    simulated_beddot_signals[i] = normalized_signal

# --- 保存结果 ---
try:
    np.save(output_filepath, simulated_beddot_signals)
    print(f"Simulated BedDot signals saved to '{output_filepath}'")
    print(f"Saved array shape: {simulated_beddot_signals.shape}")
except Exception as e:
    print(f"Error saving data to {output_filepath}: {e}")

# --- 可视化 ---
if num_samples_to_process > 0:
    t_axis_signal = np.arange(n_points_signal) / fs
    # t_axis_ir is already defined above

    plt.figure(figsize=(14, 14)) # 调整画布大小

    # 绘制总冲激响应 h_total(t)
    plt.subplot(3, 1, 1)
    plt.plot(ir_time_axis, h_direct_t, label=f'h_direct(t) (Q={q_factor})', color='green', linestyle='--')
    # 绘制混响部分时用较低的 alpha，因为它很密集
    plt.plot(ir_time_axis, h_reverb_t, label=f'h_reverb(t) (Exp. Noise, tau={reverb_decay_tau}, amp={reverb_noise_rel_amplitude:.2f})', color='orange', alpha=0.6)
    plt.plot(ir_time_axis, h_total_t, label='h_total(t) = h_direct + h_reverb', color='black')
    plt.title('Total Bed Impulse Response (Physical Main Path + Exponential Noise Reverb)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, ir_duration_s) # 限制 x 轴范围
    # 调整 Y 轴限制以更好地显示 IR
    ymin = min(h_total_t.min(), h_direct_t.min(), h_reverb_t.min())
    ymax = max(h_total_t.max(), h_direct_t.max(), h_reverb_t.max())
    plt.ylim(ymin * 1.1, ymax * 1.1)
    plt.grid(True)
    plt.legend(fontsize='small')

    # 绘制原始 BCG 信号
    plt.subplot(3, 1, 2)
    plt.plot(t_axis_signal, bcg_signals[1], label='Original BCG Signal (Sample 0)', color='blue')
    plt.title('Original BCG vs. Simulated BedDot Signal (Sample 0)')
    plt.ylabel('Amplitude (Original Scale)')
    plt.grid(True)
    plt.legend()

    # 绘制模拟的 BedDot 信号 (卷积后)
    plt.subplot(3, 1, 3)
    plt.plot(t_axis_signal, simulated_beddot_signals[1], label='Simulated BedDot (Exp. Reverb Conv, Noisy, Norm.)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (Normalized [0, 1])')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No samples processed, skipping visualization.")

print("Script finished.")

