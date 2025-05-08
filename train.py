import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal.windows import gaussian
from scipy.signal import butter, filtfilt

# ====== Padding 工具函数 ======
def pad_signal(signal, pad_length=200):
    return np.concatenate((signal[pad_length-1::-1], signal, signal[-1:-pad_length-1:-1]))

def process_signal_with_padding(signal, process_fn, pad_length=200):
    padded = pad_signal(signal, pad_length)
    processed = process_fn(padded)
    return processed[pad_length:-pad_length]

# 高通滤波器（去除呼吸信号）
def highpass_filter(data, cutoff=0.5, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# 低通滤波器（提取呼吸信号）
def extract_respiration(data, cutoff=0.5, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# 小波去噪
def wavelet_denoise(data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(data)]

# 卡尔曼滤波
def kalman_filter(data, process_noise=0.1, measurement_noise=1.0):
    n = len(data)
    x_hat = np.zeros(n)
    P = np.ones(n)
    Q = process_noise
    R = measurement_noise
    x_hat[0] = data[0]
    for t in range(1, n):
        x_hat_minus = x_hat[t - 1]
        P_minus = P[t - 1] + Q
        K = P_minus / (P_minus + R)
        x_hat[t] = x_hat_minus + K * (data[t] - x_hat_minus)
        P[t] = (1 - K) * P_minus
    return x_hat

# NLMS 自适应滤波
def nlms_adaptive_filter(R, S, filter_length=100, mu=0.01):
    h = np.zeros(filter_length)
    n = len(R)
    S_reconstructed = np.zeros(n)
    for i in range(filter_length, n):
        R_window = R[i - filter_length:i][::-1]
        S_reconstructed[i] = np.dot(h, R_window)
        error = S[i] - S_reconstructed[i]
        norm = np.dot(R_window, R_window) + 1e-10
        h += (mu / norm) * error * R_window
    return S_reconstructed

# Wiener反卷积
def wiener_deconvolution(R, H, K=0.01):
    H_fft = np.fft.fft(H, n=len(R))
    R_fft = np.fft.fft(R)
    H_conj = np.conj(H_fft)
    S_fft = (H_conj / (H_fft * H_conj + K)) * R_fft
    return np.fft.ifft(S_fft).real

# 平滑和校正
def smooth_and_correct(S_reconstructed, S_original, window_size=11, segment_size=100):
    window = gaussian(window_size, 2)
    pad_len = window_size // 2
    padded = np.pad(S_reconstructed, (pad_len,), mode='reflect')
    S_smoothed = np.convolve(padded, window / window.sum(), mode='valid')
    n = len(S_smoothed)
    for start in range(0, n, segment_size):
        end = min(start + segment_size, n)
        rms_original = np.sqrt(np.mean(S_original[start:end] ** 2))
        rms_smoothed = np.sqrt(np.mean(S_smoothed[start:end] ** 2))
        if rms_smoothed != 0:
            S_smoothed[start:end] *= (rms_original / rms_smoothed)
    bias = np.mean(S_original[:100]) - np.mean(S_smoothed[:100])
    S_smoothed[:100] += bias
    return S_smoothed

# ========== 主逻辑 ==========
fs = 100
N = 1000
pad_len = 200

# 加载数据
all_bcg_signals = np.load('./data/BCG_before_bed.npy')[:6000]
all_beddot_signals = np.load('./data/BCG_after_bed.npy')[:6000]

# 分为训练和测试集
train_bcg = all_bcg_signals[:1000]
test_bcg = all_bcg_signals[1000:]

train_beddot = all_beddot_signals[:1000]
test_beddot = all_beddot_signals[1000:]

# 训练集预处理（包含padding）
train_bcg_filtered = np.array([
    process_signal_with_padding(sig, lambda x: highpass_filter(x), pad_len) for sig in train_bcg])
train_beddot_filtered = np.array([
    process_signal_with_padding(sig, lambda x: highpass_filter(x), pad_len) for sig in train_beddot])
train_beddot_denoised = np.array([
    process_signal_with_padding(sig, lambda x: kalman_filter(wavelet_denoise(x)), pad_len) for sig in train_beddot_filtered])

# 选择最优参数 mu 和 filter_length
mu_values = [0.01, 0.1, 0.5]
filter_length_values = [100, 200]
min_mae = float('inf')

for mu in mu_values:
    for fl in filter_length_values:
        reconstructed = np.zeros_like(train_beddot)
        for i in range(1000):
            reconstructed[i] = nlms_adaptive_filter(train_beddot_denoised[i], train_bcg_filtered[i], fl, mu)
        mae = np.mean([np.mean(np.abs(train_bcg_filtered[i][fs:] - reconstructed[i][fs:])) for i in range(1000)])
        if mae < min_mae:
            min_mae = mae
            best_mu = mu
            best_fl = fl

print(f"Best params from training: mu={best_mu}, filter_length={best_fl}")

# 估计平均通道响应 H
H_avg = np.zeros(N)
for i in range(1000):
    S = train_bcg_filtered[i]
    R = train_beddot_denoised[i]
    H = np.convolve(R, S[::-1], mode='full')[N - 1:2 * N - 1] / (np.sum(S ** 2) + 1e-10)
    H_avg += H
H_avg /= 1000

# 测试集预处理（包含padding）
test_bcg_filtered = np.array([
    process_signal_with_padding(sig, lambda x: highpass_filter(x), pad_len) for sig in test_bcg])
test_beddot_filtered = np.array([
    process_signal_with_padding(sig, lambda x: highpass_filter(x), pad_len) for sig in test_beddot])
test_beddot_denoised = np.array([
    process_signal_with_padding(sig, lambda x: kalman_filter(wavelet_denoise(x)), pad_len) for sig in test_beddot_filtered])

# 重建测试集信号
reconstructed_signals_test = np.zeros_like(test_beddot)
for i in range(5000):
    nlms_output = nlms_adaptive_filter(test_beddot_denoised[i], test_bcg_filtered[i], filter_length=best_fl, mu=best_mu)
    wiener_output = wiener_deconvolution(test_beddot_denoised[i], H_avg)
    combined = 0.5 * nlms_output + 0.5 * wiener_output
    final_output = smooth_and_correct(combined, test_bcg_filtered[i])
    reconstructed_signals_test[i] = final_output

# 加回呼吸信号
reconstructed_with_resp_test = np.zeros_like(reconstructed_signals_test)
for i in range(5000):
    respiration = extract_respiration(test_bcg[i])  # 从原始BCG中提取
    reconstructed_with_resp_test[i] = reconstructed_signals_test[i] + respiration

# 评估 MAE
mae_list = [np.mean(np.abs(test_bcg_filtered[i] - reconstructed_signals_test[i])) for i in range(5000)]
avg_mae = np.mean(mae_list)
print(f"Test Set Average MAE (5000 samples): {avg_mae:.6f}")

# ========== 可视化 ==========
t = np.arange(0, 10, 1 / fs)

plt.figure(figsize=(12, 3))
plt.plot(t, test_bcg[2], label='Original BCG (with Respiration)', alpha=0.8)
plt.plot(t, test_beddot[2], label='Original BedDot', alpha=0.6)
plt.plot(t, reconstructed_with_resp_test[2], label='Reconstructed BCG + Resp', linewidth=2, linestyle='--')
plt.title('Signal Comparison (Test Sample 2)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
