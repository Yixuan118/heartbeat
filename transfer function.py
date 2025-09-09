import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.font_manager as fm
from scipy import signal  # 可用于交叉相关

# --- 配置matplotlib以支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'DejaVu Sans']  # 尝试多种字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
print("Matplotlib已配置为尝试显示中文。")

# 数据文件路径
train_on_bed_file    = r"D:\UGA\heartbeat_system(1)\vibration_analysis.npy"
train_under_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-09-06T185308_2025-09-06T193848.npy"
test_on_bed_file     = r"D:\UGA\heartbeat_system(1)\heartbeat_analysis.npy"
test_under_bed_file  = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-09-06T203856_2025-09-06T204856.npy"

# 参数设置
fs = 100  # 采样率 (Hz)
sample_duration = 10  # 每个样本的持续时间 (秒)
N = fs * sample_duration  # 每个样本的数据点数

# --- 1. 训练数据加载与预处理 ---
print("--- 1. 数据加载与预处理（训练集） ---")
try:
    train_on_bed_raw    = np.load(train_on_bed_file)
    train_under_bed_raw = np.load(train_under_bed_file)
    print("训练数据文件加载成功。")
except FileNotFoundError as e:
    print(f"错误：训练文件加载失败，请检查文件路径是否正确。错误信息: {e}")
    raise SystemExit

# 删除首尾样本以规避边界异常（若样本数>2）
if train_on_bed_raw.shape[0] > 2:
    train_on_bed = train_on_bed_raw[1:-1, :]
    train_under_bed = train_under_bed_raw[1:-1, :]
    print(f"训练数据已删除首尾样本，剩余 {train_on_bed.shape[0]} 个样本。")
else:
    print("警告：训练数据样本数不足（<=2），未进行首尾样本删除。")
    train_on_bed = train_on_bed_raw
    train_under_bed = train_under_bed_raw

# 长度检查
if train_on_bed.shape[1] != N:
    print(f"警告：训练数据样本长度不匹配。预期 {N}，实际 {train_on_bed.shape[1]}。")

# --- 辅助函数 ---
def _shift_signal(sig, shift_samples):
    """将信号平移 shift_samples 个点；右移=延迟，左移=提前；越界补零。"""
    L = len(sig)
    if shift_samples == 0:
        return sig.copy()
    out = np.zeros_like(sig)
    if shift_samples > 0:
        if shift_samples >= L:
            return out
        out[shift_samples:] = sig[:-shift_samples]
    else:
        s = -shift_samples
        if s >= L:
            return out
        out[:L - s] = sig[s:]
    return out

def moving_average(sig, window_size=11):
    window = np.ones(window_size) / window_size
    return np.convolve(sig, window, mode='same')

# --- 2. 传递函数估计（训练阶段）---
print("\n--- 2. 传递函数估计（训练阶段）---")
num_train_samples = train_on_bed.shape[0]
print(f"将使用 {num_train_samples} 个训练样本进行传递函数估计。")

S_xy_accumulator = np.zeros(N, dtype=complex)  # Y * conj(U)
S_xx_accumulator = np.zeros(N, dtype=complex)  # U * conj(U)

for i in range(num_train_samples):
    u_train = train_under_bed[i, :]
    y_train = train_on_bed[i, :]

    U_fft = np.fft.fft(u_train)
    Y_fft = np.fft.fft(y_train)

    S_xy_accumulator += Y_fft * np.conj(U_fft)
    S_xx_accumulator += U_fft * np.conj(U_fft)

min_denominator_threshold = 1e-10
zero_input_freqs = np.abs(S_xx_accumulator) < min_denominator_threshold
H_est = np.zeros(N, dtype=complex)
H_est[~zero_input_freqs] = S_xy_accumulator[~zero_input_freqs] / S_xx_accumulator[~zero_input_freqs]

print("传递函数估计完成。")

# --- 3. 训练集：信号重建与可视化 ---
print("\n--- 3. 训练集：信号重建与可视化 ---")
num_eval_samples_train = train_on_bed.shape[0]
print(f"将使用 {num_eval_samples_train} 个训练样本进行重建与可视化。")

# 训练集的可视化对齐（保持原设定：提前 0.6 s）
train_shift_seconds = -0.6
train_shift_samples = int(train_shift_seconds * fs)
print(f"训练集可视化：")

# 选取绘图样本
sample_indices_to_plot_train = [0, num_eval_samples_train // 2, num_eval_samples_train - 1]
if num_eval_samples_train < 3:
    sample_indices_to_plot_train = list(range(num_eval_samples_train))
sample_indices_to_plot_train = [idx for idx in sample_indices_to_plot_train if idx < num_eval_samples_train]

total_subplots_per_sample = 3
plt.figure(figsize=(15, 6 * len(sample_indices_to_plot_train) * total_subplots_per_sample))
plt.suptitle(f'训练集：振动信号重建对比',
             fontsize=16, y=1.00)

time = np.arange(N) / fs

for i, sample_k in enumerate(sample_indices_to_plot_train):
    u_eval = train_under_bed[sample_k, :]
    y_eval_original_raw = train_on_bed[sample_k, :]  # 未平滑原始（用来算相关系数）
    y_eval_original_smoothed = moving_average(y_eval_original_raw, window_size=11)  # 仅用于显示

    # 重建
    U_eval_fft = np.fft.fft(u_eval)
    Y_reconstructed_fft = H_est * U_eval_fft
    y_reconstructed_unaligned = np.real(np.fft.ifft(Y_reconstructed_fft))
    y_reconstructed_aligned = _shift_signal(y_reconstructed_unaligned, train_shift_samples)

    # 相关系数：与“未平滑的原始信号”对比（按你的要求）
    corr_coeff_final, _ = pearsonr(y_eval_original_raw, y_reconstructed_aligned)
    print(f"训练样本 {sample_k+1} - （相关系数: {corr_coeff_final:.3f}")

    # 子图1：仅显示平滑后的原始床上传感器信号
    ax1 = plt.subplot(len(sample_indices_to_plot_train) * total_subplots_per_sample, 1, i * total_subplots_per_sample + 1)
    ax1.plot(time, y_eval_original_smoothed, label='床上传感器信号', color='blue')
    ax1.set_title(f'训练样本 {sample_k+1} - 床上传感器信号', fontsize=12)
    ax1.set_xlabel('时间 (秒)', fontsize=10)
    ax1.set_ylabel('信号强度', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True)

    # 子图2：床下输入
    ax2 = plt.subplot(len(sample_indices_to_plot_train) * total_subplots_per_sample, 1, i * total_subplots_per_sample + 2)
    ax2.plot(time, u_eval, label='床下传感器信号 (输入)', color='orange')
    ax2.set_title(f'训练样本 {sample_k+1} - 床下传感器信号 (输入)', fontsize=12)
    ax2.set_xlabel('时间 (秒)', fontsize=10)
    ax2.set_ylabel('信号强度', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True)

    # 子图3：重建信号（对齐后）
    ax3 = plt.subplot(len(sample_indices_to_plot_train) * total_subplots_per_sample, 1, i * total_subplots_per_sample + 3)
    ax3.plot(time, y_reconstructed_aligned, label=f'重建床上传感器信号 (相关系数: {corr_coeff_final:.3f})', color='green', linestyle='--')
    ax3.set_title(f'训练样本 {sample_k+1} - 重建床上传感器信号', fontsize=12)
    ax3.set_xlabel('时间 (秒)', fontsize=10)
    ax3.set_ylabel('信号强度', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("\n--- 训练集可视化完成 ---")

# --- 4. 测试集：加载、（不平滑的）可视化与评估 ---
print("\n--- 4. 测试集：加载与可视化（不进行平滑） ---")
try:
    test_on_bed_raw    = np.load(test_on_bed_file)
    test_under_bed_raw = np.load(test_under_bed_file)
    print("测试数据文件加载成功。")
except FileNotFoundError as e:
    print(f"错误：测试文件加载失败，请检查文件路径是否正确。错误信息: {e}")
    raise SystemExit

# 与训练集一致：若样本数>2则删除首尾样本（如不需要，可改为直接使用原数组）
if test_on_bed_raw.shape[0] > 2:
    test_on_bed = test_on_bed_raw[1:-1, :]
    test_under_bed = test_under_bed_raw[1:-1, :]
    print(f"测试数据已删除首尾样本，剩余 {test_on_bed.shape[0]} 个样本。")
else:
    print("警告：测试数据样本数不足（<=2），未进行首尾样本删除。")
    test_on_bed = test_on_bed_raw
    test_under_bed = test_under_bed_raw

# 长度检查
if test_on_bed.shape[1] != N:
    print(f"警告：测试数据样本长度不匹配。预期 {N}，实际 {test_on_bed.shape[1]}。")

num_eval_samples_test = test_on_bed.shape[0]
print(f"将使用 {num_eval_samples_test} 个测试样本进行重建与可视化。")

# 测试集的可视化对齐：按你的要求改为提前 0.7 s
test_shift_seconds = -0.7
test_shift_samples = int(test_shift_seconds * fs)
print(f"测试集可视化：")

# 选取绘图样本
sample_indices_to_plot_test = [0, num_eval_samples_test // 2, num_eval_samples_test - 1]
if num_eval_samples_test < 3:
    sample_indices_to_plot_test = list(range(num_eval_samples_test))
sample_indices_to_plot_test = [idx for idx in sample_indices_to_plot_test if idx < num_eval_samples_test]

plt.figure(figsize=(15, 6 * len(sample_indices_to_plot_test) * total_subplots_per_sample))
plt.suptitle(f'测试集：振动信号重建对比',
             fontsize=16, y=1.00)

for i, sample_k in enumerate(sample_indices_to_plot_test):
    u_eval = test_under_bed[sample_k, :]
    y_eval_original = test_on_bed[sample_k, :]

    # 重建
    U_eval_fft = np.fft.fft(u_eval)
    Y_reconstructed_fft = H_est * U_eval_fft
    y_reconstructed_unaligned = np.real(np.fft.ifft(Y_reconstructed_fft))
    y_reconstructed_aligned = _shift_signal(y_reconstructed_unaligned, test_shift_samples)

    # 相关系数：
    corr_coeff_final, _ = pearsonr(y_eval_original, y_reconstructed_aligned)
    print(f"测试样本 {sample_k+1} - 相关系数: {corr_coeff_final:.3f}")

    # 子图1：原始床上传感器信号（不平滑）
    ax1 = plt.subplot(len(sample_indices_to_plot_test) * total_subplots_per_sample, 1, i * total_subplots_per_sample + 1)
    ax1.plot(time, y_eval_original, label='床上传感器信号(原始)', color='blue')
    ax1.set_title(f'测试样本 {sample_k+1} - 床上传感器信号(原始)', fontsize=12)
    ax1.set_xlabel('时间 (秒)', fontsize=10)
    ax1.set_ylabel('信号强度', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True)

    # 子图2：床下输入
    ax2 = plt.subplot(len(sample_indices_to_plot_test) * total_subplots_per_sample, 1, i * total_subplots_per_sample + 2)
    ax2.plot(time, u_eval, label='床下传感器信号 (输入)', color='orange')
    ax2.set_title(f'测试样本 {sample_k+1} - 床下传感器信号 (输入)', fontsize=12)
    ax2.set_xlabel('时间 (秒)', fontsize=10)
    ax2.set_ylabel('信号强度', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True)

    # 子图3：重建信号（对齐后）
    ax3 = plt.subplot(len(sample_indices_to_plot_test) * total_subplots_per_sample, 1, i * total_subplots_per_sample + 3)
    ax3.plot(time, y_reconstructed_aligned, label=f'重建床上传感器信号 (相关系数: {corr_coeff_final:.3f})', color='green', linestyle='--')
    ax3.set_title(f'测试样本 {sample_k+1} - 重建床上传感器信号', fontsize=12)
    ax3.set_xlabel('时间 (秒)', fontsize=10)
    ax3.set_ylabel('信号强度', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("\n--- 测试集可视化完成（未对原始信号做平滑；提前 0.7s） ---")
