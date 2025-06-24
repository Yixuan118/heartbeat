# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt, find_peaks
import warnings
from scipy.fft import fft, ifft

# 设置中文显示
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 参数配置 ==========
FS = 100  # 采样率 (Hz)
PAD_LENGTH = 200  # 信号填充长度
N_CHANNEL_RESPONSE = 1000  # 通道响应长度
MIN_PEAK_DISTANCE = 50  # find_peaks的参数: 最小峰间距
PEAK_PROMINENCE = 0.1  # find_peaks的参数: 峰值突出度，用于更准确地找峰


# ========== 信号处理工具函数 (增加了一个更鲁棒的找峰函数) ==========
def pad_signal(signal, pad_length=PAD_LENGTH):
    """信号填充函数"""
    if len(signal) == 0: return np.array([])
    return np.pad(signal, (pad_length, pad_length), mode='edge')


def process_signal_with_padding(signal, process_fn, pad_length=PAD_LENGTH):
    """带填充的信号处理"""
    if len(signal) == 0: return np.array([])
    processed = process_fn(pad_signal(signal, pad_length))
    return processed[pad_length:-pad_length]


def highpass_filter(data, cutoff=0.7, fs=FS, order=4):
    """高通滤波器，用于提取心跳分量"""
    if len(data) <= order * 3: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


def extract_respiration(data, cutoff=0.5, fs=FS, order=4):
    """低通滤波器，用于提取呼吸成分"""
    if len(data) <= order * 3: return np.zeros_like(data)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def wavelet_denoise(data, wavelet='db4', level=4):
    """小波去噪"""
    if len(data) < pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet)) + 1 or level <= 0: return data
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
    except ValueError:
        return data
    if not coeffs or not coeffs[-1].any(): return data
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745) if np.median(np.abs(coeffs[-1])) > 1e-9 else 1e-9
    threshold = sigma * np.sqrt(2 * np.log(len(data))) if len(data) > 1 else 0
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    try:
        denoised_data = pywt.waverec(new_coeffs, wavelet, mode='symmetric')
    except ValueError:
        return data
    return denoised_data[:len(data)]


def wiener_deconvolution(R, H, K=0.02):
    """维纳反卷积"""
    len_R, len_H = len(R), len(H)
    if len_R == 0: return np.array([])
    H_fft = fft(H, n=len_R)
    R_fft = fft(R)
    S_fft = (np.conj(H_fft) / (np.abs(H_fft) ** 2 + K)) * R_fft
    return np.real(ifft(S_fft))


def find_signal_peaks(signal, distance=MIN_PEAK_DISTANCE, prominence=PEAK_PROMINENCE):
    """更鲁棒的找峰辅助函数"""
    # 找峰前进行归一化，使得prominence参数的意义更一致
    if np.max(signal) - np.min(signal) < 1e-6: return np.array([], dtype=int)
    signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    peaks, _ = find_peaks(signal_norm, distance=distance, prominence=prominence)
    return peaks


# ========== 模型训练、重建与评估 ==========

def load_training_data():
    """专门加载训练数据"""
    print("加载训练数据...")
    try:
        all_bcg_signals = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-06-21T111240_2025-06-21T111434.npy')
        all_beddot_signals = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-06-21T111240_2025-06-21T111434.npy')
        print(f"成功加载训练数据: BCG {all_bcg_signals.shape}, BedDot {all_beddot_signals.shape}")
        min_len = min(all_bcg_signals.shape[-1], all_beddot_signals.shape[-1])
        return all_bcg_signals[..., :min_len], all_beddot_signals[..., :min_len]
    except FileNotFoundError:
        print("错误: 训练数据文件未找到！请检查路径。")
        return None, None


def load_testing_data():
    """专门加载测试数据"""
    print("\n加载独立的测试数据...")
    try:
        ground_truth_signal = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-06-21T103410_2025-06-21T103740.npy')
        beddot_to_predict = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-06-21T103410_2025-06-21T103740.npy')
        print(f"成功加载测试数据: BCG {ground_truth_signal.shape}, BedDot {beddot_to_predict.shape}")
        return ground_truth_signal, beddot_to_predict
    except FileNotFoundError:
        print("错误: 测试数据文件未找到！请检查路径。")
        return None, None


def train_reconstruction_model(bcg_train, beddot_train):
    """
    训练模型，幅度因子计算采用峰值对齐策略
    """
    print("\n--- 开始模型训练阶段 ---")
    preprocess_fn = lambda x: wavelet_denoise(highpass_filter(x))
    bcg_train_processed = np.array([process_signal_with_padding(s, preprocess_fn) for s in bcg_train])
    beddot_train_processed = np.array([process_signal_with_padding(s, preprocess_fn) for s in beddot_train])

    # 步骤1: 估计平均通道响应 H (逻辑保持不变)
    print("步骤1: 估计平均通道响应 H (整体形态)...")
    H_sum = np.zeros(N_CHANNEL_RESPONSE, dtype=complex)
    count = 0
    for s_bcg, r_beddot in zip(bcg_train_processed, beddot_train_processed):
        if len(s_bcg) != len(r_beddot) or len(s_bcg) == 0: continue
        temp_H_fft = fft(r_beddot) / (fft(s_bcg) + 1e-9)
        temp_H_time = ifft(temp_H_fft)
        H_sum[:min(len(temp_H_time), N_CHANNEL_RESPONSE)] += temp_H_time[:min(len(temp_H_time), N_CHANNEL_RESPONSE)]
        count += 1
    H_avg = np.real(H_sum / count) if count > 0 else np.array([1.0])
    print("通道响应 H 学习完成。")

    # 步骤2: 基于“峰值对齐”计算幅度缩放因子
    print("步骤2: 计算幅度缩放因子...")
    reconstructed_morphology_train = np.array([wiener_deconvolution(r, H_avg) for r in beddot_train_processed])

    true_peak_amps, recon_peak_amps = [], []
    for s_bcg, s_recon in zip(bcg_train_processed, reconstructed_morphology_train):
        # 分别在原始信号和重建信号中找峰
        peaks_true = find_signal_peaks(s_bcg)
        peaks_recon = find_signal_peaks(s_recon)

        # 只有在两者中都找到了峰，才进行统计，以保证比较的公平性
        if len(peaks_true) > 0 and len(peaks_recon) > 0:
            true_peak_amps.extend(s_bcg[peaks_true])
            # 注意：这里要用s_recon的峰值位置，来索引s_recon的值
            recon_peak_amps.extend(s_recon[peaks_recon])

    if true_peak_amps and recon_peak_amps:
        # 计算平均峰高
        avg_true_amp = np.mean(true_peak_amps)
        avg_recon_amp = np.mean(recon_peak_amps)
        # 计算缩放因子
        amplitude_scaling_factor = avg_true_amp / avg_recon_amp if abs(avg_recon_amp) > 1e-6 else 1.0
        print(f"基于 {len(true_peak_amps)} 个真实峰和 {len(recon_peak_amps)} 个重建峰进行计算。")
    else:
        # 如果找不到足够的峰，则回退到原来的方法
        print("警告: 未找到足够的峰值用于对齐，回退到基于标准差的缩放方法。")
        std_original = [np.std(s) for s in bcg_train_processed if np.std(s) > 1e-9]
        std_reconstructed = [np.std(s) for s in reconstructed_morphology_train if np.std(s) > 1e-9]
        if not std_original or not std_reconstructed:
            amplitude_scaling_factor = 1.0
        else:
            amplitude_scaling_factor = np.mean(std_original) / np.mean(std_reconstructed)

    model_params = {'H_channel': H_avg, 'scaling_factor': amplitude_scaling_factor}
    print(f"幅度缩放因子: {amplitude_scaling_factor:.4f}")
    print("--- 模型训练完成 ---")
    return model_params


def reconstruct_signals(beddot_data, model_params):
    """通用重建函数 (无修改)"""
    H_channel = model_params['H_channel']
    scaling_factor = model_params['scaling_factor']

    preprocess_fn = lambda x: wavelet_denoise(highpass_filter(x))
    beddot_processed = np.array([process_signal_with_padding(s, preprocess_fn) for s in beddot_data])
    reconstructed_heart = np.array([wiener_deconvolution(r, H_channel) for r in beddot_processed])
    reconstructed_heart_scaled = reconstructed_heart * scaling_factor
    respiration = np.array([extract_respiration(s) for s in beddot_data])

    final_reconstructed_signals = []
    for heart, resp in zip(reconstructed_heart_scaled, respiration):
        min_len = min(len(heart), len(resp))
        final_reconstructed_signals.append(heart[:min_len] + resp[:min_len])

    return np.array(final_reconstructed_signals)


# ========== 评估与可视化函数 (无修改) ==========

def calculate_overall_signal_metrics(all_true_signals, all_pred_signals):
    """计算整体信号指标"""
    all_mae, all_rmse, all_corrs = [], [], []
    for true_s, pred_s in zip(all_true_signals, all_pred_signals):
        min_len = min(len(true_s), len(pred_s))
        if min_len == 0: continue
        true_s, pred_s = true_s[:min_len], pred_s[:min_len]
        all_mae.append(np.mean(np.abs(true_s - pred_s)))
        all_rmse.append(np.sqrt(np.mean((true_s - pred_s) ** 2)))
        if np.var(true_s) > 1e-9 and np.var(pred_s) > 1e-9:
            all_corrs.append(np.corrcoef(true_s, pred_s)[0, 1])
    return {'MAE': np.nanmean(all_mae), 'RMSE': np.nanmean(all_rmse), 'Correlation': np.nanmean(all_corrs)}


def print_metrics_dict(title, metrics_dict):
    """打印指标字典"""
    print(f"\n--- {title} ---")
    if not metrics_dict: print("  指标字典为空。"); return
    for key, val in metrics_dict.items():
        print(f"  平均 {key}: {val:.4f}" if not np.isnan(val) else f"  {key}: N/A")


def plot_comparison_result(original_bcg, original_beddot, reconstructed_signal, dataset_type, sample_idx=0, fs=FS):
    """通用的对比绘图函数 (无修改)"""
    if not (0 <= sample_idx < len(original_bcg)):
        print(f"错误：无效的样本索引 {sample_idx}。");
        return

    true_s, beddot_s, pred_s = original_bcg[sample_idx], original_beddot[sample_idx], reconstructed_signal[sample_idx]
    min_len = min(len(true_s), len(beddot_s), len(pred_s))
    if min_len == 0: print("错误：样本中存在空信号。"); return
    t = np.arange(min_len) / fs

    plt.figure(figsize=(18, 6))
    title_prefix = f'{dataset_type}信号比较 (样本 {sample_idx})'
    title_suffix = '模型拟合效果' if dataset_type == "训练集" else '盲重建结果'
    title = f'{title_prefix} - {title_suffix}'
    true_label = f'原始BCG ({dataset_type}真值)'
    pred_label = '重建信号 (模型输出)'

    plt.plot(t, true_s[:min_len], label=true_label, color='blue', linewidth=2, linestyle='-')
    plt.plot(t, beddot_s[:min_len], label=f'BedDot信号 ({dataset_type}输入)', color='green', alpha=0.6, linewidth=1.5)
    plt.plot(t, pred_s[:min_len], label=pred_label, color='red', linewidth=2, linestyle='--')

    plt.title(title, fontsize=16);
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度')
    plt.legend(fontsize=12);
    plt.grid(True, linestyle=':');
    plt.tight_layout();
    plt.show()


# ========== 主程序 (无修改) ==========
def main():
    """主程序: 清晰地分离训练、验证和测试"""
    # 1. 训练阶段
    bcg_train, beddot_train = load_training_data()
    if bcg_train is None: return
    model_params = train_reconstruction_model(bcg_train, beddot_train)

    # 2. 训练集验证
    print("\n--- 在训练集上验证模型拟合效果 ---")
    reconstructed_train_signals = reconstruct_signals(beddot_train, model_params)
    train_metrics = calculate_overall_signal_metrics(bcg_train, reconstructed_train_signals)
    print_metrics_dict("训练集拟合性能指标 (峰值优化后)", train_metrics)
    plot_comparison_result(bcg_train, beddot_train, reconstructed_train_signals, dataset_type="训练集", sample_idx=5)

    # 3. 独立测试阶段
    print("\n\n=== 进入独立测试阶段 ===")
    bcg_test, beddot_test = load_testing_data()
    if bcg_test is None: return

    print("\n--- 开始在测试集上进行盲重建 ---")
    reconstructed_test_signals = reconstruct_signals(beddot_test, model_params)
    print("--- 所有测试样本重建完成 ---")

    # 4. 评估阶段
    print("\n--- 开始性能评估阶段 ---")
    test_metrics = calculate_overall_signal_metrics(bcg_test, reconstructed_test_signals)
    print_metrics_dict("测试集泛化性能指标 (峰值优化后)", test_metrics)

    # 5. 可视化
    plot_comparison_result(bcg_test, beddot_test, reconstructed_test_signals, dataset_type="测试集", sample_idx=20)


if __name__ == "__main__":
    main()
