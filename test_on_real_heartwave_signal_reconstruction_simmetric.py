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
PEAK_PROMINENCE = 0.1  # find_peaks的参数: 峰值突出度
MIN_SEGMENT_SAMPLES = 50  # 分段最小样本数（PDF段落1-70）
W_DISTANCE = 0.6  # 距离相似性权重（PDF段落1-94）


# ========== 信号处理工具函数 ==========
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
    """高通滤波器（PDF段落1-6）"""
    if len(data) <= order * 3: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


def extract_respiration(data, cutoff=0.5, fs=FS, order=4):
    """低通滤波器"""
    if len(data) <= order * 3: return np.zeros_like(data)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def wavelet_denoise(data, wavelet='db4', level=4):
    """小波去噪（PDF段落1-70）"""
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
    """更鲁棒的找峰函数"""
    if np.max(signal) - np.min(signal) < 1e-6: return np.array([], dtype=int)
    signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    peaks, _ = find_peaks(signal_norm, distance=distance, prominence=prominence)
    return peaks


# ========== PDF相似性指标核心实现 ==========
def segment_signal(signal, min_samples=MIN_SEGMENT_SAMPLES):
    """
    基于梯度变化的信号分段（模拟PDF段落1-70的PLR方法）
    """
    if len(signal) < 2 * min_samples:
        return [signal], [0, len(signal)]

    gradient = np.abs(np.gradient(signal))
    change_points = [0]
    current = min_samples

    while current < len(signal) - min_samples:
        window = gradient[current:current + min_samples]
        max_idx = np.argmax(window) + current
        change_points.append(max_idx)
        current = max_idx + min_samples

    change_points.append(len(signal))
    return [signal[cp:change_points[i + 1]] for i, cp in enumerate(change_points[:-1])], change_points


def distance_similarity(orig_seg, rec_seg):
    """
    计算单段距离相似性（PDF段落1-84公式(1)）
    """
    min_len = min(len(orig_seg), len(rec_seg))
    if min_len < 2: return 0.0

    mad = np.mean(np.abs(orig_seg[:min_len] - rec_seg[:min_len]))
    ds = -2 / (1 + np.exp(-2.2 * (mad - 5.5))) + 1  # PDF公式(1)
    return ds


def trend_similarity(orig_seg, rec_seg):
    """
    计算单段趋势相似性（PDF段落1-86-1-91）
    """
    min_len = min(len(orig_seg), len(rec_seg))
    if min_len < 2: return 0.0

    t = np.arange(min_len)
    # 线性拟合
    p_orig = np.polyfit(t, orig_seg[:min_len], 1)
    p_rec = np.polyfit(t, rec_seg[:min_len], 1)
    slope_orig, _ = p_orig
    slope_rec = p_rec[0]

    # 均值对齐（PDF段落1-86）
    rec_aligned = rec_seg[:min_len] - (np.mean(rec_seg[:min_len]) - np.mean(orig_seg[:min_len]))
    p_rec_aligned = np.polyfit(t, rec_aligned, 1)
    slope_rec_aligned = p_rec_aligned[0]

    # 计算角度差
    angle_orig = np.arctan(slope_orig)
    angle_rec = np.arctan(slope_rec_aligned)
    angle_diff = np.abs(angle_orig - angle_rec)

    # 计算最大角度（PDF表4）
    max_val = max(np.max(orig_seg), np.max(rec_aligned))
    min_val = min(np.min(orig_seg), np.min(rec_aligned))
    max_slope = (max_val - min_val) / min_len
    max_angle = np.arctan(max_slope) if max_slope > 0 else np.pi / 2

    # 趋势方向判断（PDF段落1-91）
    if slope_orig * slope_rec_aligned >= 0:
        ts = 1 - (angle_diff / max_angle) if max_angle > 1e-9 else 1.0
    else:
        ts = - (angle_diff / max_angle) if max_angle > 1e-9 else -1.0

    return ts


def composite_similarity(ds, ts, w_dist=W_DISTANCE):
    """
    计算复合相似性（PDF段落1-93公式(2)）
    """
    return w_dist * ds + (1 - w_dist) * ts


def calculate_pdf_metrics(original, reconstructed):
    """
    计算PDF定义的全信号相似性指标（含时间归一化）
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return {'DS': 0, 'TS': 0, 'CS': 0}

    # 信号分段（PDF段落1-66）
    orig_segments, _ = segment_signal(original[:min_len])
    rec_segments, _ = segment_signal(reconstructed[:min_len])
    max_segments = min(len(orig_segments), len(rec_segments))

    if max_segments == 0:
        return {'DS': 0, 'TS': 0, 'CS': 0}

    segment_ds, segment_ts, segment_cs = [], [], []
    segment_lengths = []

    for i in range(max_segments):
        seg_orig = orig_segments[i]
        seg_rec = rec_segments[i]
        len_seg = len(seg_orig)
        if len_seg < 2:
            continue

        ds = distance_similarity(seg_orig, seg_rec)
        ts = trend_similarity(seg_orig, seg_rec)
        cs = composite_similarity(ds, ts)

        segment_ds.append(ds)
        segment_ts.append(ts)
        segment_cs.append(cs)
        segment_lengths.append(len_seg)

    if not segment_ds:
        return {'DS': 0, 'TS': 0, 'CS': 0}

    # 时间归一化加权（PDF段落1-96）
    total_length = sum(segment_lengths)
    ds = sum(d * l for d, l in zip(segment_ds, segment_lengths)) / total_length
    ts = sum(t * l for t, l in zip(segment_ts, segment_lengths)) / total_length
    cs = sum(c * l for c, l in zip(segment_cs, segment_lengths)) / total_length

    return {'DS': ds, 'TS': ts, 'CS': cs}


# ========== 模型训练与重建 ==========
def load_training_data():
    """专门加载训练数据"""
    print("加载训练数据...")
    try:
        all_bcg_signals = np.load(
            r'D:\UGA\heartbeat_system\data\bcg_signals.npy')
        all_beddot_signals = np.load(
            r'D:\UGA\heartbeat_system\data\beddot_signals.npy')
        print(f"成功加载训练数据: BCG {all_bcg_signals.shape}, BedDot {all_beddot_signals.shape}")
        min_len = min(all_bcg_signals.shape[-1], all_beddot_signals.shape[-1])
        return all_bcg_signals[..., :min_len], all_beddot_signals[..., :min_len]
    except FileNotFoundError:
        print("错误: 训练数据文件未找到！请检查路径。")
        return None, None


def load_testing_data():
    """加载测试数据"""
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
    """训练重建模型"""
    print("\n--- 开始模型训练阶段 ---")
    preprocess_fn = lambda x: wavelet_denoise(highpass_filter(x))
    bcg_train_processed = np.array([process_signal_with_padding(s, preprocess_fn) for s in bcg_train])
    beddot_train_processed = np.array([process_signal_with_padding(s, preprocess_fn) for s in beddot_train])

    # 估计平均通道响应H
    print("步骤1: 估计平均通道响应 H...")
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

    # 计算幅度缩放因子
    print("步骤2: 计算幅度缩放因子...")
    reconstructed_morphology_train = np.array([wiener_deconvolution(r, H_avg) for r in beddot_train_processed])

    true_peak_amps, recon_peak_amps = [], []
    for s_bcg, s_recon in zip(bcg_train_processed, reconstructed_morphology_train):
        peaks_true = find_signal_peaks(s_bcg)
        peaks_recon = find_signal_peaks(s_recon)
        if len(peaks_true) > 0 and len(peaks_recon) > 0:
            true_peak_amps.extend(s_bcg[peaks_true])
            recon_peak_amps.extend(s_recon[peaks_recon])

    if true_peak_amps and recon_peak_amps:
        avg_true_amp = np.mean(true_peak_amps)
        avg_recon_amp = np.mean(recon_peak_amps)
        amplitude_scaling_factor = avg_true_amp / avg_recon_amp if abs(avg_recon_amp) > 1e-6 else 1.0
        print(f"基于 {len(true_peak_amps)} 个真实峰和 {len(recon_peak_amps)} 个重建峰计算。")
    else:
        print("警告: 未找到足够峰值，回退到标准差缩放。")
        std_original = [np.std(s) for s in bcg_train_processed if np.std(s) > 1e-9]
        std_reconstructed = [np.std(s) for s in reconstructed_morphology_train if np.std(s) > 1e-9]
        amplitude_scaling_factor = np.mean(std_original) / np.mean(
            std_reconstructed) if std_original and std_reconstructed else 1.0

    model_params = {'H_channel': H_avg, 'scaling_factor': amplitude_scaling_factor}
    print(f"幅度缩放因子: {amplitude_scaling_factor:.4f}")
    print("--- 模型训练完成 ---")
    return model_params


def reconstruct_signals(beddot_data, model_params):
    """信号重建"""
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


# ========== 评估与可视化 ==========
def calculate_overall_metrics(original_signals, reconstructed_signals):
    """计算整体评估指标"""
    all_mae, all_rmse, all_corr = [], [], []
    all_ds, all_ts, all_cs = [], [], []

    for true_s, pred_s in zip(original_signals, reconstructed_signals):
        min_len = min(len(true_s), len(pred_s))
        if min_len == 0:
            continue

        true_s, pred_s = true_s[:min_len], pred_s[:min_len]
        mae = np.mean(np.abs(true_s - pred_s))
        rmse = np.sqrt(np.mean((true_s - pred_s) ** 2))
        corr = np.corrcoef(true_s, pred_s)[0, 1] if np.var(true_s) > 1e-9 and np.var(pred_s) > 1e-9 else 0

        pdf_metrics = calculate_pdf_metrics(true_s, pred_s)

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_corr.append(corr)
        all_ds.append(pdf_metrics['DS'])
        all_ts.append(pdf_metrics['TS'])
        all_cs.append(pdf_metrics['CS'])

    return {
        'MAE': np.nanmean(all_mae),
        'RMSE': np.nanmean(all_rmse),
        'Correlation': np.nanmean(all_corr),
        'DistanceSimilarity': np.nanmean(all_ds),
        'TrendSimilarity': np.nanmean(all_ts),
        'CompositeSimilarity': np.nanmean(all_cs)
    }


def print_evaluation_metrics(title, metrics):
    """打印评估指标（含PDF相似性指标）"""
    print(f"\n--- {title} ---")
    print("  传统指标:")
    print(f"    MAE: {metrics['MAE']:.4f} mmHg")
    print(f"    RMSE: {metrics['RMSE']:.4f} mmHg")
    print(f"    相关系数: {metrics['Correlation']:.4f}")

    print("\n  PDF相似性指标（范围[-1,1]）:")
    print(f"    距离相似性(DS): {metrics['DistanceSimilarity']:.4f} "
          f"（≥0表示距离可靠，越接近1误差越小）")
    print(f"    趋势相似性(TS): {metrics['TrendSimilarity']:.4f} "
          f"（≥0表示趋势同向，越接近1一致性越高）")
    print(f"    复合相似性(CS): {metrics['CompositeSimilarity']:.4f} "
          f"（≥0表示具备变化跟踪能力，越接近1综合表现越好）")


def plot_signal_comparison(original_bcg, original_beddot, reconstructed_signal, dataset_type, sample_idx=0):
    """绘制信号对比图"""
    if not (0 <= sample_idx < len(original_bcg)):
        print(f"错误：无效的样本索引 {sample_idx}。");
        return

    true_s, beddot_s, pred_s = original_bcg[sample_idx], original_beddot[sample_idx], reconstructed_signal[sample_idx]
    min_len = min(len(true_s), len(beddot_s), len(pred_s))
    if min_len == 0: print("错误：样本中存在空信号。"); return
    t = np.arange(min_len) / FS

    plt.figure(figsize=(18, 6))
    title_prefix = f'{dataset_type}信号比较 (样本 {sample_idx})'
    title_suffix = '模型拟合效果' if dataset_type == "训练集" else '盲重建结果'
    title = f'{title_prefix} - {title_suffix}'

    plt.plot(t, true_s[:min_len], 'b-', label='原始信号', linewidth=2)
    plt.plot(t, beddot_s[:min_len], 'g--', label='输入信号', alpha=0.6, linewidth=1.5)
    plt.plot(t, pred_s[:min_len], 'r-', label='重建信号', linewidth=2)

    plt.title(title, fontsize=16);
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度')
    plt.legend(fontsize=12);
    plt.grid(True, linestyle=':');
    plt.tight_layout();
    plt.show()


def plot_similarity_analysis(original, reconstructed):
    """绘制PDF风格的相似性分析图"""
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        print("信号长度不足，无法绘制分析图")
        return

    t = np.arange(min_len) / FS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    # 信号对比与分段点
    orig_segments, change_points = segment_signal(original[:min_len])
    ax1.plot(t, original[:min_len], 'b-', label='原始信号', linewidth=2)
    ax1.plot(t, reconstructed[:min_len], 'r--', label='重建信号', linewidth=2)

    for cp in change_points:
        if cp < min_len:
            ax1.axvline(x=t[cp], color='gray', linestyle='--', alpha=0.5)

    ax1.set_title('信号对比及分段', fontsize=16)
    ax1.set_xlabel('时间 (秒)');
    ax1.set_ylabel('幅度')
    ax1.legend(fontsize=12);
    ax1.grid(True, linestyle=':')

    # 趋势相似性分析
    pdf_metrics = calculate_pdf_metrics(original, reconstructed)
    orig_segments, _ = segment_signal(original[:min_len])
    rec_segments, _ = segment_signal(reconstructed[:min_len])
    max_segments = min(len(orig_segments), len(rec_segments))

    t_seg = np.arange(MIN_SEGMENT_SAMPLES) / FS
    for i in range(max_segments):
        seg_orig = orig_segments[i]
        seg_rec = rec_segments[i]
        len_seg = len(seg_orig)
        if len_seg < 2:
            continue

        p_orig = np.polyfit(np.arange(len_seg), seg_orig, 1)
        p_rec = np.polyfit(np.arange(len_seg), seg_rec, 1)
        rec_aligned = seg_rec - (np.mean(seg_rec) - np.mean(seg_orig))
        p_rec_aligned = np.polyfit(np.arange(len_seg), rec_aligned, 1)

        slope_orig = p_orig[0]
        slope_rec_aligned = p_rec_aligned[0]
        color = 'g-' if slope_orig * slope_rec_aligned >= 0 else 'r-'

        ax2.plot(t_seg[:len_seg], np.polyval(p_orig, np.arange(len_seg)), color, alpha=0.7)
        ax2.plot(t_seg[:len_seg], np.polyval(p_rec_aligned, np.arange(len_seg)), color, linestyle='--', alpha=0.7)

    ax2.set_title(f'分段趋势相似性分析 (CS={pdf_metrics["CompositeSimilarity"]:.4f})', fontsize=16)
    ax2.set_xlabel('时间 (秒)');
    ax2.set_ylabel('趋势拟合值')
    ax2.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


# ========== 主程序 ==========
def main():
    """主程序：集成PDF相似性指标的完整评估流程"""
    # 1. 训练阶段
    bcg_train, beddot_train = load_training_data()
    if bcg_train is None: return
    model_params = train_reconstruction_model(bcg_train, beddot_train)

    # 2. 训练集评估
    print("\n--- 训练集评估 ---")
    reconstructed_train = reconstruct_signals(beddot_train, model_params)
    train_metrics = calculate_overall_metrics(bcg_train, reconstructed_train)
    print_evaluation_metrics("训练集性能指标", train_metrics)
    plot_signal_comparison(bcg_train, beddot_train, reconstructed_train, "训练集", sample_idx=5)
    plot_similarity_analysis(bcg_train[5], reconstructed_train[5])

    # 3. 测试集评估
    print("\n\n=== 测试集评估 ===")
    bcg_test, beddot_test = load_testing_data()
    if bcg_test is None: return

    print("\n--- 测试集信号重建 ---")
    reconstructed_test = reconstruct_signals(beddot_test, model_params)
    print("--- 重建完成，开始评估 ---")

    test_metrics = calculate_overall_metrics(bcg_test, reconstructed_test)
    print_evaluation_metrics("测试集性能指标", test_metrics)
    plot_signal_comparison(bcg_test, beddot_test, reconstructed_test, "测试集", sample_idx=20)
    plot_similarity_analysis(bcg_test[20], reconstructed_test[20])


if __name__ == "__main__":
    main()