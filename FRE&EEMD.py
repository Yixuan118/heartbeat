import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from PyEMD import EEMD

# --- Matplotlib 中文字体设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("警告：无法设置中文字体。")

# =============================================================================
# 1. 配置
# =============================================================================
fs = 100
segment_length_seconds = 10
samples_per_segment = int(fs * segment_length_seconds)
channel_to_analyze = 0
lowcut_freq = 4.5
highcut_freq = 9.5
filter_order = 5
CHOSEN_FRF_ESTIMATOR = 'Hv_geometric_mean'
FFT_DENOISE_THRESHOLD_RATIO = 0.35

# 文件路径
try:
    train_on_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-10-04T161532_2025-10-04T162821.npy"
    train_under_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-10-04T161532_2025-10-04T162821.npy"
    test_on_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-09-15T233703_2025-09-15T234233.npy"
    test_under_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-09-15T233703_2025-09-15T234233.npy"
except Exception:
    print("警告：默认文件路径无效...")


# =============================================================================
# 2. 辅助函数 (与上一版一致)
# =============================================================================
def load_and_preprocess_single_channel(file_path, *args):
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}"); return None
    if data.ndim != 2: return None
    target_channel, expected_segment_len_per_channel, samples_to_remove_from_ends = args[1], args[2], args[0]
    if data.shape[1] % expected_segment_len_per_channel != 0: return None
    num_channels_in_file = data.shape[1] // expected_segment_len_per_channel
    if num_channels_in_file <= target_channel: return None
    selected_channel_segments = data[:, target_channel * expected_segment_len_per_channel:(
                                                                                                      target_channel + 1) * expected_segment_len_per_channel]
    print(
        f"文件 {file_path.split('/')[-1].split('\\')[-1]}：加载了 {len(selected_channel_segments)} 个{segment_length_seconds}s段。")
    if selected_channel_segments.shape[1] < 2 * samples_to_remove_from_ends:
        return selected_channel_segments
    else:
        return selected_channel_segments[:,
               samples_to_remove_from_ends:selected_channel_segments.shape[1] - samples_to_remove_from_ends]


def align_segments_cross_correlation(signal_ref, signal_target, max_lag_samples):
    min_len_initial = min(len(signal_ref), len(signal_target));
    signal_ref_trunc, signal_target_trunc = signal_ref[:min_len_initial], signal_target[:min_len_initial]
    n = len(signal_ref_trunc)
    if n == 0: return np.array([]), np.array([]), 0
    correlation = signal.correlate(signal_target_trunc, signal_ref_trunc, mode='full');
    lags = signal.correlation_lags(n, n, mode='full');
    lag = lags[np.argmax(correlation)]
    if abs(lag) > max_lag_samples: lag = np.sign(lag) * max_lag_samples
    if abs(lag) >= n: return np.array([]), np.array([]), lag
    if lag > 0:
        aligned_ref, aligned_target = signal_ref_trunc[:-lag], signal_target_trunc[lag:]
    elif lag < 0:
        aligned_ref, aligned_target = signal_ref_trunc[abs(lag):], signal_target_trunc[:-abs(lag)]
    else:
        aligned_ref, aligned_target = signal_ref_trunc, signal_target_trunc
    min_len = min(len(aligned_ref), len(aligned_target))
    return aligned_ref[:min_len], aligned_target[:min_len], lag


def estimate_frf_H(input_segments, output_segments, fs, nfft):
    if len(input_segments) != len(output_segments) or len(input_segments) == 0: raise ValueError(
        "Segments must be non-empty.")
    actual_segment_len = input_segments[0].shape[0];
    window = signal.windows.hann(actual_segment_len)
    S_xx_sum = np.zeros(nfft // 2 + 1, dtype=complex);
    S_yy_sum = np.zeros(nfft // 2 + 1, dtype=complex);
    S_yx_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    for i in range(len(input_segments)):
        x, y = input_segments[i] * window, output_segments[i] * window
        X_f, Y_f = fft(x, n=nfft), fft(y, n=nfft)
        S_xx_sum += np.conj(X_f[:nfft // 2 + 1]) * X_f[:nfft // 2 + 1];
        S_yy_sum += np.conj(Y_f[:nfft // 2 + 1]) * Y_f[:nfft // 2 + 1];
        S_yx_sum += Y_f[:nfft // 2 + 1] * np.conj(X_f[:nfft // 2 + 1])
    n_avg = len(input_segments);
    S_xx_avg, S_yy_avg, S_yx_avg = S_xx_sum / n_avg, S_yy_sum / n_avg, S_yx_sum / n_avg
    epsilon = 1e-12 * np.max(np.abs(S_xx_avg));
    epsilon_yx = 1e-12 * np.max(np.abs(S_yx_avg))
    H1_freq = S_yx_avg / (S_xx_avg + epsilon);
    H2_freq = S_yy_avg / (np.conj(S_yx_avg) + epsilon_yx)
    Hv_freq = np.sqrt(H1_freq * H2_freq);
    Hv_freq[np.isnan(Hv_freq) | np.isinf(Hv_freq)] = 0
    freqs = np.fft.fftfreq(nfft, d=1 / fs)[:nfft // 2 + 1]
    return Hv_freq, freqs


def reconstruct_signal_from_frf(input_segments, H_freq_estimator, nfft, seg_len):
    predicted = [];
    H_full = np.concatenate((H_freq_estimator, np.conj(H_freq_estimator[-2:0:-1])))
    for x in input_segments:
        y_pred = ifft(H_full * fft(x, n=nfft));
        predicted.append(np.real(y_pred[:seg_len]))
    return predicted


def butter_bandpass(low, high, fs, order=5):
    nyq = 0.5 * fs;
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band');
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    if segments is None: return None
    b, a = butter_bandpass(low, high, fs, order=order)
    return [signal.filtfilt(b, a, seg) for seg in segments]


def fft_denoise(signal, threshold_ratio):
    fft_result = fft(signal);
    spectrum = np.abs(fft_result)
    threshold = threshold_ratio * np.max(spectrum)
    fft_result[spectrum < threshold] = 0
    denoised_signal = np.real(ifft(fft_result))
    return denoised_signal


def apply_eemd_pipeline(segments, fft_threshold_ratio):
    processed_segments = [];
    eemd = EEMD();
    eemd.trials = 50
    for i, seg in enumerate(segments):
        print(f"\r正在使用EEMD处理段 {i + 1}/{len(segments)}...", end="")
        seg_centered = seg - np.mean(seg);
        IMFs = eemd.eemd(seg_centered)
        filtered_emd = seg_centered - IMFs[0]
        denoised_signal = fft_denoise(filtered_emd, threshold_ratio=fft_threshold_ratio)
        processed_segments.append(denoised_signal)
    print("\nEEMD处理完成。")
    return processed_segments


def calculate_metrics(true, pred, method_name):
    print(f"\n--- {method_name} ---")
    true_flat = np.concatenate(true);
    pred_flat = np.concatenate(pred)
    min_len = min(len(true_flat), len(pred_flat));
    true_flat, pred_flat = true_flat[:min_len], pred_flat[:min_len]
    if np.std(true_flat) == 0 or np.std(pred_flat) == 0:
        corr = np.nan
    else:
        corr, _ = pearsonr(true_flat, pred_flat)
    mae = np.mean(np.abs(true_flat - pred_flat));
    mse = np.mean((true_flat - pred_flat) ** 2)
    mean_abs_true = np.mean(np.abs(true_flat));
    amplitude_error = (mae / mean_abs_true if mean_abs_true > 1e-9 else np.inf) * 100
    metrics = {"Correlation": corr, "MAE": mae, "MSE": mse, "Amplitude Error (%)": amplitude_error}
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    return metrics


# =============================================================================
# 3. 主程序执行
# =============================================================================
if __name__ == "__main__":
    print(f"--- 最终对比实验：FRF还原 vs EEMD提纯 ---")

    # --- 步骤 1: 加载并预处理真实数据 ---
    train_on_loaded = load_and_preprocess_single_channel(train_on_bed_file, 1, channel_to_analyze, samples_per_segment)
    train_under_loaded = load_and_preprocess_single_channel(train_under_bed_file, 1, channel_to_analyze,
                                                            samples_per_segment)
    test_on_loaded = load_and_preprocess_single_channel(test_on_bed_file, 1, channel_to_analyze, samples_per_segment)
    test_under_loaded = load_and_preprocess_single_channel(test_under_bed_file, 1, channel_to_analyze,
                                                           samples_per_segment)
    if any(x is None for x in [train_on_loaded, train_under_loaded, test_on_loaded, test_under_loaded]): print(
        "\n错误：加载失败，程序终止。"); exit()

    train_on_filt = bandpass_filter_segments([s for s in train_on_loaded], lowcut_freq, highcut_freq, fs, filter_order)
    train_under_filt = bandpass_filter_segments([s for s in train_under_loaded], lowcut_freq, highcut_freq, fs,
                                                filter_order)
    test_on_filt = bandpass_filter_segments([s for s in test_on_loaded], lowcut_freq, highcut_freq, fs, filter_order)
    test_under_filt = bandpass_filter_segments([s for s in test_under_loaded], lowcut_freq, highcut_freq, fs,
                                               filter_order)

    n_train = min(len(train_on_filt), len(train_under_filt));
    n_test = min(len(test_on_filt), len(test_under_filt))
    aligned_train_under_list, aligned_train_on_list = [], []
    for i in range(n_train):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(train_under_filt[i], train_on_filt[i], int(fs * 1))
        if len(aligned_u) > 0: aligned_train_under_list.append(aligned_u); aligned_train_on_list.append(aligned_o)
    min_len_train = min(len(s) for s in aligned_train_on_list) if aligned_train_on_list else 0
    train_under_final = [s[:min_len_train] for s in aligned_train_under_list];
    train_on_final = [s[:min_len_train] for s in aligned_train_on_list]
    aligned_test_under_list, aligned_test_on_list = [], []
    for i in range(n_test):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(test_under_filt[i], test_on_filt[i], int(fs * 1))
        if len(aligned_u) > 0: aligned_test_under_list.append(aligned_u); aligned_test_on_list.append(aligned_o)
    min_len_test = min(len(s) for s in aligned_test_on_list) if aligned_test_on_list else 0
    test_under_final = [s[:min_len_test] for s in aligned_test_under_list];
    test_on_final = [s[:min_len_test] for s in aligned_test_on_list]
    if not train_under_final or not test_on_final: print("错误：对齐后数据为空。"); exit()
    nfft_val = int(2 ** np.ceil(np.log2(max(min_len_train, 256))))

    # --- 步骤 2: 执行两种方法 ---
    # 方法一：FRF还原
    print("\n" + "=" * 20 + " 正在执行方法一：FRF还原... " + "=" * 20)
    h_estimated, _ = estimate_frf_H(train_under_final, train_on_final, fs, nfft_val)
    train_pred_raw_frf = reconstruct_signal_from_frf(train_under_final, h_estimated, nfft_val, min_len_train)
    coherent_gain_correction = 0.5
    train_pred_frf = [seg * coherent_gain_correction for seg in train_pred_raw_frf]
    # <<< 新增：对测试集进行FRF还原 >>>
    test_pred_raw_frf = reconstruct_signal_from_frf(test_under_final, h_estimated, nfft_val, min_len_test)
    test_pred_frf = [seg * coherent_gain_correction for seg in test_pred_raw_frf]
    print("FRF还原完成。")

    # 方法二：EEMD提纯
    print("\n" + "=" * 20 + " 正在执行方法二：EEMD提纯... " + "=" * 20)
    print("\n正在提纯训练集...")
    train_pred_eemd = apply_eemd_pipeline(train_on_final, FFT_DENOISE_THRESHOLD_RATIO)
    # <<< 新增：对测试集进行EEMD提纯 >>>
    print("\n正在提纯测试集...")
    test_pred_eemd = apply_eemd_pipeline(test_on_final, FFT_DENOISE_THRESHOLD_RATIO)

    # --- 步骤 3: 评估对比 ---
    print("\n" + "=" * 25 + " 最终评估对比 " + "=" * 25)
    print("\n" + "*" * 15 + " 训练集 " + "*" * 15)
    metrics_frf_train = calculate_metrics(train_on_final, train_pred_frf, "方法一：FRF还原 (训练集)")
    metrics_eemd_train = calculate_metrics(train_on_final, train_pred_eemd, "方法二：EEMD提纯 (训练集)")

    print("\n" + "*" * 15 + " 测试集 " + "*" * 15)
    metrics_frf_test = calculate_metrics(test_on_final, test_pred_frf, "方法一：FRF还原 (测试集)")
    metrics_eemd_test = calculate_metrics(test_on_final, test_pred_eemd, "方法二：EEMD提纯 (测试集)")

    # --- 步骤 4: 可视化对比 ---
    print("\n--- 分析完成，正在生成对比图表... ---")

    # 训练集可视化
    plot_idx_train = 10
    time_axis_train = np.arange(len(train_on_final[plot_idx_train])) / fs
    plt.figure(figsize=(18, 8));
    plt.plot(time_axis_train, train_on_final[plot_idx_train], label='原始 On-Bed 信号 (含噪)', color='black', alpha=0.6)
    plt.plot(time_axis_train, train_pred_frf[plot_idx_train],
             label=f'方法一：FRF还原 (Corr: {metrics_frf_train["Correlation"]:.3f})', color='deepskyblue',
             linestyle='--')
    plt.plot(time_axis_train, train_pred_eemd[plot_idx_train],
             label=f'方法二：EEMD提纯 (Corr: {metrics_eemd_train["Correlation"]:.3f})', color='green', linestyle='-.',
             linewidth=2)
    plt.title(f'最终方法对比 (训练集样本 #{plot_idx_train})', fontsize=16);
    plt.xlabel('时间 (s)'), plt.ylabel('幅度'), plt.legend(), plt.grid(True);
    plt.show()

    # 测试集可视化
    plot_idx_test = 10
    if n_test > plot_idx_test:
        time_axis_test = np.arange(len(test_on_final[plot_idx_test])) / fs
        plt.figure(figsize=(18, 8));
        plt.plot(time_axis_test, test_on_final[plot_idx_test], label='原始 On-Bed 信号 (含噪)', color='black',
                 alpha=0.6)
        plt.plot(time_axis_test, test_pred_frf[plot_idx_test],
                 label=f'方法一：FRF还原 (Corr: {metrics_frf_test["Correlation"]:.3f})', color='deepskyblue',
                 linestyle='--')
        plt.plot(time_axis_test, test_pred_eemd[plot_idx_test],
                 label=f'方法二：EEMD提纯 (Corr: {metrics_eemd_test["Correlation"]:.3f})', color='green', linestyle='-.',
                 linewidth=2)
        plt.title(f'最终方法对比 (测试集样本 #{plot_idx_test})', fontsize=16);
        plt.xlabel('时间 (s)'), plt.ylabel('幅度'), plt.legend(), plt.grid(True);
        plt.show()