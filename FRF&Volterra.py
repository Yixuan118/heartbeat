import numpy as np
from scipy.fft import fft, ifft
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import itertools

# --- Matplotlib 中文字体设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("警告：无法设置中文字体。请确保系统安装至少一种指定字体。将使用默认字体。")

# =============================================================================
# 1. 配置
# =============================================================================
fs = 100
segment_length_seconds = 10
samples_per_segment = int(fs * segment_length_seconds)
channel_to_analyze = 0
VOLTERRA_MEMORY_M = 50

# 文件路径
try:
    train_on_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-10-04T161532_2025-10-04T162821.npy"
    train_under_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-10-04T161532_2025-10-04T162821.npy"
    test_on_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-09-15T233703_2025-09-15T234233.npy"
    test_under_bed_file = r"D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-09-15T233703_2025-09-15T234233.npy"
except Exception:
    print("警告：默认文件路径无效，请手动修改为您的.npy文件路径。")
    # ... (路径保持不变)

# 带通滤波器参数
lowcut_freq = 5
highcut_freq = 9.5
filter_order = 5

CHOSEN_FRF_ESTIMATOR = 'Hv_geometric_mean'


# =============================================================================
# 2. 辅助函数 (与您提供的版本一致)
# =============================================================================
def load_and_preprocess_single_channel(file_path, samples_to_remove_from_ends, target_channel,
                                       expected_segment_len_per_channel):
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        return None
    if data.ndim != 2: return None
    if data.shape[1] % expected_segment_len_per_channel != 0: return None
    num_channels_in_file = data.shape[1] // expected_segment_len_per_channel
    if num_channels_in_file <= target_channel: return None
    selected_channel_segments = data[:, target_channel * expected_segment_len_per_channel:(
                                                                                                  target_channel + 1) * expected_segment_len_per_channel]
    print(f"文件 {file_path.split('\\')[-1]}：加载了 {len(selected_channel_segments)} 个10s段。")
    if selected_channel_segments.shape[1] < 2 * samples_to_remove_from_ends:
        return selected_channel_segments
    else:
        return selected_channel_segments[:,
               samples_to_remove_from_ends:selected_channel_segments.shape[1] - samples_to_remove_from_ends]


def align_segments_cross_correlation(signal_ref, signal_target, max_lag_samples):
    min_len_initial = min(len(signal_ref), len(signal_target))
    signal_ref_trunc, signal_target_trunc = signal_ref[:min_len_initial], signal_target[:min_len_initial]
    n = len(signal_ref_trunc)
    if n == 0: return np.array([]), np.array([]), 0
    correlation = signal.correlate(signal_target_trunc, signal_ref_trunc, mode='full')
    lags = signal.correlation_lags(n, n, mode='full')
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
        "Segments must be non-empty and of equal number.")
    actual_segment_len = input_segments[0].shape[0]
    window = signal.windows.hann(actual_segment_len)
    S_xx_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yy_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yx_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    for i in range(len(input_segments)):
        x, y = input_segments[i] * window, output_segments[i] * window
        X_f, Y_f = fft(x, n=nfft), fft(y, n=nfft)
        S_xx_sum += np.conj(X_f[:nfft // 2 + 1]) * X_f[:nfft // 2 + 1]
        S_yy_sum += np.conj(Y_f[:nfft // 2 + 1]) * Y_f[:nfft // 2 + 1]
        S_yx_sum += Y_f[:nfft // 2 + 1] * np.conj(X_f[:nfft // 2 + 1])
    n_avg = len(input_segments)
    S_xx_avg, S_yy_avg, S_yx_avg = S_xx_sum / n_avg, S_yy_sum / n_avg, S_yx_sum / n_avg
    epsilon = 1e-12 * np.max(np.abs(S_xx_avg))
    epsilon_yx = 1e-12 * np.max(np.abs(S_yx_avg))
    H1_freq = S_yx_avg / (S_xx_avg + epsilon)
    H2_freq = S_yy_avg / (np.conj(S_yx_avg) + epsilon_yx)
    Hv_freq = np.sqrt(H1_freq * H2_freq)
    Hv_freq[np.isnan(Hv_freq) | np.isinf(Hv_freq)] = 0
    coherence = np.abs(S_yx_avg) ** 2 / (S_xx_avg * S_yy_avg + epsilon)
    freqs = np.fft.fftfreq(nfft, d=1 / fs)[:nfft // 2 + 1]
    return H1_freq, H2_freq, Hv_freq, coherence, freqs


def reconstruct_signal_from_frf(input_segments, H_freq_estimator, nfft, seg_len):
    predicted = []
    H_full = np.concatenate((H_freq_estimator, np.conj(H_freq_estimator[-2:0:-1])))
    for x in input_segments:
        y_pred = ifft(H_full * fft(x, n=nfft))
        predicted.append(np.real(y_pred[:seg_len]))
    return predicted


def butter_bandpass(low, high, fs, order=5):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    if segments is None: return None
    b, a = butter_bandpass(low, high, fs, order=order)
    return [signal.filtfilt(b, a, seg) for seg in segments]


def build_volterra_regressor_matrix(x, M):
    N, num_linear_terms = len(x), M
    num_quadratic_terms = M * (M + 1) // 2
    total_terms = num_linear_terms + num_quadratic_terms
    A = np.zeros((N - M, total_terms))
    quadratic_indices = list(itertools.combinations_with_replacement(range(M), 2))
    for t in range(M, N):
        history_reversed = x[t - M:t][::-1]
        A[t - M, :num_linear_terms] = history_reversed
        A[t - M, num_linear_terms:] = [history_reversed[i] * history_reversed[j] for i, j in quadratic_indices]
    return A


def train_volterra_model(input_segments, output_segments, M):
    print(f"开始训练Volterra模型 (记忆长度 M={M})...")
    A_all, b_all = [], []
    for x, y in zip(input_segments, output_segments):
        if len(x) <= M: continue
        A_segment, b_segment = build_volterra_regressor_matrix(x, M), y[M:]
        A_all.append(A_segment);
        b_all.append(b_segment)
    if not A_all:
        print("错误: 训练数据过短，无法构建Volterra模型。");
        return None
    A_total, b_total = np.vstack(A_all), np.concatenate(b_all)
    print(f"求解 {A_total.shape[0]}x{A_total.shape[1]} 的最小二乘问题...")
    kernels, _, _, _ = np.linalg.lstsq(A_total, b_total, rcond=None)
    print("Volterra模型训练完成。");
    return kernels


def reconstruct_with_volterra(input_segments, kernels, M):
    reconstructed_segments = []
    for x in input_segments:
        if len(x) <= M:
            reconstructed_segments.append(np.zeros_like(x));
            continue
        y_pred = build_volterra_regressor_matrix(x, M) @ kernels
        reconstructed_segments.append(np.concatenate([np.zeros(M), y_pred]))
    return reconstructed_segments


def calculate_metrics(true, pred):
    true_flat, pred_flat = np.concatenate(true), np.concatenate(pred)
    if np.std(true_flat) == 0 or np.std(pred_flat) == 0:
        corr = np.nan
    else:
        corr, _ = pearsonr(true_flat, pred_flat)
    mae = np.mean(np.abs(true_flat - pred_flat))
    mse = np.mean((true_flat - pred_flat) ** 2)
    mean_abs_true = np.mean(np.abs(true_flat))
    amplitude_error = mae / mean_abs_true if mean_abs_true > 1e-9 else np.inf
    return {"Correlation": corr, "MAE": mae, "MSE": mse, "Amplitude Error (%)": amplitude_error * 100}


# =============================================================================
# 3. 主程序执行
# =============================================================================
if __name__ == "__main__":
    print(f"--- 分析开始 (使用 {segment_length_seconds}s 段长) ---")

    # --- 数据加载 ---
    train_on_loaded = load_and_preprocess_single_channel(train_on_bed_file, 1, channel_to_analyze, samples_per_segment)
    train_under_loaded = load_and_preprocess_single_channel(train_under_bed_file, 1, channel_to_analyze,
                                                            samples_per_segment)
    test_on_loaded = load_and_preprocess_single_channel(test_on_bed_file, 1, channel_to_analyze, samples_per_segment)
    test_under_loaded = load_and_preprocess_single_channel(test_under_bed_file, 1, channel_to_analyze,
                                                           samples_per_segment)

    if any(x is None for x in [train_on_loaded, train_under_loaded, test_on_loaded, test_under_loaded]):
        print("\n错误：一个或多个数据文件加载失败，程序终止。")
        exit()

    # --- 1. 滤波 ---
    print("\n步骤 1: 对所有原始信号进行带通滤波...")
    train_on_filt = bandpass_filter_segments([seg for seg in train_on_loaded], lowcut_freq, highcut_freq, fs,
                                             filter_order)
    train_under_filt = bandpass_filter_segments([seg for seg in train_under_loaded], lowcut_freq, highcut_freq, fs,
                                                filter_order)
    test_on_filt = bandpass_filter_segments([seg for seg in test_on_loaded], lowcut_freq, highcut_freq, fs,
                                            filter_order)
    test_under_filt = bandpass_filter_segments([seg for seg in test_under_loaded], lowcut_freq, highcut_freq, fs,
                                               filter_order)

    # --- 2. 对齐 ---
    print("\n步骤 2: 对滤波后的干净信号进行互相关对齐...")
    n_train = min(len(train_on_filt), len(train_under_filt))
    n_test = min(len(test_on_filt), len(test_under_filt))

    aligned_train_under_list, aligned_train_on_list = [], []
    for i in range(n_train):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(train_under_filt[i], train_on_filt[i], int(fs * 1))
        if len(aligned_u) > 0:
            aligned_train_under_list.append(aligned_u)
            aligned_train_on_list.append(aligned_o)
    min_len_train = min(len(s) for s in aligned_train_under_list) if aligned_train_under_list else 0
    train_under_final = [s[:min_len_train] for s in aligned_train_under_list]
    train_on_final = [s[:min_len_train] for s in aligned_train_on_list]

    aligned_test_under_list, aligned_test_on_list = [], []
    for i in range(n_test):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(test_under_filt[i], test_on_filt[i], int(fs * 1))
        if len(aligned_u) > 0:
            aligned_test_under_list.append(aligned_u)
            aligned_test_on_list.append(aligned_o)
    min_len_test = min(len(s) for s in aligned_test_under_list) if aligned_test_under_list else 0
    test_under_final = [s[:min_len_test] for s in aligned_test_under_list]
    test_on_final = [s[:min_len_test] for s in aligned_test_on_list]

    if not train_under_final or not test_on_final:
        print("错误：对齐后数据为空，程序退出。");
        exit()

    print(f"\n匹配后：使用训练段 {len(train_on_final)}，测试段 {len(test_on_final)}")
    nfft_val = int(2 ** np.ceil(np.log2(max(min_len_train, 256))))

    # --- 3. 建模与评估 ---
    print("\n步骤 3: 使用对齐后的信号进行建模和评估...")
    # (线性模型)
    print("\n" + "=" * 20 + " 1. 线性FRF模型评估 " + "=" * 20)
    print("\n正在估计传递函数...")
    H1, H2, Hv, coh_train, frq = estimate_frf_H(train_under_final, train_on_final, fs, nfft_val)
    H_bed = {'H1': H1, 'H2': H2, 'Hv_geometric_mean': Hv}[CHOSEN_FRF_ESTIMATOR]
    train_pred_frf = reconstruct_signal_from_frf(train_under_final, H_bed, nfft_val, min_len_train)
    print("\n--- 训练集指标 (线性FRF模型) ---")
    metrics_train_frf = calculate_metrics(train_on_final, train_pred_frf)
    for k, v in metrics_train_frf.items(): print(f"{k}: {v:.4f}")
    test_pred_frf = reconstruct_signal_from_frf(test_under_final, H_bed, nfft_val, min_len_test)
    print("\n--- 测试集指标 (线性FRF模型) ---")
    metrics_test_frf = calculate_metrics(test_on_final, test_pred_frf)
    for k, v in metrics_test_frf.items(): print(f"{k}: {v:.4f}")

    # (非线性模型)
    print("\n" + "=" * 20 + " 2. 非线性Volterra模型评估 " + "=" * 20)
    volterra_kernels = train_volterra_model(train_under_final, train_on_final, VOLTERRA_MEMORY_M)
    train_pred_volterra = []
    test_pred_volterra = []
    if volterra_kernels is not None:
        train_pred_volterra = reconstruct_with_volterra(train_under_final, volterra_kernels, VOLTERRA_MEMORY_M)
        print("\n--- 训练集指标 (非线性Volterra模型) ---")
        metrics_train_volterra = calculate_metrics(train_on_final, train_pred_volterra)
        for k, v in metrics_train_volterra.items(): print(f"{k}: {v:.4f}")
        test_pred_volterra = reconstruct_with_volterra(test_under_final, volterra_kernels, VOLTERRA_MEMORY_M)
        print("\n--- 测试集指标 (非线性Volterra模型) ---")
        metrics_test_volterra = calculate_metrics(test_on_final, test_pred_volterra)
        for k, v in metrics_test_volterra.items(): print(f"{k}: {v:.4f}")

    # --- 4. 评估与可视化 ---
    print("\n--- 分析完成，正在生成图表... ---")

    # --- 4.1 训练集效果可视化 ---
    plot_idx_train =0# 选择第一个训练段进行可视化
    if len(train_on_final) > plot_idx_train:
        time_axis_train = np.arange(len(train_on_final[plot_idx_train])) / fs

        plt.figure(figsize=(15, 7))
        plt.plot(time_axis_train, train_on_final[plot_idx_train], label='真实 On-Bed 信号', color='black', linewidth=2)

        plt.plot(time_axis_train, train_pred_frf[plot_idx_train],
                 label=f'线性FRF重建 (Corr: {metrics_train_frf["Correlation"]:.3f})',
                 color='dodgerblue', linestyle='--')

        if volterra_kernels is not None and len(train_pred_volterra) > plot_idx_train:
            plt.plot(time_axis_train, train_pred_volterra[plot_idx_train],
                     label=f'Volterra重建 (Corr: {metrics_train_volterra["Correlation"]:.3f})',
                     color='orangered', linestyle=':', alpha=0.8)

        plt.title(f'模型重建效果对比 (训练集样本 #{plot_idx_train})', fontsize=16)
        plt.xlabel('时间 (s)'), plt.ylabel('信号幅值')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()

    # --- 4.2 测试集效果可视化 ---
    plot_idx_test = 0#选择第一个测试段进行可视化
    if len(test_on_final) > plot_idx_test:
        time_axis_test = np.arange(len(test_on_final[plot_idx_test])) / fs

        plt.figure(figsize=(15, 7))
        plt.plot(time_axis_test, test_on_final[plot_idx_test], label='真实 On-Bed 信号', color='black', linewidth=2)

        plt.plot(time_axis_test, test_pred_frf[plot_idx_test],
                 label=f'线性FRF重建 (Corr: {metrics_test_frf["Correlation"]:.3f})',
                 color='dodgerblue', linestyle='--')

        if volterra_kernels is not None and len(test_pred_volterra) > plot_idx_test:
            plt.plot(time_axis_test, test_pred_volterra[plot_idx_test],
                     label=f'Volterra重建 (Corr: {metrics_test_volterra["Correlation"]:.3f})',
                     color='orangered', linestyle=':', alpha=0.8)

        plt.title(f'模型重建效果对比 (测试集样本 #{plot_idx_test})', fontsize=16)
        plt.xlabel('时间 (s)'), plt.ylabel('信号幅值')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()