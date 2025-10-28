import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import warnings
import random

# =============================================================================
# 1. Core Configuration
# =============================================================================
SAMPLING_RATE = 100
CHANNEL_TO_ANALYZE = 0
SAMPLES_PER_SEGMENT = 1000

# --- Filter Parameters ---
LOWCUT_FREQ = 4.5
HIGHCUT_FREQ = 9.5
FILTER_ORDER = 5

# --- 三次 Volterra 模型参数 (循环列表) ---
MODEL_NAME = "Volterra (3rd-Order Diag)"
VOLTERRA_MEMORY_DEPTH_TO_TRY = [10, 15, 20, 25, 30]  # 尝试的 M 阶数
ALPHA_VALUES_TO_TRY = [0.1, 1.0, 10.0, 100.0, 1000.0]  # 尝试的 Alpha 值

# --- Modified Attenuation Parameters ---
NEW_ATTENUATION_FACTOR = 0.25
CORNER_FREQ = 6.0
FIXED_POLY_BETA = 0.125
NOISE_LEVEL = 0.0  # 无噪声
RANDOM_SEED = 42

# --- Metric Configuration ---
CROP_LENGTH = 800
CLIP_THRESHOLD_STD_FACTOR = 20
MAIN_PEAK_WINDOW_SEC = 0.75

# --- Peak Finding Parameters ---
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3
PEAK_PROMINENCE_STD_FACTOR = 0.1
PEAK_MIN_DISTANCE_SEC = 0.1

# --- Data File Paths ---
TRAIN_ON_BED_RAW_FILE = r"./data/vibration_analysis.npy"
TRAIN_UNDER_BED_SAVE_FILE = "train_under_bed_nonlinear_attenuation.npy"
TEST_ON_BED_RAW_FILE = r"./data/BSG.npy"
TEST_UNDER_BED_SAVE_FILE = "test_under_bed_nonlinear_attenuation.npy"


# =============================================================================
# 2. Helper Functions
# =============================================================================

# --- [固定 Beta] 的非线性失真生成函数 (同前) ---
def generate_under_bed_signals_modified(on_bed_segments, fs, save_path):
    if not on_bed_segments: return []
    np.random.seed(RANDOM_SEED)
    # random.seed(RANDOM_SEED)

    under_bed_segments = []
    print("\nGenerating signals with MODIFIED non-linear attenuation (FIXED Beta)...")
    print(f" - Attenuation Factor: {NEW_ATTENUATION_FACTOR}")
    print(f" - Clipping: Removed")
    print(f" - Non-linearity: Cubic y = x - beta * x^3 / max(|x|)^2, FIXED beta = {FIXED_POLY_BETA}")
    print(f" - Noise Level: {NOISE_LEVEL}")

    for i, on_bed_seg in enumerate(on_bed_segments):
        n = len(on_bed_seg)
        signal_fft = fft(on_bed_seg)
        freqs = fftfreq(n, 1 / fs)
        attenuation_curve = np.exp(-(np.abs(freqs) / CORNER_FREQ) * NEW_ATTENUATION_FACTOR)
        attenuated_seg = np.real(ifft(signal_fft * attenuation_curve))
        max_abs_x = np.max(np.abs(attenuated_seg))
        if max_abs_x > 1e-9:
            beta = FIXED_POLY_BETA
            attenuated_seg = attenuated_seg - beta * np.power(attenuated_seg, 3) / (max_abs_x ** 2)
        output_seg = attenuated_seg
        if NOISE_LEVEL > 1e-9:
            signal_std = np.std(attenuated_seg)
            noise = NOISE_LEVEL * np.random.randn(len(attenuated_seg)) * (signal_std if signal_std > 1e-6 else 1e-6)
            output_seg = attenuated_seg + noise
        under_bed_segments.append(output_seg)
        print(f"\rGenerating under-bed signals: {i + 1}/{len(on_bed_segments)}", end="")

    under_bed_np = np.array(under_bed_segments)
    np.save(save_path, under_bed_np)
    print(f"\nSaved MODIFIED signals (Fixed Beta): {save_path}, shape: {under_bed_np.shape}")
    return under_bed_segments


# --- 加载, 对齐, 滤波 函数 (同前) ---
def load_and_segment_signal(file_path, segment_len, samples_to_remove=0):
    if not os.path.exists(file_path): raise FileNotFoundError(f"Signal file not found: {file_path}")
    raw_data = np.load(file_path);
    print(f"\nLoading signal file: {os.path.basename(file_path)}, original shape: {raw_data.shape}")
    if raw_data.ndim == 2 and raw_data.shape[1] > segment_len: print(
        f"Warning: Signal columns ({raw_data.shape[1]}) > target length ({segment_len}). Cropping."); raw_data = raw_data[
                                                                                                                 :,
                                                                                                                 :segment_len]
    if raw_data.ndim == 1:
        num_segments = len(raw_data) // segment_len; segmented_data = [raw_data[i * segment_len: (i + 1) * segment_len]
                                                                       for i in range(num_segments)]
    elif raw_data.ndim == 2:
        if raw_data.shape[1] != segment_len: raise ValueError(
            f"Loaded data has {raw_data.shape[1]} columns but expected {segment_len}.")
        segmented_data = [row for row in raw_data]
    else:
        raise ValueError(f"Unsupported signal dimension: {raw_data.ndim}")
    if samples_to_remove > 0:
        processed_segments = [seg[samples_to_remove:-samples_to_remove] for seg in segmented_data if
                              len(seg) > 2 * samples_to_remove]
    else:
        processed_segments = segmented_data
    print(f"Signal processing complete: {len(processed_segments)} segments.");
    return processed_segments


def align_segments_cross_correlation(signal_ref, signal_target, max_lag_samples=100):
    min_len = min(len(signal_ref), len(signal_target));
    ref, target = signal_ref[:min_len], signal_target[:min_len]
    try:
        corr = signal.correlate(target, ref, mode='full'); lags = signal.correlation_lags(min_len, min_len, mode='full')
    except ValueError:
        return ref, target, 0
    if len(lags) == 0: return ref, target, 0
    lag = lags[np.argmax(corr)]
    if abs(lag) > max_lag_samples: lag = np.sign(lag) * max_lag_samples
    if lag > 0:
        ref, target = ref[:-lag], target[lag:]
    elif lag < 0:
        ref, target = ref[abs(lag):], target[:-abs(lag)]
    final_len = min(len(ref), len(target));
    return ref[:final_len], target[:final_len], lag


def align_with_fixed_lag(signal_ref, signal_target, lag):
    if lag > 0:
        ref_aligned, target_aligned = signal_ref[:-lag], signal_target[lag:]
    elif lag < 0:
        ref_aligned, target_aligned = signal_ref[abs(lag):], signal_target[:-abs(lag)]
    else:
        ref_aligned, target_aligned = signal_ref, signal_target
    min_len = min(len(ref_aligned), len(target_aligned));
    return ref_aligned[:min_len], target_aligned[:min_len]


def butter_bandpass(low, high, fs, order=5):
    nyquist = 0.5 * fs;
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype='band');
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    if not segments: return []
    b, a = butter_bandpass(low, high, fs, order=order)
    filtered_segments = []
    for seg in segments:
        if len(seg) > 2 * order:
            try:
                filtered_segments.append(signal.filtfilt(b, a, seg))
            except ValueError:
                filtered_segments.append(np.zeros_like(seg))
        else:
            filtered_segments.append(np.array([]))
    return filtered_segments


# --- [新] 简化的三次 Volterra 模型函数 ---
def train_volterra_3rd_diag(input_segments, output_segments, memory_depth, alpha):
    """
    训练简化的三次Volterra模型：
    y(k) = sum(h[i] * u(k-i)) + sum(q[i] * u(k-i)^3)
    """
    print(f"\nTraining 3rd-Order Diagonal Volterra model with M={memory_depth}, Alpha={alpha}...")

    valid_inputs = [];
    valid_outputs = []
    for i in range(len(input_segments)):
        inp = input_segments[i];
        outp = output_segments[i]
        min_len = min(len(inp), len(outp))
        if min_len > memory_depth:
            valid_inputs.append(inp[:min_len]);
            valid_outputs.append(outp[:min_len])

    if not valid_inputs or not valid_outputs:
        print("Error: No valid training segments found.");
        return None

    M = memory_depth
    num_coeffs = 2 * M  # M 个线性系数, M 个三次系数

    # 构建回归矩阵 Phi 和目标向量 Y
    total_valid_rows = sum(len(s) - M + 1 for s in valid_inputs)
    Phi = np.zeros((total_valid_rows, num_coeffs))
    Y_target = np.zeros(total_valid_rows)

    current_row = 0
    for i in range(len(valid_inputs)):
        seg_u = valid_inputs[i]  # u 是输入
        seg_y = valid_outputs[i]  # y 是目标
        seg_len = len(seg_u)
        num_samples_in_seg = seg_len - M + 1

        for n in range(num_samples_in_seg):  # n 是段内的起始索引
            u_delayed = seg_u[n + M - 1: n - 1 if n > 0 else None: -1]

            # 填充线性项
            Phi[current_row, :M] = u_delayed

            # 填充三次项
            with np.errstate(over='ignore', invalid='ignore'):
                u_delayed_cubed = np.power(u_delayed, 3)
                if not np.all(np.isfinite(u_delayed_cubed)):
                    u_delayed_cubed = np.clip(u_delayed_cubed, -1e20, 1e20)
                    u_delayed_cubed[~np.isfinite(u_delayed_cubed)] = 0
            Phi[current_row, M:] = u_delayed_cubed

            Y_target[current_row] = seg_y[n + M - 1]
            current_row += 1

    print("Solving regularized linear system for 3rd-order Volterra kernels...")

    # L2 正则化
    reg_matrix = np.sqrt(alpha) * np.eye(num_coeffs)
    zeros_vector = np.zeros(num_coeffs)
    Phi_aug = np.vstack((Phi, reg_matrix))
    y_aug = np.concatenate((Y_target, zeros_vector))

    try:
        kernel, _, _, _ = np.linalg.lstsq(Phi_aug, y_aug, rcond=None)
        print(f"Volterra 3rd-order training complete. Learned {len(kernel)} coefficients.")
        return kernel
    except np.linalg.LinAlgError as e:
        print(f"Error solving linear system: {e}")
        return None


def apply_volterra_3rd_diag(input_segments, kernel, memory_depth):
    """
    应用简化的三次Volterra模型
    """
    # print("Applying trained 3rd-Order Diagonal Volterra model...") # 在循环中减少打印
    if kernel is None:
        print("Cannot apply model: kernel is None.");
        return [np.array([]) for _ in input_segments]

    M = memory_depth
    num_coeffs = 2 * M
    if len(kernel) != num_coeffs:
        raise ValueError(f"Kernel size mismatch. Expected {num_coeffs}, got {len(kernel)}.")

    h_coeffs = kernel[:M]  # 线性核
    q_coeffs = kernel[M:]  # 三次核

    reconstructed_segments = []
    for u in input_segments:
        if len(u) < M:
            reconstructed_segments.append(np.array([]));
            continue

        # 1. 计算线性部分
        y_linear = signal.lfilter(h_coeffs, [1.0], u)

        # 2. 计算非线性部分
        with np.errstate(over='ignore', invalid='ignore'):
            u_cubed = np.power(u, 3)
            if not np.all(np.isfinite(u_cubed)):
                u_cubed = np.clip(u_cubed, -1e20, 1e20)
                u_cubed[~np.isfinite(u_cubed)] = 0
        y_cubic = signal.lfilter(q_coeffs, [1.0], u_cubed)

        # 3. 叠加
        y_pred = y_linear + y_cubic

        reconstructed_segments.append(y_pred)

    return reconstructed_segments


# --- 度量和绘图函数 (同前) ---
def get_peak_indices(seg, fs=100):
    if seg is None or len(seg) < SAVGOL_WINDOW: return [], []
    if not np.all(np.isfinite(seg)): return [], []
    try:
        seg_smooth = signal.savgol_filter(seg, SAVGOL_WINDOW, SAVGOL_POLY, mode='mirror')
    except ValueError:
        seg_smooth = seg
    if not np.all(np.isfinite(seg_smooth)): return [], []
    min_dist_samples = int(fs * PEAK_MIN_DISTANCE_SEC);
    std_seg = np.std(seg_smooth)
    if not np.isfinite(std_seg) or std_seg < 1e-9:
        prominence = 1e-9
    else:
        prominence = std_seg * PEAK_PROMINENCE_STD_FACTOR
    if not np.isfinite(prominence) or prominence < 1e-9: prominence = 1e-9
    try:
        peaks, _ = signal.find_peaks(seg_smooth, prominence=prominence, distance=min_dist_samples)
        troughs, _ = signal.find_peaks(-seg_smooth, prominence=prominence, distance=min_dist_samples)
    except Exception as e:
        return [], []
    return peaks, troughs


def get_peak_to_peak_amplitudes(seg, fs=100):
    peaks, troughs = get_peak_indices(seg, fs)
    amplitudes = [];
    peak_indices = [];
    trough_indices = [];
    original_indices = []
    if len(peaks) == 0 or len(troughs) == 0: return amplitudes, peak_indices, trough_indices, original_indices
    extrema = [];
    for i in peaks: extrema.append((i, seg[i], 'peak'))
    for i in troughs: extrema.append((i, seg[i], 'trough'))
    extrema.sort(key=lambda x: x[0]);
    for i in range(1, len(extrema)):
        if extrema[i][2] != extrema[i - 1][2]:
            val1 = extrema[i][1];
            val2 = extrema[i - 1][1]
            if not np.isfinite(val1) or not np.isfinite(val2): continue
            amp_diff = np.abs(val1 - val2)
            if amp_diff > 1e-6:
                original_indices.append(len(amplitudes))
                amplitudes.append(amp_diff)
                if extrema[i][2] == 'peak':
                    peak_indices.append(extrema[i][0]);
                    trough_indices.append(extrema[i - 1][0])
                else:
                    peak_indices.append(extrema[i - 1][0]);
                    trough_indices.append(extrema[i][0])
    return amplitudes, peak_indices, trough_indices, original_indices


def calculate_metrics_core(true_flat_in, pred_flat_in, verbose_clipping=False):
    metrics = {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
               "Main P-T Amp Err (%)": np.nan}
    if true_flat_in is None or pred_flat_in is None or true_flat_in.size < SAVGOL_WINDOW or pred_flat_in.size < SAVGOL_WINDOW or not np.all(
        np.isfinite(true_flat_in)) or not np.all(np.isfinite(pred_flat_in)): return metrics
    min_len = min(len(true_flat_in), len(pred_flat_in));
    true_flat_in = true_flat_in[:min_len];
    pred_flat_in = pred_flat_in[:min_len]
    mean_true = np.mean(true_flat_in);
    std_true = np.std(true_flat_in)
    if not np.isfinite(std_true): std_true = 0
    clip_lower = mean_true - CLIP_THRESHOLD_STD_FACTOR * max(std_true, 1e-6);
    clip_upper = mean_true + CLIP_THRESHOLD_STD_FACTOR * max(std_true, 1e-6)
    true_clip_lower_indices = np.where(true_flat_in < clip_lower)[0];
    true_clip_upper_indices = np.where(true_flat_in > clip_upper)[0];
    pred_clip_lower_indices = np.where(pred_flat_in < clip_lower)[0];
    pred_clip_upper_indices = np.where(pred_flat_in > clip_upper)[0]
    num_true_clipped = len(true_clip_lower_indices) + len(true_clip_upper_indices);
    num_pred_clipped = len(pred_clip_lower_indices) + len(pred_clip_upper_indices)
    if verbose_clipping: print(
        f"    Clipping Report: Bounds=[{clip_lower:.2f}, {clip_upper:.2f}] (Mean={mean_true:.2f}, Std={std_true:.2f})");
    if verbose_clipping:
        if num_true_clipped == 0 and num_pred_clipped == 0:
            print("      No points clipped.")
        else:
            if num_true_clipped > 0: print(f"      True Signal: Clipped {num_true_clipped} points.")
            if num_pred_clipped > 0: print(f"      Pred Signal: Clipped {num_pred_clipped} points.")
    true_flat = np.clip(true_flat_in, clip_lower, clip_upper);
    pred_flat = np.clip(pred_flat_in, clip_lower, clip_upper)
    std_true_clipped = np.std(true_flat);
    std_pred_clipped = np.std(pred_flat);
    mae = np.mean(np.abs(true_flat - pred_flat));
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat));
    metrics["Mean Absolute Error (MAE)"] = mae;
    metrics["RMSE"] = rmse
    if std_true_clipped < 1e-9 and std_pred_clipped < 1e-9:
        corr = 1.0
    elif std_true_clipped < 1e-9 or std_pred_clipped < 1e-9:
        corr = 0.0
    else:
        try:
            corr, _ = pearsonr(true_flat, pred_flat);
        except ValueError:
            corr = np.nan
        if not np.isfinite(corr): corr = 0.0
    metrics["Correlation"] = corr
    result_pt = get_peak_to_peak_amplitudes(true_flat, fs=SAMPLING_RATE)
    if len(result_pt) != 4: return metrics
    true_amps, true_peak_indices, true_trough_indices, _ = result_pt;
    if not true_amps: return metrics
    main_true_amps = [];
    main_true_peak_indices = [];
    main_true_trough_indices = [];
    window_radius_samples = int(MAIN_PEAK_WINDOW_SEC * SAMPLING_RATE)
    if not true_amps: return metrics
    np_true_peak_indices = np.array(true_peak_indices);
    np_true_trough_indices = np.array(true_trough_indices);
    np_true_amps = np.array(true_amps)
    if len(np_true_peak_indices) == 0 or len(np_true_trough_indices) == 0: return metrics
    mid_indices = (np_true_peak_indices + np_true_trough_indices) / 2.0;
    added_indices_set = set()
    for i in range(len(true_amps)):
        current_amp = np_true_amps[i];
        current_mid_idx = mid_indices[i];
        window_start_idx = current_mid_idx - window_radius_samples;
        window_end_idx = current_mid_idx + window_radius_samples
        indices_in_window = np.where((mid_indices >= window_start_idx) & (mid_indices <= window_end_idx))[0];
        is_local_max = False
        if len(indices_in_window) > 0:
            max_amp_in_window = np.max(np_true_amps[indices_in_window])
            if np.isclose(current_amp, max_amp_in_window, atol=1e-9): is_local_max = True
        if is_local_max:
            if i not in added_indices_set: main_true_amps.append(current_amp); main_true_peak_indices.append(
                true_peak_indices[i]); main_true_trough_indices.append(true_trough_indices[i]); added_indices_set.add(i)
    if not main_true_amps: return metrics
    matched_main_true_amps = [];
    matched_main_pred_amps = []
    for i in range(len(main_true_amps)):
        pk_idx = main_true_peak_indices[i];
        tr_idx = main_true_trough_indices[i]
        if pk_idx >= len(pred_flat) or tr_idx >= len(pred_flat) or pk_idx < 0 or tr_idx < 0: continue
        true_main_amp_i = main_true_amps[i];
        pred_peak_val = pred_flat[pk_idx];
        pred_trough_val = pred_flat[tr_idx]
        if not np.isfinite(pred_peak_val) or not np.isfinite(pred_trough_val): continue
        pred_main_amp_i = np.abs(pred_peak_val - pred_trough_val);
        matched_main_true_amps.append(true_main_amp_i);
        matched_main_pred_amps.append(pred_main_amp_i)
    if matched_main_true_amps:
        true_arr = np.array(matched_main_true_amps);
        pred_arr = np.array(matched_main_pred_amps);
        denominator = np.maximum(true_arr, 1e-9)
        percent_errors = np.divide(np.abs(true_arr - pred_arr), denominator, out=np.zeros_like(denominator),
                                   where=denominator > 1e-9) * 100.0
        valid_errors = percent_errors[np.isfinite(percent_errors)]
        if valid_errors.size > 0:
            main_pt_amp_err = np.mean(valid_errors)
        else:
            main_pt_amp_err = np.nan
        metrics["Main P-T Amp Err (%)"] = main_pt_amp_err
    return metrics


def calculate_and_print_metrics(ground_truth_list, predicted_list, method_name, crop_len=None):
    title = f"\n--- {method_name} (Aggregate) Final Evaluation Results ---";
    if crop_len: title += f" (Cropped to middle {crop_len} samples)"
    print(title)
    valid_pairs = [(gt, pred) for gt, pred in zip(ground_truth_list, predicted_list) if
                   gt is not None and pred is not None and len(gt) > 0 and len(pred) > 0]
    if not valid_pairs: print("Input list contains no valid pairs."); return {}
    all_true_cropped = [];
    all_pred_cropped = []
    for true_seg, pred_seg in valid_pairs:
        true_aligned, pred_aligned = true_seg, pred_seg
        min_len = min(len(true_aligned), len(pred_aligned));
        true_aligned = true_aligned[:min_len];
        pred_aligned = pred_aligned[:min_len]
        if crop_len and min_len >= crop_len:
            start = (min_len - crop_len) // 2; end = start + crop_len; true_aligned = true_aligned[
                                                                                      start:end]; pred_aligned = pred_aligned[
                                                                                                                 start:end]
        elif crop_len and min_len < crop_len:
            continue
        if len(true_aligned) < SAVGOL_WINDOW: continue
        all_true_cropped.append(true_aligned);
        all_pred_cropped.append(pred_aligned)
    if not all_true_cropped: print("No valid segments remaining."); return {}
    true_flat = np.concatenate(all_true_cropped);
    pred_flat = np.concatenate(all_pred_cropped)
    metrics = calculate_metrics_core(true_flat, pred_flat, verbose_clipping=False)
    print(f"Correlation: {metrics['Correlation']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['Mean Absolute Error (MAE)']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"Main P-T Amp Err (%): {metrics['Main P-T Amp Err (%)']:.4f}")
    return metrics


def calculate_segment_metrics(ground_truth_seg, predicted_seg, crop_len=None):
    if ground_truth_seg is None or predicted_seg is None or len(ground_truth_seg) == 0 or len(
        predicted_seg) == 0: return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                                     "Main P-T Amp Err (%)": np.nan}
    true_flat, pred_flat = ground_truth_seg, predicted_seg
    min_len = min(len(true_flat), len(pred_flat));
    true_flat = true_flat[:min_len];
    pred_flat = pred_flat[:min_len]
    if crop_len and min_len >= crop_len:
        start = (min_len - crop_len) // 2; end = start + crop_len; true_flat = true_flat[
                                                                               start:end]; pred_flat = pred_flat[
                                                                                                       start:end]
    elif crop_len and min_len < crop_len:
        return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                "Main P-T Amp Err (%)": np.nan}
    if len(true_flat) < SAVGOL_WINDOW: return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan,
                                               "RMSE": np.nan, "Main P-T Amp Err (%)": np.nan}
    return calculate_metrics_core(true_flat, pred_flat, verbose_clipping=True)


# =============================================================================
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False

    # --- Step 1/2: Load Data and Generate (Fixed Beta, No Noise) ---
    print("\n" + "=" * 50 + " Step 1&2: Loading and Generating MODIFIED Data (FIXED Beta, NO NOISE) " + "=" * 50)
    train_on_bed_segments_raw = load_and_segment_signal(file_path=TRAIN_ON_BED_RAW_FILE,
                                                        segment_len=SAMPLES_PER_SEGMENT)
    test_on_bed_segments_raw = load_and_segment_signal(file_path=TEST_ON_BED_RAW_FILE, segment_len=SAMPLES_PER_SEGMENT)

    # --- 强制重新生成失真数据 ---
    print("\nGenerating Training Data (Modified, Fixed Beta, No Noise)...")
    train_under_bed_segments_gen = generate_under_bed_signals_modified(
        train_on_bed_segments_raw, SAMPLING_RATE, TRAIN_UNDER_BED_SAVE_FILE
    )
    print("\nGenerating Test Data (Modified, Fixed Beta, No Noise)...")
    test_under_bed_segments_gen = generate_under_bed_signals_modified(
        test_on_bed_segments_raw, SAMPLING_RATE, TEST_UNDER_BED_SAVE_FILE
    )

    # Truncate test set
    NUM_TEST_SAMPLES = 100
    if len(test_on_bed_segments_raw) > NUM_TEST_SAMPLES:
        print(f"\nTruncating test set to the first {NUM_TEST_SAMPLES} segments.")
        test_on_bed_segments_raw = test_on_bed_segments_raw[:NUM_TEST_SAMPLES]
        test_under_bed_segments_gen = test_under_bed_segments_gen[:NUM_TEST_SAMPLES]

    # Create "working" segments
    samples_to_remove = 1
    train_on_bed_working = [s[samples_to_remove:-samples_to_remove] for s in train_on_bed_segments_raw]
    train_under_bed_working = [s[samples_to_remove:-samples_to_remove] for s in train_under_bed_segments_gen]
    test_on_bed_working = [s[samples_to_remove:-samples_to_remove] for s in test_on_bed_segments_raw]
    test_under_bed_working = [s[samples_to_remove:-samples_to_remove] for s in test_under_bed_segments_gen]

    # --- Step 3: Aligning RAW (Unfiltered) Signals ---
    print("\n" + "=" * 70);
    print("Step 3: Aligning RAW (Unfiltered) Signals");
    print("=" * 70)
    print("\nAligning training data and learning average lag...")
    train_on_aligned, train_under_aligned, lags_from_training = [], [], []
    num_fail_align_train = 0
    for i, (o, u) in enumerate(zip(train_on_bed_working, train_under_bed_working)):
        if len(o) < 10 or len(u) < 10:
            o_a, u_a, lag = o, u, 0; num_fail_align_train += 1
        else:
            o_a, u_a, lag = align_segments_cross_correlation(o, u);
        if len(o_a) == 0: num_fail_align_train += 1
        train_on_aligned.append(o_a);
        train_under_aligned.append(u_a);
        lags_from_training.append(lag)
    valid_lags = [l for i, l in enumerate(lags_from_training) if len(train_on_aligned[i]) > 0]
    average_lag = int(np.round(np.mean(valid_lags))) if valid_lags else 0
    print(f"Learned average lag from {len(valid_lags)} valid segments: {average_lag} samples.")
    print("Aligning test data using the fixed average lag...")
    test_on_aligned, test_under_aligned = [], []
    for o, u in zip(test_on_bed_working, test_under_bed_working):
        o_a, u_a = align_with_fixed_lag(o, u, average_lag);
        test_on_aligned.append(o_a);
        test_under_aligned.append(u_a)

    # --- [修改] Step 4-7: 简化的三次 Volterra 模型参数循环 ---
    print("\n" + "=" * 70);
    print(f"Step 4-7: Hyperparameter Tuning for {MODEL_NAME} Model");
    print("=" * 70)

    # 预处理：带通滤波（只需要做一次）
    print("Applying bandpass filter to ALIGNED RAW input signals...")
    train_under_filtered = bandpass_filter_segments(train_under_aligned, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    test_under_filtered = bandpass_filter_segments(test_under_aligned, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)

    # 存储结果
    all_results = []
    best_result = {'M': 0, 'Alpha': 0, 'AmpErr': 999.0, 'Corr': 0.0}
    best_train_predicted = None
    best_test_predicted = None

    for M in VOLTERRA_MEMORY_DEPTH_TO_TRY:
        for Alpha in ALPHA_VALUES_TO_TRY:
            print("\n" + "=" * 30 + f" Testing M={M}, Alpha={Alpha} " + "=" * 30)

            # 步骤 1: 训练
            model_kernel = train_volterra_3rd_diag(
                train_under_filtered,
                train_on_aligned,
                M,  # <--- 使用循环中的 M
                Alpha  # <--- 使用循环中的 Alpha
            )

            if model_kernel is None:
                print(f"Training failed for M={M}, Alpha={Alpha}. Skipping.")
                all_results.append((M, Alpha, 999.0, 0.0))
                continue

            # 步骤 2: 应用
            train_predicted = apply_volterra_3rd_diag(
                train_under_filtered,
                model_kernel,
                M
            )
            test_predicted = apply_volterra_3rd_diag(
                test_under_filtered,
                model_kernel,
                M
            )

            # 步骤 3: 评估（无缩放）
            print("--- Evaluation (No Scaling) ---")
            train_metrics_agg = calculate_and_print_metrics(train_on_aligned, train_predicted,
                                                            f"Training Set (M={M}, A={Alpha})", crop_len=CROP_LENGTH)
            test_metrics_agg = calculate_and_print_metrics(test_on_aligned, test_predicted,
                                                           f"Test Set (M={M}, A={Alpha})", crop_len=CROP_LENGTH)

            test_amp_err = test_metrics_agg.get("Main P-T Amp Err (%)", 999.0)
            if not np.isfinite(test_amp_err): test_amp_err = 999.0
            test_corr = test_metrics_agg.get("Correlation", 0.0)
            if not np.isfinite(test_corr): test_corr = 0.0

            all_results.append((M, Alpha, test_amp_err, test_corr))

            # 步骤 4: 跟踪最佳模型
            if test_amp_err < best_result['AmpErr']:
                print(f"*** New Best Model Found! M={M}, Alpha={Alpha}, AmpErr={test_amp_err:.2f}% ***")
                best_result = {'M': M, 'Alpha': Alpha, 'AmpErr': test_amp_err, 'Corr': test_corr}
                best_train_predicted = train_predicted
                best_test_predicted = test_predicted

    # --- 循环结束，打印总结 ---
    print("\n" + "=" * 70);
    print("Hyperparameter Tuning Summary");
    print("=" * 70)
    print(f"{'M (Memory)':<12} | {'Alpha (Reg)':<12} | {'Test Amp Err (%)':<18} | {'Test Correlation':<18}")
    print("-" * 66)
    for M, Alpha, AmpErr, Corr in all_results:
        print(f"{M:<12} | {Alpha:<12} | {AmpErr:<18.2f} | {Corr:<18.4f}")

    print("\n" + "=" * 70);
    print("Best Model Found");
    print("=" * 70)
    print(f"Memory Depth (M): {best_result['M']}")
    print(f"Alpha:            {best_result['Alpha']}")
    print(f"Test Correlation: {best_result['Corr']:.4f}")
    print(f"Test Main P-T Amp Err (%): {best_result['AmpErr']:.2f}%")

    # --- Step 7.5 & 8: 仅对最佳模型执行详细指标和可视化 ---
    if best_test_predicted is None:
        print("\nNo valid model was trained. Skipping detailed metrics and visualization.")
    else:
        MODEL_NAME = f"Volterra-3 (Best M={best_result['M']}, A={best_result['Alpha']})"

        print(
            "\n" + "=" * 50 + f" Step 7.5: Per-Segment Metrics ({MODEL_NAME}, Fixed Beta, No Noise, Cropped)" + "=" * 50)
        all_train_metrics = [];
        all_test_metrics = []
        # 重新计算最佳模型的逐段指标 (用于绘图)
        for i in range(len(best_train_predicted)):
            gt_seg = train_on_aligned[i] if i < len(train_on_aligned) else None;
            pred_seg = best_train_predicted[i]
            metrics = calculate_segment_metrics(gt_seg, pred_seg, crop_len=CROP_LENGTH)
            all_train_metrics.append(metrics)
            corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
            mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
                metrics['Mean Absolute Error (MAE)']) else "nan";
            rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
            main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%" if np.isfinite(
                metrics['Main P-T Amp Err (%)']) else "nan%"
            # 简化打印
            # print(f"  Train Segment #{i:03d}: Corr={corr_str}, MAE={mae_str}, RMSE={rmse_str}, Main P-T Amp Err={main_pt_err_str}")
        print(f"Calculated {len(all_train_metrics)} training segment metrics.")
        for i in range(len(best_test_predicted)):
            gt_seg = test_on_aligned[i] if i < len(test_on_aligned) else None;
            pred_seg = best_test_predicted[i]
            metrics = calculate_segment_metrics(gt_seg, pred_seg, crop_len=CROP_LENGTH)
            all_test_metrics.append(metrics)
            corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
            mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
                metrics['Mean Absolute Error (MAE)']) else "nan";
            rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
            main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%" if np.isfinite(
                metrics['Main P-T Amp Err (%)']) else "nan%"
            print(
                f"  Test Segment #{i:03d}: Corr={corr_str}, MAE={mae_str}, RMSE={rmse_str}, Main P-T Amp Err={main_pt_err_str}")

        # --- Step 8: Visualization (Best Model) ---
        print("\n" + "=" * 70);
        print(f"Step 8: Visualizing Results for {MODEL_NAME} (Fixed Beta, No Noise)");
        print("=" * 70)
        train_plot_idx = 15
        if 0 <= train_plot_idx < len(train_on_aligned) and 0 <= train_plot_idx < len(
                best_train_predicted) and 0 <= train_plot_idx < len(train_under_aligned):
            plot_truth = train_on_aligned[train_plot_idx];
            plot_recon = best_train_predicted[train_plot_idx];
            plot_atten = train_under_aligned[train_plot_idx]
            plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon)) and not np.all(plot_recon == 0)
            plot_recon_display = plot_recon if plot_recon_valid else np.full_like(plot_truth, np.nan)
            if not plot_recon_valid: print(f"\nNote: Training Segment #{train_plot_idx} is invalid.")
            plt.figure(figsize=(20, 12));
            plt.suptitle(f"Training Set Reconstruction ({MODEL_NAME}) - Segment {train_plot_idx}", fontsize=20, y=0.98)
            plt.subplot(2, 1, 1);
            plt.plot(plot_truth, label='Original Raw On-Bed Signal (Aligned)', color='black', linewidth=2.5);
            plt.plot(plot_recon_display, label='Reconstructed Signal (NaN if invalid)', color='green', linewidth=2,
                     alpha=0.9);
            plt.plot(plot_atten, label='Distorted Under-Bed Signal (Aligned, Fixed Beta, No Noise)', color='blue',
                     linewidth=1.5, alpha=0.7, linestyle='--');
            plt.title('Full Time Series', fontsize=16);
            plt.ylabel('Amplitude', fontsize=14);
            plt.legend(fontsize=12);
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.subplot(2, 1, 2)
            if len(plot_truth) >= CROP_LENGTH:
                zoom_start = (len(plot_truth) - CROP_LENGTH) // 2;
                zoom_end = zoom_start + CROP_LENGTH;
                time_axis_zoom = np.arange(zoom_start, zoom_end)
                plot_truth_zoom = plot_truth[zoom_start:zoom_end];
                plot_recon_zoom = plot_recon_display[zoom_start:zoom_end];
                plot_atten_zoom = plot_atten[zoom_start:zoom_end]
                plt.plot(time_axis_zoom, plot_truth_zoom, label='Original Raw On-Bed Signal', color='black',
                         linewidth=2.5);
                plt.plot(time_axis_zoom, plot_recon_zoom, label='Reconstructed Signal', color='green', linewidth=2,
                         alpha=0.9);
                plt.plot(time_axis_zoom, plot_atten_zoom, label='Attenuated Under-Bed Signal (Fixed Beta, No Noise)',
                         color='blue', linewidth=1.5, alpha=0.7, linestyle='--')
                min_dist_samples_vis = int(SAMPLING_RATE * PEAK_MIN_DISTANCE_SEC)
                try:
                    truth_smooth_zoom = signal.savgol_filter(plot_truth_zoom, SAVGOL_WINDOW, SAVGOL_POLY, mode='mirror')
                except ValueError:
                    truth_smooth_zoom = plot_truth_zoom
                std_true_zoom = np.std(truth_smooth_zoom);
                prominence_vis = 1e-6
                if np.isfinite(
                    std_true_zoom) and std_true_zoom > 1e-9: prominence_vis = std_true_zoom * PEAK_PROMINENCE_STD_FACTOR
                if not np.isfinite(prominence_vis) or prominence_vis < 1e-9: prominence_vis = 1e-9
                try:
                    zoomed_true_peaks, _ = signal.find_peaks(truth_smooth_zoom, prominence=prominence_vis,
                                                             distance=min_dist_samples_vis)
                    zoomed_true_troughs, _ = signal.find_peaks(-truth_smooth_zoom, prominence=prominence_vis,
                                                               distance=min_dist_samples_vis)
                    plt.scatter(time_axis_zoom[zoomed_true_peaks], plot_truth_zoom[zoomed_true_peaks], color='black',
                                marker='o', s=50, label='True Peaks (Vis)', zorder=5)
                    plt.scatter(time_axis_zoom[zoomed_true_troughs], plot_truth_zoom[zoomed_true_troughs],
                                color='black', marker='x', s=50, label='True Troughs (Vis)', zorder=5)
                except Exception:
                    pass
                if plot_recon_valid:
                    try:
                        recon_smooth_zoom = signal.savgol_filter(plot_recon_zoom, SAVGOL_WINDOW, SAVGOL_POLY,
                                                                 mode='mirror')
                    except ValueError:
                        recon_smooth_zoom = plot_recon_zoom
                    if np.all(np.isfinite(recon_smooth_zoom)):
                        try:
                            zoomed_pred_peaks, _ = signal.find_peaks(recon_smooth_zoom, prominence=prominence_vis,
                                                                     distance=min_dist_samples_vis)
                            zoomed_pred_troughs, _ = signal.find_peaks(-recon_smooth_zoom, prominence=prominence_vis,
                                                                       distance=min_dist_samples_vis)
                            plt.scatter(time_axis_zoom[zoomed_pred_peaks], plot_recon_zoom[zoomed_pred_peaks],
                                        color='lime', marker='o', s=30, alpha=0.7, label='Pred. Peaks (Vis)', zorder=4)
                            plt.scatter(time_axis_zoom[zoomed_pred_troughs], plot_recon_zoom[zoomed_pred_troughs],
                                        color='lime', marker='x', s=30, alpha=0.7, label='Pred. Troughs (Vis)',
                                        zorder=4)
                        except Exception:
                            pass
                if 0 <= train_plot_idx < len(all_train_metrics):
                    metrics = all_train_metrics[train_plot_idx]
                    corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
                    mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
                        metrics['Mean Absolute Error (MAE)']) else "nan";
                    rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
                    main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%" if np.isfinite(
                        metrics['Main P-T Amp Err (%)']) else "nan%"
                    metrics_text = f"Segment Metrics (Cropped):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain P-T Amp Err: {main_pt_err_str}"
                    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=11,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                plt.title(f'Zoomed-in View (Middle {CROP_LENGTH} Samples)', fontsize=16);
                plt.xlabel('Sample Index');
                plt.ylabel('Amplitude', fontsize=14);
                plt.legend(fontsize=10, loc='upper right');
                plt.grid(True, linestyle='--', linewidth=0.5);
                plt.tight_layout(rect=[0, 0, 1, 0.95]);
                plt.show()
            else:
                print(f"Skipping zoom plot for train segment {train_plot_idx}.")
        else:
            print(f"Skipping train plot: Index {train_plot_idx} out of bounds.")

        test_plot_idx = 5
        if 0 <= test_plot_idx < len(test_on_aligned) and 0 <= test_plot_idx < len(
                best_test_predicted) and 0 <= test_plot_idx < len(test_under_aligned):
            plot_truth = test_on_aligned[test_plot_idx];
            plot_recon = best_test_predicted[test_plot_idx];
            plot_atten = test_under_aligned[test_plot_idx]
            plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon)) and not np.all(plot_recon == 0)
            plot_recon_display = plot_recon if plot_recon_valid else np.full_like(plot_truth, np.nan)
            if not plot_recon_valid: print(f"\nNote: Test Segment #{test_plot_idx} is invalid.")
            plt.figure(figsize=(20, 12));
            plt.suptitle(f"Test Set Reconstruction ({MODEL_NAME}) - Segment {test_plot_idx}", fontsize=20, y=0.98)
            plt.subplot(2, 1, 1);
            plt.plot(plot_truth, label='Original Raw On-Bed Signal (Aligned)', color='darkred', linewidth=2.5);
            plt.plot(plot_recon_display, label='Reconstructed Signal (NaN if invalid)', color='darkgreen', linewidth=2,
                     alpha=0.9);
            plt.plot(plot_atten, label='Distorted Under-Bed Signal (Aligned, Fixed Beta, No Noise)', color='darkblue',
                     linewidth=1.5, alpha=0.7, linestyle='--');
            plt.title('Full Time Series', fontsize=16);
            plt.ylabel('Amplitude', fontsize=14);
            plt.legend(fontsize=12);
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.subplot(2, 1, 2)
            if len(plot_truth) >= CROP_LENGTH:
                zoom_start = (len(plot_truth) - CROP_LENGTH) // 2;
                zoom_end = zoom_start + CROP_LENGTH;
                time_axis_zoom = np.arange(zoom_start, zoom_end)
                plot_truth_zoom = plot_truth[zoom_start:zoom_end];
                plot_recon_zoom = plot_recon_display[zoom_start:zoom_end];
                plot_atten_zoom = plot_atten[zoom_start:zoom_end]
                plt.plot(time_axis_zoom, plot_truth_zoom, label='Original Raw On-Bed Signal', color='darkred',
                         linewidth=2.5);
                plt.plot(time_axis_zoom, plot_recon_zoom, label='Reconstructed Signal', color='darkgreen', linewidth=2,
                         alpha=0.9);
                plt.plot(time_axis_zoom, plot_atten_zoom, label='Attenuated Under-Bed Signal (Fixed Beta, No Noise)',
                         color='darkblue', linewidth=1.5, alpha=0.7, linestyle='--')
                min_dist_samples_vis = int(SAMPLING_RATE * PEAK_MIN_DISTANCE_SEC)
                try:
                    truth_smooth_zoom = signal.savgol_filter(plot_truth_zoom, SAVGOL_WINDOW, SAVGOL_POLY, mode='mirror')
                except ValueError:
                    truth_smooth_zoom = plot_truth_zoom
                std_true_zoom = np.std(truth_smooth_zoom);
                prominence_vis = 1e-6
                if np.isfinite(
                    std_true_zoom) and std_true_zoom > 1e-9: prominence_vis = std_true_zoom * PEAK_PROMINENCE_STD_FACTOR
                if not np.isfinite(prominence_vis) or prominence_vis < 1e-9: prominence_vis = 1e-9
                try:
                    zoomed_true_peaks, _ = signal.find_peaks(truth_smooth_zoom, prominence=prominence_vis,
                                                             distance=min_dist_samples_vis)
                    zoomed_true_troughs, _ = signal.find_peaks(-truth_smooth_zoom, prominence=prominence_vis,
                                                               distance=min_dist_samples_vis)
                    plt.scatter(time_axis_zoom[zoomed_true_peaks], plot_truth_zoom[zoomed_true_peaks], color='black',
                                marker='o', s=50, label='True Peaks (Vis)', zorder=5)
                    plt.scatter(time_axis_zoom[zoomed_true_troughs], plot_truth_zoom[zoomed_true_troughs],
                                color='black', marker='x', s=50, label='True Troughs (Vis)', zorder=5)
                except Exception:
                    pass
                if plot_recon_valid:
                    try:
                        recon_smooth_zoom = signal.savgol_filter(plot_recon_zoom, SAVGOL_WINDOW, SAVGOL_POLY,
                                                                 mode='mirror')
                    except ValueError:
                        recon_smooth_zoom = plot_recon_zoom
                    if np.all(np.isfinite(recon_smooth_zoom)):
                        try:
                            zoomed_pred_peaks, _ = signal.find_peaks(recon_smooth_zoom, prominence=prominence_vis,
                                                                     distance=min_dist_samples_vis)
                            zoomed_pred_troughs, _ = signal.find_peaks(-recon_smooth_zoom, prominence=prominence_vis,
                                                                       distance=min_dist_samples_vis)
                            plt.scatter(time_axis_zoom[zoomed_pred_peaks], plot_recon_zoom[zoomed_pred_peaks],
                                        color='lime', marker='o', s=30, alpha=0.7, label='Pred. Peaks (Vis)', zorder=4)
                            plt.scatter(time_axis_zoom[zoomed_pred_troughs], plot_recon_zoom[zoomed_pred_troughs],
                                        color='lime', marker='x', s=30, alpha=0.7, label='Pred. Troughs (Vis)',
                                        zorder=4)
                        except Exception:
                            pass
                if 0 <= test_plot_idx < len(all_test_metrics):
                    metrics = all_test_metrics[test_plot_idx]
                    corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
                    mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
                        metrics['Mean Absolute Error (MAE)']) else "nan";
                    rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
                    main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%" if np.isfinite(
                        metrics['Main P-T Amp Err (%)']) else "nan%"
                    metrics_text = f"Segment Metrics (Cropped):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain P-T Amp Err: {main_pt_err_str}"
                    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=11,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                plt.title(f'Zoomed-in View (Middle {CROP_LENGTH} Samples)', fontsize=16);
                plt.xlabel('Sample Index');
                plt.ylabel('Amplitude', fontsize=14);
                plt.legend(fontsize=10, loc='upper right');
                plt.grid(True, linestyle='--', linewidth=0.5);
                plt.tight_layout(rect=[0, 0, 1, 0.95]);
                plt.show()
            else:
                print(f"Skipping zoom plot for test segment {test_plot_idx}.")
        else:
            print(f"Skipping test plot: Index {test_plot_idx} out of bounds.")

    print("\n" + "=" * 70);
    print("All Processes Completed");

    print("=" * 70)
