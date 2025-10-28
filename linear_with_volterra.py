import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error  # 导入 RMSE 计算

# =============================================================================
# 1. Core Configuration
# =============================================================================
SAMPLING_RATE = 100
CHANNEL_TO_ANALYZE = 0
SAMPLES_PER_SEGMENT = 1000

# --- Filter Parameters (用于 Volterra 训练) ---
LOWCUT_FREQ = 4.5
HIGHCUT_FREQ = 9.5
FILTER_ORDER = 5

# --- Volterra Model Parameters ---
VOLTERRA_MEMORY_DEPTH = 15

# --- Exponential Attenuation Parameters ---
CORNER_FREQ = 6.0
ATTENUATION_FACTOR = 0.5
RANDOM_SEED = 42

# --- Noise Parameters ---
FIXED_NOISE_STD = 0.05

# --- Metric Configuration  ---
CROP_LENGTH = 800
CLIP_THRESHOLD_STD_FACTOR = 20  # 裁剪阈值因子

# --- Peak Finding Parameters  ---
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3
PEAK_PROMINENCE_STD_FACTOR = 0.1
PEAK_MIN_DISTANCE_SEC = 0.1

# --- Data File Paths ---
TRAIN_ON_BED_RAW_FILE = r"./data/vibration_analysis.npy"
TRAIN_UNDER_BED_SAVE_FILE = "train_under_bed_linear_attenuation.npy"
TEST_ON_BED_RAW_FILE = r"./data/BSG.npy"
TEST_UNDER_BED_SAVE_FILE = "test_under_bed_linear_attenuation.npy"


# =============================================================================
# 2. Core Helper Functions
# =============================================================================

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


def generate_under_bed_signals(on_bed_segments, fs, save_path):
    np.random.seed(RANDOM_SEED);
    under_bed_segments = []
    print("\nGenerating signals with LINEAR attenuation...")
    for i, on_bed_seg in enumerate(on_bed_segments):
        n = len(on_bed_seg);
        signal_fft = fft(on_bed_seg);
        freqs = fftfreq(n, 1 / fs)
        attenuation_curve = np.exp(-(np.abs(freqs) / CORNER_FREQ) * ATTENUATION_FACTOR)
        signal_fft_attenuated = signal_fft * attenuation_curve;
        attenuated_seg = np.real(ifft(signal_fft_attenuated))
        noise = FIXED_NOISE_STD * np.random.randn(len(attenuated_seg));
        noisy_attenuated_seg = attenuated_seg + noise
        under_bed_segments.append(noisy_attenuated_seg);
        print(f"\rGenerating under-bed signals: {i + 1}/{len(on_bed_segments)} segments", end="")
    under_bed_np = np.array(under_bed_segments);
    np.save(save_path, under_bed_np)
    print(f"\nUnder-bed signals saved: {save_path}, shape: {under_bed_np.shape}");
    return under_bed_segments


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


# --- Volterra/Filter Specific Helpers ---
def butter_bandpass(low, high, fs, order=5):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype='band')
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    if not segments: return []
    b, a = butter_bandpass(low, high, fs, order=order)
    filtered_segments = []
    for seg in segments:
        if len(seg) > (order * 3):  # 确保信号长度足够进行 filtfilt
            filtered_segments.append(signal.filtfilt(b, a, seg))
        else:
            filtered_segments.append(np.array([]))  # 如果太短则返回空
    return filtered_segments


# --- Volterra Model Functions ---
def train_volterra_model(input_segments, output_segments, memory_depth):
    print(f"\nTraining 2nd order Volterra model with memory M={memory_depth}...")
    # 筛选出有效的（非空）对
    valid_inputs = []
    valid_outputs = []
    for i, o in zip(input_segments, output_segments):
        if len(i) > memory_depth and len(o) > memory_depth:
            # 确保对齐后长度仍然足够
            min_len = min(len(i), len(o))
            if min_len > memory_depth:
                valid_inputs.append(i[:min_len])
                valid_outputs.append(o[:min_len])

    if not valid_inputs or not valid_outputs:
        print("Error: No valid training segments after filtering and length check.")
        return None

    x = np.concatenate(valid_inputs)
    y = np.concatenate(valid_outputs)
    M = memory_depth
    if len(x) < M: raise ValueError("Signal length is smaller than model memory depth.")

    num_coeffs = M + M * M
    num_samples = len(x) - M + 1
    Phi = np.zeros((num_samples, num_coeffs))

    # Pre-allocate y_target
    y_target = y[M - 1:]

    # 确保 Phi 和 y_target 的样本数一致
    if len(Phi) > len(y_target):
        # print(f"Warning: Phi has {len(Phi)} samples, y_target has {len(y_target)}. Truncating Phi.")
        Phi = Phi[:len(y_target)]
    elif len(y_target) > len(Phi):
        # print(f"Warning: y_target has {len(y_target)} samples, Phi has {len(Phi)}. Truncating y_target.")
        y_target = y_target[:len(Phi)]

    for n in range(len(y_target)):  # Iterate up to the length of the target
        x_delayed = x[n + M - 1:n - 1:-1] if n > 0 else x[n + M - 1::-1]
        if len(x_delayed) < M:  # Safety check for edge cases, though loop range should prevent this
            # This should not happen if len(x) > M and num_samples is correct
            print(f"Warning: x_delayed length is {len(x_delayed)} at n={n}. Skipping.")
            continue
        Phi[n, :M] = x_delayed
        Phi[n, M:] = np.outer(x_delayed, x_delayed).flatten()

    print("Solving linear system for Volterra kernels...")
    try:
        kernel, _, _, _ = np.linalg.lstsq(Phi, y_target, rcond=None)
        print(f"Volterra model training complete. Learned {len(kernel)} coefficients.")
        return kernel
    except np.linalg.LinAlgError as e:
        print(f"Error: Linear algebra solver failed: {e}")
        return None


def apply_volterra_model(input_segments, kernel, memory_depth):
    print("Applying trained Volterra model...")
    if kernel is None:
        print("Cannot apply model: kernel is None.")
        return [np.array([]) for _ in input_segments]
    M = memory_depth
    num_coeffs = M + M * M
    if len(kernel) != num_coeffs:
        print(f"Error: Kernel size mismatch. Expected {num_coeffs}, got {len(kernel)}.")
        return [np.array([]) for _ in input_segments]

    reconstructed_segments = []
    for x in input_segments:
        if len(x) < M:
            reconstructed_segments.append(np.array([]))
            continue

        num_samples = len(x) - M + 1
        Phi_test = np.zeros((num_samples, num_coeffs))
        for n in range(num_samples):
            x_delayed = x[n + M - 1:n - 1:-1] if n > 0 else x[n + M - 1::-1]
            if len(x_delayed) < M:  # Safety check
                continue
            Phi_test[n, :M] = x_delayed
            Phi_test[n, M:] = np.outer(x_delayed, x_delayed).flatten()

        y_pred = Phi_test @ kernel
        # Pad at the beginning to align with the original signal length
        y_padded = np.pad(y_pred, (M - 1, 0), 'constant', constant_values=0)
        reconstructed_segments.append(y_padded)
    print("Model application complete.")
    return reconstructed_segments


# --- Robust Metric Functions  ---

def get_peak_indices(seg, fs=100):
    """ Finds peaks and troughs using smoothing, prominence, and distance. """
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
    """ Finds P-T amplitudes and corresponding indices based on the provided segment. """
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
    """
    Calculates robust metrics: Correlation, MAE, RMSE, and the
    "Main Cycle P-T Amp Err (%)" based on max P-T amp within cycles defined by true peaks.
    Includes input validation, value clipping, and optional clipping report.
    """
    metrics = {
        "Correlation": np.nan,
        "Mean Absolute Error (MAE)": np.nan,
        "RMSE": np.nan,
        "Main Cycle P-T Amp Err (%)": np.nan
    }

    if true_flat_in is None or pred_flat_in is None or \
            true_flat_in.size < SAVGOL_WINDOW or pred_flat_in.size < SAVGOL_WINDOW or \
            not np.all(np.isfinite(true_flat_in)) or not np.all(np.isfinite(pred_flat_in)):
        return metrics

    min_len = min(len(true_flat_in), len(pred_flat_in))
    true_flat_in = true_flat_in[:min_len];
    pred_flat_in = pred_flat_in[:min_len]

    mean_true = np.mean(true_flat_in);
    std_true = np.std(true_flat_in)
    if not np.isfinite(std_true): std_true = 0
    clip_lower = mean_true - CLIP_THRESHOLD_STD_FACTOR * max(std_true, 1e-6)
    clip_upper = mean_true + CLIP_THRESHOLD_STD_FACTOR * max(std_true, 1e-6)

    true_clip_lower_indices = np.where(true_flat_in < clip_lower)[0]
    true_clip_upper_indices = np.where(true_flat_in > clip_upper)[0]
    pred_clip_lower_indices = np.where(pred_flat_in < clip_lower)[0]
    pred_clip_upper_indices = np.where(pred_flat_in > clip_upper)[0]
    num_true_clipped = len(true_clip_lower_indices) + len(true_clip_upper_indices)
    num_pred_clipped = len(pred_clip_lower_indices) + len(pred_clip_upper_indices)

    if verbose_clipping:
        print(
            f"    Clipping Report: Bounds=[{clip_lower:.2f}, {clip_upper:.2f}] (Mean={mean_true:.2f}, Std={std_true:.2f})")
        if num_true_clipped == 0 and num_pred_clipped == 0:
            print("      No points clipped.")
        else:
            if num_true_clipped > 0: print(f"      True Signal: Clipped {num_true_clipped} points.")
            if num_pred_clipped > 0: print(f"      Pred Signal: Clipped {num_pred_clipped} points.")

    true_flat = np.clip(true_flat_in, clip_lower, clip_upper)
    pred_flat = np.clip(pred_flat_in, clip_lower, clip_upper)

    std_true_clipped = np.std(true_flat);
    std_pred_clipped = np.std(pred_flat)
    mae = np.mean(np.abs(true_flat - pred_flat));
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
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

    true_amps, true_peak_indices, true_trough_indices, _ = \
        get_peak_to_peak_amplitudes(true_flat, fs=SAMPLING_RATE)
    true_main_peaks, _ = get_peak_indices(true_flat, fs=SAMPLING_RATE)

    if not true_amps or len(true_main_peaks) < 2: return metrics

    main_cycle_true_amps = [];
    main_cycle_peak_indices = [];
    main_cycle_trough_indices = []
    selected_pt_amp_indices = set()

    for i in range(len(true_main_peaks) - 1):
        cycle_start_idx = true_main_peaks[i];
        cycle_end_idx = true_main_peaks[i + 1]
        best_amp_in_cycle = -1.0;
        best_amp_original_idx = -1
        indices_in_cycle_candidates = []
        for j in range(len(true_amps)):
            pk_idx = true_peak_indices[j];
            tr_idx = true_trough_indices[j]
            if cycle_start_idx <= pk_idx < cycle_end_idx and cycle_start_idx <= tr_idx < cycle_end_idx:
                indices_in_cycle_candidates.append((true_amps[j], j))
        if indices_in_cycle_candidates:
            best_amp_tuple = max(indices_in_cycle_candidates, key=lambda item: item[0])
            best_amp_in_cycle = best_amp_tuple[0];
            best_amp_original_idx = best_amp_tuple[1]
            if best_amp_original_idx not in selected_pt_amp_indices:
                main_cycle_true_amps.append(best_amp_in_cycle)
                main_cycle_peak_indices.append(true_peak_indices[best_amp_original_idx])
                main_cycle_trough_indices.append(true_trough_indices[best_amp_original_idx])
                selected_pt_amp_indices.add(best_amp_original_idx)

    if not main_cycle_true_amps: return metrics

    matched_main_cycle_true_amps = [];
    matched_main_cycle_pred_amps = []
    for i in range(len(main_cycle_true_amps)):
        pk_idx = main_cycle_peak_indices[i];
        tr_idx = main_cycle_trough_indices[i]
        if pk_idx >= len(pred_flat) or tr_idx >= len(pred_flat) or pk_idx < 0 or tr_idx < 0: continue
        true_main_amp_i = main_cycle_true_amps[i]
        pred_peak_val = pred_flat[pk_idx];
        pred_trough_val = pred_flat[tr_idx]
        if not np.isfinite(pred_peak_val) or not np.isfinite(pred_trough_val): continue
        pred_main_amp_i = np.abs(pred_peak_val - pred_trough_val)
        matched_main_cycle_true_amps.append(true_main_amp_i);
        matched_main_cycle_pred_amps.append(pred_main_amp_i)

    if matched_main_cycle_true_amps:
        true_arr = np.array(matched_main_cycle_true_amps);
        pred_arr = np.array(matched_main_cycle_pred_amps)
        denominator = np.maximum(true_arr, 1e-9)
        percent_errors = np.divide(np.abs(true_arr - pred_arr), denominator, out=np.zeros_like(denominator),
                                   where=denominator > 1e-9) * 100.0
        valid_errors = percent_errors[np.isfinite(percent_errors)]
        if valid_errors.size > 0:
            main_cycle_pt_amp_err = np.mean(valid_errors)
        else:
            main_cycle_pt_amp_err = np.nan
        metrics["Main Cycle P-T Amp Err (%)"] = main_cycle_pt_amp_err

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
    metrics = calculate_metrics_core(true_flat, pred_flat, verbose_clipping=False)  # 聚合时不打印裁剪
    print(f"Correlation: {metrics['Correlation']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['Mean Absolute Error (MAE)']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"Main Cycle P-T Amp Err (%): {metrics['Main Cycle P-T Amp Err (%)']:.4f}")  # 使用 Main Cycle 指标
    return metrics


def calculate_segment_metrics(ground_truth_seg, predicted_seg, crop_len=None):
    if ground_truth_seg is None or predicted_seg is None or len(ground_truth_seg) == 0 or len(predicted_seg) == 0:
        return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                "Main Cycle P-T Amp Err (%)": np.nan}
    true_flat, pred_flat = ground_truth_seg, predicted_seg
    min_len = min(len(true_flat), len(pred_flat));
    true_flat = true_flat[:min_len];
    pred_flat = pred_flat[:min_len]
    if crop_len and min_len >= crop_len:
        start = (min_len - crop_len) // 2;
        end = start + crop_len
        true_flat = true_flat[start:end];
        pred_flat = pred_flat[start:end]
    elif crop_len and min_len < crop_len:
        return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                "Main Cycle P-T Amp Err (%)": np.nan}
    if len(true_flat) < SAVGOL_WINDOW:
        return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                "Main Cycle P-T Amp Err (%)": np.nan}
    return calculate_metrics_core(true_flat, pred_flat, verbose_clipping=True)  # 分段时打印裁剪


# =============================================================================
# 3. Main Execution
# =============================================================================
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 70);
    print("Step 1/2: Loading and Generating Data");
    print("=" * 70)
    # --- Data Loading ---
    train_on_bed_segments_raw = load_and_segment_signal(file_path=TRAIN_ON_BED_RAW_FILE,
                                                        segment_len=SAMPLES_PER_SEGMENT)
    test_on_bed_segments_raw = load_and_segment_signal(file_path=TEST_ON_BED_RAW_FILE, segment_len=SAMPLES_PER_SEGMENT)
    if not os.path.exists(TRAIN_UNDER_BED_SAVE_FILE): generate_under_bed_signals(train_on_bed_segments_raw,
                                                                                 SAMPLING_RATE,
                                                                                 TRAIN_UNDER_BED_SAVE_FILE)
    if not os.path.exists(TEST_UNDER_BED_SAVE_FILE): generate_under_bed_signals(test_on_bed_segments_raw, SAMPLING_RATE,
                                                                                TEST_UNDER_BED_SAVE_FILE)
    train_under_bed_segments_raw = load_and_segment_signal(file_path=TRAIN_UNDER_BED_SAVE_FILE,
                                                           segment_len=SAMPLES_PER_SEGMENT)
    test_under_bed_segments_raw = load_and_segment_signal(file_path=TEST_UNDER_BED_SAVE_FILE,
                                                          segment_len=SAMPLES_PER_SEGMENT)

    # --- Remove Edge Samples ---
    samples_to_remove = 1
    train_on_working = [s[samples_to_remove:-samples_to_remove] for s in train_on_bed_segments_raw]
    train_under_working = [s[samples_to_remove:-samples_to_remove] for s in train_under_bed_segments_raw]
    test_on_working = [s[samples_to_remove:-samples_to_remove] for s in test_on_bed_segments_raw]
    test_under_working = [s[samples_to_remove:-samples_to_remove] for s in test_under_bed_segments_raw]

    # --- Step 3: Aligning RAW (Unfiltered) Signals ---
    print("\n" + "=" * 70);
    print("Step 3: Aligning RAW (Unfiltered) Signals");
    print("=" * 70)
    print("\nAligning training data and learning average lag...")
    train_on_aligned, train_under_aligned, lags_from_training = [], [], []
    num_fail_align_train = 0
    for i, (o, u) in enumerate(zip(train_on_working, train_under_working)):
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
    for o, u in zip(test_on_working, test_under_working): o_a, u_a = align_with_fixed_lag(o, u,
                                                                                          average_lag); test_on_aligned.append(
        o_a); test_under_aligned.append(u_a)

    # --- Step 4: Filter Data for Volterra Training ---
    print("\n" + "=" * 70);
    print("Step 4: Filtering Data for Volterra Training");
    print("=" * 70)
    # 过滤对齐后的信号用于训练
    train_on_aligned_filtered = bandpass_filter_segments(train_on_aligned, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    train_under_aligned_filtered = bandpass_filter_segments(train_under_aligned, LOWCUT_FREQ, HIGHCUT_FREQ,
                                                            SAMPLING_RATE)
    print("Filtering complete for training data.")

    # --- Step 5: Model Training & Prediction ---
    print("\n" + "=" * 70);
    print("Step 5: Volterra Model Training & Prediction");
    print("=" * 70)
    volterra_kernel = train_volterra_model(train_under_aligned_filtered, train_on_aligned_filtered,
                                           VOLTERRA_MEMORY_DEPTH)

    # 预测时也需要过滤输入信号
    print("\nFiltering test data for prediction...")
    test_under_aligned_filtered = bandpass_filter_segments(test_under_aligned, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)


    train_predicted = apply_volterra_model(train_under_aligned_filtered, volterra_kernel, VOLTERRA_MEMORY_DEPTH)
    test_predicted = apply_volterra_model(test_under_aligned_filtered, volterra_kernel, VOLTERRA_MEMORY_DEPTH)

    # --- Step 6: Final Evaluation (vs RAW Aligned Signals) ---
    print("\n" + "=" * 70);
    print("Step 6: Final Evaluation (Volterra vs. RAW)");
    print("=" * 70)

    # Ground truth 是 train_on_aligned 和 test_on_aligned (未经过滤波的原始对齐信号)
    # Prediction 是 train_predicted 和 test_predicted (来自 Volterra 模型的输出)

    print(f"\n--- Evaluating Training Set (Volterra) vs. RAW ---")
    train_metrics_agg = calculate_and_print_metrics(train_on_aligned, train_predicted,
                                                    f"Training Set (Volterra) vs. RAW",
                                                    crop_len=CROP_LENGTH)

    print(f"\n--- Evaluating Test Set (Volterra) vs. RAW ---")
    test_metrics_agg = calculate_and_print_metrics(test_on_aligned, test_predicted,
                                                   f"Test Set (Volterra) vs. RAW",
                                                   crop_len=CROP_LENGTH)

    print("\n" + "=" * 50 + f" Step 6.5: Per-Segment Metrics (Volterra, Cropped)" + "=" * 50)
    all_train_metrics = [];
    all_test_metrics = []
    # Train Metrics
    print("--- Training Set Per-Segment Metrics ---")
    for i in range(len(train_predicted)):
        gt_seg = train_on_aligned[i] if i < len(train_on_aligned) else None;
        pred_seg = train_predicted[i]
        metrics = calculate_segment_metrics(gt_seg, pred_seg,
                                            crop_len=CROP_LENGTH)  # verbose_clipping=True is called inside
        all_train_metrics.append(metrics)
        corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
        mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
            metrics['Mean Absolute Error (MAE)']) else "nan";
        rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
        main_cycle_pt_err_str = f"{metrics['Main Cycle P-T Amp Err (%)']:.2f}%" if np.isfinite(
            metrics['Main Cycle P-T Amp Err (%)']) else "nan%"
        print(
            f"  Train Segment #{i:03d}: Corr={corr_str}, MAE={mae_str}, RMSE={rmse_str}, Main Cycle P-T Amp Err={main_cycle_pt_err_str}")
    # Test Metrics
    print("\n--- Test Set Per-Segment Metrics ---")
    for i in range(len(test_predicted)):
        gt_seg = test_on_aligned[i] if i < len(test_on_aligned) else None;
        pred_seg = test_predicted[i]
        metrics = calculate_segment_metrics(gt_seg, pred_seg,
                                            crop_len=CROP_LENGTH)  # verbose_clipping=True is called inside
        all_test_metrics.append(metrics)
        corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
        mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
            metrics['Mean Absolute Error (MAE)']) else "nan";
        rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
        main_cycle_pt_err_str = f"{metrics['Main Cycle P-T Amp Err (%)']:.2f}%" if np.isfinite(
            metrics['Main Cycle P-T Amp Err (%)']) else "nan%"
        print(
            f"  Test Segment #{i:03d}: Corr={corr_str}, MAE={mae_str}, RMSE={rmse_str}, Main Cycle P-T Amp Err={main_cycle_pt_err_str}")

    # --- Visualization ---
    print("\n" + "=" * 70);
    print(f"Step 7: Visualizing Results for Volterra Model");
    print("=" * 70)

    # --- Train Plot ---
    train_plot_idx = 165
    if 0 <= train_plot_idx < len(train_on_aligned) and 0 <= train_plot_idx < len(
            train_predicted) and 0 <= train_plot_idx < len(train_under_aligned):
        plot_truth = train_on_aligned[train_plot_idx];
        plot_recon = train_predicted[train_plot_idx];
        plot_atten = train_under_aligned[train_plot_idx]
        plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon)) and not np.all(plot_recon == 0)
        plot_recon_display = plot_recon if plot_recon_valid else np.full_like(plot_truth, np.nan)
        if not plot_recon_valid: print(f"\nNote: Training Segment #{train_plot_idx} is invalid.")
        plt.figure(figsize=(20, 12));
        plt.suptitle(f"Training Set Reconstruction (Volterra Model) - Segment {train_plot_idx}", fontsize=20, y=0.98)
        plt.subplot(2, 1, 1);
        plt.plot(plot_truth, label='Original Raw On-Bed Signal (Aligned)', color='black', linewidth=2.5);
        plt.plot(plot_recon_display, label='Reconstructed Signal (Volterra)', color='green', linewidth=2, alpha=0.9);
        plt.plot(plot_atten, label='Original Raw Under-Bed Signal (Aligned)', color='blue', linewidth=1.5, alpha=0.7,
                 linestyle='--');
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
            plt.plot(time_axis_zoom, plot_truth_zoom, label='Original Raw On-Bed Signal', color='black', linewidth=2.5);
            plt.plot(time_axis_zoom, plot_recon_zoom, label='Reconstructed Signal', color='green', linewidth=2,
                     alpha=0.9);
            plt.plot(time_axis_zoom, plot_atten_zoom, label='Attenuated Under-Bed Signal', color='blue', linewidth=1.5,
                     alpha=0.7, linestyle='--')
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
                plt.scatter(time_axis_zoom[zoomed_true_troughs], plot_truth_zoom[zoomed_true_troughs], color='black',
                            marker='x', s=50, label='True Troughs (Vis)', zorder=5)
            except Exception:
                pass
            if plot_recon_valid:
                try:
                    recon_smooth_zoom = signal.savgol_filter(plot_recon_zoom, SAVGOL_WINDOW, SAVGOL_POLY, mode='mirror')
                except ValueError:
                    recon_smooth_zoom = plot_recon_zoom
                if np.all(np.isfinite(recon_smooth_zoom)):
                    try:
                        zoomed_pred_peaks, _ = signal.find_peaks(recon_smooth_zoom, prominence=prominence_vis,
                                                                 distance=min_dist_samples_vis)
                        zoomed_pred_troughs, _ = signal.find_peaks(-recon_smooth_zoom, prominence=prominence_vis,
                                                                   distance=min_dist_samples_vis)
                        plt.scatter(time_axis_zoom[zoomed_pred_peaks], plot_recon_zoom[zoomed_pred_peaks], color='lime',
                                    marker='o', s=30, alpha=0.7, label='Pred. Peaks (Vis)', zorder=4)
                        plt.scatter(time_axis_zoom[zoomed_pred_troughs], plot_recon_zoom[zoomed_pred_troughs],
                                    color='lime', marker='x', s=30, alpha=0.7, label='Pred. Troughs (Vis)', zorder=4)
                    except Exception:
                        pass
            if 0 <= train_plot_idx < len(all_train_metrics):
                metrics = all_train_metrics[train_plot_idx]
                corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
                mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
                    metrics['Mean Absolute Error (MAE)']) else "nan";
                rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
                main_cycle_pt_err_str = f"{metrics['Main Cycle P-T Amp Err (%)']:.2f}%" if np.isfinite(
                    metrics['Main Cycle P-T Amp Err (%)']) else "nan%"
                metrics_text = f"Segment Metrics (Cropped):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain Cycle P-T Amp Err: {main_cycle_pt_err_str}"
                plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
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

    # --- Test Plot ---
    test_plot_idx = 5
    if 0 <= test_plot_idx < len(test_on_aligned) and 0 <= test_plot_idx < len(
            test_predicted) and 0 <= test_plot_idx < len(test_under_aligned):
        plot_truth = test_on_aligned[test_plot_idx];
        plot_recon = test_predicted[test_plot_idx];
        plot_atten = test_under_aligned[test_plot_idx]
        plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon)) and not np.all(plot_recon == 0)
        plot_recon_display = plot_recon if plot_recon_valid else np.full_like(plot_truth, np.nan)
        if not plot_recon_valid: print(f"\nNote: Test Segment #{test_plot_idx} is invalid.")
        plt.figure(figsize=(20, 12));
        plt.suptitle(f"Test Set Reconstruction (Volterra Model) - Segment {test_plot_idx}", fontsize=20, y=0.98)
        plt.subplot(2, 1, 1);
        plt.plot(plot_truth, label='Original Raw On-Bed Signal (Aligned)', color='darkred', linewidth=2.5);
        plt.plot(plot_recon_display, label='Reconstructed Signal (Volterra)', color='darkgreen', linewidth=2,
                 alpha=0.9);
        plt.plot(plot_atten, label='Original Raw Under-Bed Signal (Aligned)', color='darkblue', linewidth=1.5,
                 alpha=0.7, linestyle='--');
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
            plt.plot(time_axis_zoom, plot_atten_zoom, label='Attenuated Under-Bed Signal', color='darkblue',
                     linewidth=1.5, alpha=0.7, linestyle='--')
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
                plt.scatter(time_axis_zoom[zoomed_true_troughs], plot_truth_zoom[zoomed_true_troughs], color='black',
                            marker='x', s=50, label='True Troughs (Vis)', zorder=5)
            except Exception:
                pass
            if plot_recon_valid:
                try:
                    recon_smooth_zoom = signal.savgol_filter(plot_recon_zoom, SAVGOL_WINDOW, SAVGOL_POLY, mode='mirror')
                except ValueError:
                    recon_smooth_zoom = plot_recon_zoom
                if np.all(np.isfinite(recon_smooth_zoom)):
                    try:
                        zoomed_pred_peaks, _ = signal.find_peaks(recon_smooth_zoom, prominence=prominence_vis,
                                                                 distance=min_dist_samples_vis)
                        zoomed_pred_troughs, _ = signal.find_peaks(-recon_smooth_zoom, prominence=prominence_vis,
                                                                   distance=min_dist_samples_vis)
                        plt.scatter(time_axis_zoom[zoomed_pred_peaks], plot_recon_zoom[zoomed_pred_peaks], color='lime',
                                    marker='o', s=30, alpha=0.7, label='Pred. Peaks (Vis)', zorder=4)
                        plt.scatter(time_axis_zoom[zoomed_pred_troughs], plot_recon_zoom[zoomed_pred_troughs],
                                    color='lime', marker='x', s=30, alpha=0.7, label='Pred. Troughs (Vis)', zorder=4)
                    except Exception:
                        pass
            if 0 <= test_plot_idx < len(all_test_metrics):
                metrics = all_test_metrics[test_plot_idx]
                corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
                mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
                    metrics['Mean Absolute Error (MAE)']) else "nan";
                rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
                main_cycle_pt_err_str = f"{metrics['Main Cycle P-T Amp Err (%)']:.2f}%" if np.isfinite(
                    metrics['Main Cycle P-T Amp Err (%)']) else "nan%"
                metrics_text = f"Segment Metrics (Cropped):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain Cycle P-T Amp Err: {main_cycle_pt_err_str}"
                plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
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