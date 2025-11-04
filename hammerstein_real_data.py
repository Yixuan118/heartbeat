import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings
import time
import random

# =============================================================================
# 1. Configuration
# =============================================================================
SAMPLING_RATE = 100
SEGMENT_LENGTH_SEC = 10
SAMPLES_PER_SEGMENT = int(SAMPLING_RATE * SEGMENT_LENGTH_SEC)
CHANNEL_TO_ANALYZE = 0

# --- File Paths (Fixed Train/Test Split) ---
TRAIN_ON_BED_FILE = r"./data/raw_signal_before_2025-10-04T234031_2025-10-04T235841.npy"
TRAIN_UNDER_BED_FILE = r"./data/raw_signal_after_2025-10-04T234031_2025-10-04T235841.npy"
TEST_ON_BED_FILE = r"./data/raw_signal_before_2025-09-16T092000_2025-09-16T092410.npy"
TEST_UNDER_BED_FILE = r"./data/raw_signal_after_2025-09-16T092000_2025-09-16T092410.npy"

# --- Bandpass Filter Parameters ---
LOWCUT_FREQ = 0.8
HIGHCUT_FREQ = 20
FILTER_ORDER = 5

# ---  N-L (Volterra + FRF) 模型参数 ---
MODEL_NAME = "Hammerstein (Volterra N-Block + FRF L-Block)"
CHOSEN_FRF_ESTIMATOR = 'Hv_geometric_mean'
VOLTERRA_MEMORY_DEPTH_TO_TRY = [3, 5, 8, 10]
ORDERS_TO_TRY = [
    [1],  # 仅线性 (v)
    [1, 3],  # 奇次 (v, v^3)
    [1, 2],  # 含偶次 (v, v^2)
    [1, 2, 3]  # 混合 (v, v^2, v^3)
]
N_ALPHA_TO_TRY = [0.01, 0.1, 1.0, 10.0]

# --- Metric & Peak Finding Configuration ---
CROP_LENGTH = 800
CLIP_THRESHOLD_STD_FACTOR = 20
MAIN_PEAK_WINDOW_SEC = 0.75
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3
PEAK_PROMINENCE_STD_FACTOR = 0.1
PEAK_MIN_DISTANCE_SEC = 0.1
PEAK_SEARCH_RADIUS = 10
RANDOM_SEED = 42


# =============================================================================
# 2. Helper Functions (Load, Align, Filter)
# =============================================================================
def load_and_preprocess_single_channel(file_path, samples_to_remove, target_channel, expected_segment_len):
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}");
        return None
    except ValueError as e:
        print(f"Error loading {file_path}. Is it a valid .npy file? Error: {e}");
        return None
    print(f"Loaded {os.path.basename(file_path)}, shape: {data.shape}.")
    if data.ndim == 1:
        num_segments = len(data) // expected_segment_len
        if num_segments == 0: print("Warning: Data is 1D and shorter than expected_segment_len."); return None
        data = np.array([data[i * expected_segment_len: (i + 1) * expected_segment_len] for i in range(num_segments)])
        print(f"Reshaped 1D data to (segment, samples): {data.shape}")
    if data.ndim != 2: print(f"Warning: Data in {file_path} has an unexpected shape {data.shape}."); return None
    if data.shape[1] == expected_segment_len:
        selected_channel_segments = data
    elif data.shape[1] % expected_segment_len == 0:
        num_channels_in_file = data.shape[1] // expected_segment_len
        if num_channels_in_file <= target_channel: print(
            f"Warning: Target channel {target_channel} not available in {file_path}."); return None
        start_col = target_channel * expected_segment_len;
        end_col = (target_channel + 1) * expected_segment_len
        selected_channel_segments = data[:, start_col:end_col];
        print(f"Extracted channel {target_channel} from multi-channel file.")
    else:
        print(
            f"Warning: Data columns ({data.shape[1]}) not divisible by segment length ({expected_segment_len}).");
        return None
    print(f"Loaded {len(selected_channel_segments)} segments from {os.path.basename(file_path)}.")
    return [seg for seg in selected_channel_segments]  # 返回列表


def align_segments_cross_correlation(signal_ref, signal_target, max_lag_samples):
    ref_np = np.asarray(signal_ref);
    target_np = np.asarray(signal_target)
    if ref_np.ndim != 1 or target_np.ndim != 1 or len(ref_np) == 0 or len(target_np) == 0: return np.array(
        []), np.array([]), 0
    min_len = min(len(ref_np), len(target_np));
    ref, target = ref_np[:min_len], target_np[:min_len]
    if min_len < 10: return np.array([]), np.array([]), 0
    try:
        if np.std(target) < 1e-9 or np.std(ref) < 1e-9: return ref, target, 0
        correlation = signal.correlate(target, ref, mode='full');
        lags = signal.correlation_lags(min_len, min_len, mode='full')
    except ValueError:
        return np.array([]), np.array([]), 0
    if len(lags) == 0: return np.array([]), np.array([]), 0
    lag = lags[np.argmax(correlation)]
    if abs(lag) > max_lag_samples: lag = np.sign(lag) * max_lag_samples
    if abs(lag) >= min_len: return np.array([]), np.array([]), lag
    if lag > 0:
        aligned_ref, aligned_target = ref[:-lag], target[lag:]
    elif lag < 0:
        aligned_ref, aligned_target = ref[abs(lag):], target[:-abs(lag)]
    else:
        aligned_ref, aligned_target = ref, target
    if len(aligned_ref) == 0: return np.array([]), np.array([]), lag
    return aligned_ref, aligned_target, lag


def align_with_fixed_lag(signal_ref, signal_target, lag):
    ref_np = np.asarray(signal_ref);
    target_np = np.asarray(signal_target)
    if ref_np.ndim != 1 or target_np.ndim != 1 or len(ref_np) == 0 or len(target_np) == 0: return np.array(
        []), np.array([])
    if lag > 0:
        if lag >= len(ref_np): return np.array([]), np.array([])
        ref_aligned, target_aligned = ref_np[:-lag], target_np[lag:]
    elif lag < 0:
        if abs(lag) >= len(target_np): return np.array([]), np.array([])
        ref_aligned, target_aligned = ref_np[abs(lag):], target_np[:-abs(lag)]
    else:
        ref_aligned, target_aligned = ref_np, target_np
    min_len = min(len(ref_aligned), len(target_aligned));
    if min_len == 0: return np.array([]), np.array([])
    return ref_aligned[:min_len], target_aligned[:min_len]


def butter_bandpass(low, high, fs, order=5):
    nyquist = 0.5 * fs;
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype='band');
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    if segments is None: return []
    b, a = butter_bandpass(low, high, fs, order=order)
    filtered_segments = [];
    num_failed = 0
    for i, seg in enumerate(segments):
        if len(seg) > 2 * order + 2:
            try:
                filtered = signal.filtfilt(b, a, seg)
            except ValueError as e:
                filtered = np.zeros_like(seg);
                num_failed += 1
            if not np.all(np.isfinite(filtered)): filtered = np.zeros_like(seg); num_failed += 1
            filtered_segments.append(filtered)
        else:
            filtered_segments.append(np.array([]));
            num_failed += 1
    return filtered_segments


# =============================================================================
# 3. N-L (Volterra + FRF) 模型函数
# =============================================================================

# --- L-Block Functions (Standard FRF) ---
def estimate_frf(input_segments, output_segments, fs, nfft):
    print(f"Estimating FRF for L-Block (using NFFT={nfft})...")

    valid_indices = [i for i, (inp, outp) in enumerate(zip(input_segments, output_segments))
                     if len(inp) > 0 and len(outp) > 0]
    if not valid_indices: raise ValueError("No valid segment pairs for FRF.")

    input_segments_valid = [input_segments[i] for i in valid_indices]
    output_segments_valid = [output_segments[i] for i in valid_indices]

    S_xx_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yy_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yx_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    processed_count = 0

    for i in range(len(input_segments_valid)):
        inp_seg = input_segments_valid[i]
        out_seg = output_segments_valid[i]

        seg_len = min(len(inp_seg), len(out_seg))
        if seg_len == 0: continue
        nperseg = min(seg_len, nfft)

        try:
            freqs_xx, Pxx = signal.welch(inp_seg, fs=fs, nperseg=nperseg, nfft=nfft, scaling='density', average='mean')
            freqs_yy, Pyy = signal.welch(out_seg, fs=fs, nperseg=nperseg, nfft=nfft, scaling='density', average='mean')
            freqs_yx, Pyx = signal.csd(out_seg, inp_seg, fs=fs, nperseg=nperseg, nfft=nfft, scaling='density',
                                       average='mean')

            if not np.all(np.isfinite(Pxx)) or not np.all(np.isfinite(Pyy)) or not np.all(np.isfinite(Pyx)):
                continue

            S_xx_sum += Pxx
            S_yy_sum += Pyy
            S_yx_sum += Pyx
            processed_count += 1
        except Exception as e:
            continue

    if processed_count == 0: raise ValueError("No valid segments processed for FRF.")
    print(f"Calculated FRF L-Block based on {processed_count} valid segments.")

    S_xx_avg = S_xx_sum / processed_count
    S_yy_avg = S_yy_sum / processed_count
    S_yx_avg = S_yx_sum / processed_count

    epsilon = 1e-12
    H1_freq = S_yx_avg / (S_xx_avg + epsilon)
    H2_freq = S_yy_avg / (np.conj(S_yx_avg) + epsilon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Hv_freq = np.sqrt(H1_freq * H2_freq)
    Hv_freq = np.nan_to_num(Hv_freq)

    freqs = np.fft.rfftfreq(nfft, d=1 / fs)

    return H1_freq, H2_freq, Hv_freq, freqs, nfft


def reconstruct_signal_from_frf(input_segments, H_freq_estimator, nfft, target_len_list):
    reconstructed_segments = []
    if len(H_freq_estimator) == 0:
        print("Warning: FRF estimator is empty. Returning zeros.")
        return [np.zeros(L) for L in target_len_list]

    if len(H_freq_estimator) != (nfft // 2 + 1):
        print(f"Warning: FRF estimator length ({len(H_freq_estimator)}) != NFFT/2+1 ({nfft // 2 + 1}). Cannot apply.")
        return [np.zeros(L) for L in target_len_list]

    H_full = np.concatenate((H_freq_estimator, np.conj(H_freq_estimator[-2:0:-1])))
    if len(H_full) != nfft:
        print(f"Warning: H_full length ({len(H_full)}) != NFFT ({nfft}). Padding/truncating H_full.")
        if len(H_full) > nfft:
            H_full = H_full[:nfft]
        else:
            H_full = np.pad(H_full, (0, nfft - len(H_full)), 'constant')

    for i, x_segment in enumerate(input_segments):
        target_len = target_len_list[i]

        if len(x_segment) == 0:
            reconstructed_segments.append(np.zeros(target_len))
            continue

        if len(x_segment) < nfft:
            x_segment_padded = np.pad(x_segment, (0, nfft - len(x_segment)), 'constant', constant_values=(0, 0))
        else:
            x_segment_padded = x_segment[:nfft]

        y_pred_freq = H_full * fft(x_segment_padded, n=nfft)
        y_pred_time_full = np.real(ifft(y_pred_freq))

        reconstructed_segments.append(y_pred_time_full[:target_len])

    return reconstructed_segments


# --- N-Block Functions (Dynamic Volterra) ---
def train_volterra_diag_nblock(input_segments, target_segments, memory_depth, orders, regularization_alpha=0.1):
    orders = sorted(list(set(o for o in orders if o > 0)))
    num_poly_terms = len(orders)
    if num_poly_terms == 0: print("Error: No valid orders provided for N-Block."); return None, None, None
    print(f"\nTraining Volterra N-Block (M={memory_depth}, Orders={orders}, Alpha={regularization_alpha})...")

    valid_inputs = [];
    valid_targets = [];
    for i in range(min(len(input_segments), len(target_segments))):
        inp = input_segments[i];
        target = target_segments[i]
        min_len_pair = min(len(inp), len(target))
        if min_len_pair > memory_depth:
            valid_inputs.append(inp[:min_len_pair])
            valid_targets.append(target[:min_len_pair])

    if not valid_inputs or not valid_targets: print(
        "Error: No valid segments for N-block training."); return None, None, None

    try:
        #  N-Block 现在标准化 *输入* 信号 (x)
        v_concat_for_norm = np.concatenate([v for v in valid_inputs])
        v_mean = np.mean(v_concat_for_norm);
        v_std = np.std(v_concat_for_norm)
        if v_std < 1e-9: v_std = 1.0
        print(f"N-Block *input* 'x' standardized (mean={v_mean:.2f}, std={v_std:.2f})")
    except Exception as e:
        print(f"Error during N-Block standardization: {e}. Using mean=0, std=1.")
        v_mean = 0.0;
        v_std = 1.0

    M = memory_depth
    num_coeffs = len(orders) * M
    total_valid_rows = sum(len(s) - M + 1 for s in valid_inputs)
    if total_valid_rows == 0: print(
        "Error: No valid rows for Volterra matrix (check memory depth vs segment length)."); return None, None, None

    Phi_N = np.zeros((total_valid_rows, num_coeffs))
    Y_target_N = np.zeros(total_valid_rows)
    current_row = 0

    for i in range(len(valid_inputs)):
        v_seg = (valid_inputs[i] - v_mean) / v_std  # 标准化 *输入* (x)
        y_seg = valid_targets[i]  # <--- 目标 (y) 未标准化
        seg_len = len(v_seg)
        num_samples_in_seg = seg_len - M + 1

        for n in range(num_samples_in_seg):
            v_delayed = v_seg[n + M - 1: n - 1 if n > 0 else None: -1]
            current_col_offset = 0
            for power in orders:
                with np.errstate(over='ignore', invalid='ignore'):
                    if power == 1:
                        v_delayed_poly = v_delayed
                    else:
                        v_delayed_poly = np.power(v_delayed, power)
                    if not np.all(np.isfinite(v_delayed_poly)): v_delayed_poly = np.clip(v_delayed_poly, -1e20, 1e20);
                    v_delayed_poly[~np.isfinite(v_delayed_poly)] = 0
                Phi_N[current_row, current_col_offset:current_col_offset + M] = v_delayed_poly
                current_col_offset += M
            Y_target_N[current_row] = y_seg[n + M - 1]  # <--- 目标是 'y'
            current_row += 1

    print("Solving linear system for Volterra N-Block coefficients...")
    try:
        model_N = Ridge(alpha=regularization_alpha, fit_intercept=False)
        model_N.fit(Phi_N, Y_target_N)
        volterra_coeffs = model_N.coef_
        print(f"N-Block training complete. Learned {len(volterra_coeffs)} coefficients.")
        return volterra_coeffs, v_mean, v_std
    except Exception as e:
        print(f"Error solving linear system for N block: {e}");
        return None, None, None


def apply_volterra_diag_nblock(input_segments, coeffs, memory_depth, orders, v_mean, v_std):
    processed_segments = []
    orders = sorted(list(set(o for o in orders if o > 0)))
    M = memory_depth
    if coeffs is None or len(coeffs) != len(orders) * M: print(
        f"Error: N-Block coeffs are None or mismatch. Returning zeros."); return [
        np.zeros_like(s) if len(s) > 0 else np.array([]) for s in input_segments]
    if not np.isfinite(v_mean) or not np.isfinite(v_std) or v_std < 1e-9: print(
        "Error: Invalid mean/std for N-Block application."); v_mean = 0.0; v_std = 1.0

    for v in input_segments:  # 'v' 在这里代表 *输入* (x)
        if len(v) < M: processed_segments.append(np.zeros_like(v)); continue
        v_scaled = (v - v_mean) / v_std  # 标准化 *输入* (x)
        y_pred = np.zeros_like(v_scaled, dtype=float)
        current_coeff_idx = 0
        for power in orders:
            h_coeffs = coeffs[current_coeff_idx: current_coeff_idx + M]
            with np.errstate(over='ignore', invalid='ignore'):
                if power == 1:
                    v_poly = v_scaled
                else:
                    v_poly = np.power(v_scaled, power)
                if not np.all(np.isfinite(v_poly)): v_poly = np.clip(v_poly, -1e20, 1e20); v_poly[
                    ~np.isfinite(v_poly)] = 0
            # y_pred 是 *中间信号* (w)
            y_pred += signal.lfilter(h_coeffs, [1.0], v_poly)
            current_coeff_idx += M
        processed_segments.append(y_pred)
    return processed_segments  # 返回的是 *中间信号* (w)


# =============================================================================
# 4. Metric Functions (V2 P-T metric - local extrema search)
# =============================================================================
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
    prominence = (std_seg * PEAK_PROMINENCE_STD_FACTOR) if (np.isfinite(std_seg) and std_seg >= 1e-9) else 1e-9
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
    base_std_for_clip = max(std_true, 1e-6 if np.std(pred_flat_in) > 1e-9 else 0)
    clip_lower = mean_true - CLIP_THRESHOLD_STD_FACTOR * base_std_for_clip;
    clip_upper = mean_true + CLIP_THRESHOLD_STD_FACTOR * base_std_for_clip
    if not np.isfinite(clip_lower) or not np.isfinite(clip_upper) or clip_lower >= clip_upper: clip_lower = np.min(
        true_flat_in) - 1e-6; clip_upper = np.max(true_flat_in) + 1e-6
    num_true_clipped = np.sum(true_flat_in < clip_lower) + np.sum(true_flat_in > clip_upper);
    num_pred_clipped = np.sum(pred_flat_in < clip_lower) + np.sum(pred_flat_in > clip_upper)

    if verbose_clipping:
        print(
            f"    Clipping Report: Bounds=[{clip_lower:.2f}, {clip_upper:.2f}] (Mean={mean_true:.2f}, Std={std_true:.2f})")
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
    try:
        rmse = np.sqrt(mean_squared_error(true_flat, pred_flat));
    except ValueError:
        rmse = np.inf
    if not np.isfinite(rmse): rmse = np.inf
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
    result_pt = get_peak_to_peak_amplitudes(true_flat, fs=SAMPLING_RATE);
    if len(result_pt) != 4: return metrics
    true_amps, true_peak_indices, true_trough_indices, _ = result_pt;
    if not true_amps: return metrics
    main_true_amps = [];
    main_true_peak_indices = [];
    main_true_trough_indices = [];
    window_radius_samples = int(MAIN_PEAK_WINDOW_SEC * SAMPLING_RATE)
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
    if verbose_clipping: print(f"    Local extrema search radius: +/- {PEAK_SEARCH_RADIUS} samples")
    for i in range(len(main_true_amps)):
        pk_idx = main_true_peak_indices[i];
        tr_idx = main_true_trough_indices[i];
        true_main_amp_i = main_true_amps[i]
        pk_window_start = max(0, pk_idx - PEAK_SEARCH_RADIUS);
        pk_window_end = min(len(pred_flat), pk_idx + PEAK_SEARCH_RADIUS + 1)
        tr_window_start = max(0, tr_idx - PEAK_SEARCH_RADIUS);
        tr_window_end = min(len(pred_flat), tr_idx + PEAK_SEARCH_RADIUS + 1)
        if pk_window_end > pk_window_start and tr_window_end > tr_window_start:
            try:
                pred_peak_val = np.max(pred_flat[pk_window_start:pk_window_end])
                pred_trough_val = np.min(pred_flat[tr_window_start:tr_window_end])
                if np.isfinite(pred_peak_val) and np.isfinite(pred_trough_val):
                    pred_main_amp_i = np.abs(pred_peak_val - pred_trough_val)
                    matched_main_true_amps.append(true_main_amp_i);
                    matched_main_pred_amps.append(pred_main_amp_i)
            except ValueError:
                continue
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
    min_list_len = min(len(ground_truth_list), len(predicted_list))
    valid_pairs = [(gt, pred) for gt, pred in zip(ground_truth_list[:min_list_len], predicted_list[:min_list_len]) if
                   gt is not None and pred is not None and len(gt) > 0 and len(pred) > 0 and len(gt) == len(pred)]
    if not valid_pairs: print("Input list contains no valid pairs or pairs with mismatched lengths."); return {}
    all_true_cropped = [];
    all_pred_cropped = []
    for true_seg, pred_seg in valid_pairs:
        min_len_pair = len(true_seg)
        if crop_len and min_len_pair >= crop_len:
            start = (min_len_pair - crop_len) // 2;
            end = start + crop_len;
            true_aligned = true_seg[
                           start:end];
            pred_aligned = pred_seg[
                           start:end]
        elif crop_len and min_len_pair < crop_len:
            continue
        else:
            true_aligned = true_seg;
            pred_aligned = pred_seg
        if len(true_aligned) < SAVGOL_WINDOW: continue
        all_true_cropped.append(true_aligned);
        all_pred_cropped.append(pred_aligned)
    if not all_true_cropped: print("No valid segments remaining after cropping/length check."); return {}
    true_flat = np.concatenate(all_true_cropped);
    pred_flat = np.concatenate(all_pred_cropped)
    metrics = calculate_metrics_core(true_flat, pred_flat, verbose_clipping=False)
    print(f"Correlation: {metrics['Correlation']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['Mean Absolute Error (MAE)']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"Main P-T Amp Err (%): {metrics['Main P-T Amp Err (%)']:.4f}")
    return metrics


def calculate_segment_metrics(ground_truth_seg, predicted_seg, crop_len=None):
    default_metrics = {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                       "Main P-T Amp Err (%)": np.nan}
    if ground_truth_seg is None or predicted_seg is None or len(ground_truth_seg) == 0 or len(
            predicted_seg) == 0: return default_metrics
    min_len = min(len(ground_truth_seg), len(predicted_seg));
    true_flat, pred_flat = ground_truth_seg[:min_len], predicted_seg[:min_len]
    if crop_len and min_len >= crop_len:
        start = (min_len - crop_len) // 2;
        end = start + crop_len;
        true_flat = true_flat[
                    start:end];
        pred_flat = pred_flat[
                    start:end]
    elif crop_len and min_len < crop_len:
        return default_metrics
    if len(true_flat) < SAVGOL_WINDOW: return default_metrics
    return calculate_metrics_core(true_flat, pred_flat, verbose_clipping=False)


# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()
    print(f"--- Analysis Start (Using {SEGMENT_LENGTH_SEC}s segments, Model: {MODEL_NAME}) ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False
    np.random.seed(RANDOM_SEED);
    random.seed(RANDOM_SEED)

    # --- Step 1: Load Real Data ---
    print("\n" + "=" * 50 + " Step 1: Loading REAL Data " + "=" * 50)
    try:
        train_on_raw = load_and_preprocess_single_channel(TRAIN_ON_BED_FILE, 0, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)
        train_under_raw = load_and_preprocess_single_channel(TRAIN_UNDER_BED_FILE, 0, CHANNEL_TO_ANALYZE,
                                                             SAMPLES_PER_SEGMENT)
        test_on_raw = load_and_preprocess_single_channel(TEST_ON_BED_FILE, 0, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)
        test_under_raw = load_and_preprocess_single_channel(TEST_UNDER_BED_FILE, 0, CHANNEL_TO_ANALYZE,
                                                            SAMPLES_PER_SEGMENT)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}");
        exit()
    if any(x is None for x in [train_on_raw, train_under_raw, test_on_raw, test_under_raw]): print(
        "\nError: Data loading failed."); exit()

    # --- Step 2: Filter ALL Raw Signals ---
    print("\n" + "=" * 50 + " Step 2: Applying Bandpass Filter to ALL Raw Signals " + "=" * 50)
    train_on_filtered = bandpass_filter_segments(train_on_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, FILTER_ORDER)
    train_under_filtered = bandpass_filter_segments(train_under_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE,
                                                    FILTER_ORDER)
    test_on_filtered = bandpass_filter_segments(test_on_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, FILTER_ORDER)
    test_under_filtered = bandpass_filter_segments(test_under_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE,
                                                   FILTER_ORDER)
    print("Filtering complete for all datasets.")

    # --- Step 3: Align FILTERED Training Signals INDIVIDUALLY ---
    print("\n" + "=" * 50 + " Step 3: Aligning FILTERED Training Signals Individually " + "=" * 50)
    train_lags = []
    train_on_final_filtered, train_under_final_filtered = [], []
    train_target_lengths = []
    num_fail_align_train = 0
    print("Aligning FILTERED training segments individually...")
    for i in range(min(len(train_on_filtered), len(train_under_filtered))):
        o_filt, u_filt = train_on_filtered[i], train_under_filtered[i]
        if len(o_filt) < 10 or len(u_filt) < 10:
            lag = np.nan;
            num_fail_align_train += 1
        else:
            aligned_u, aligned_o, lag = align_segments_cross_correlation(u_filt, o_filt, max_lag_samples=int(
                SAMPLING_RATE * 1))  # ref=u, target=o
            if len(aligned_u) == 0:
                lag = np.nan;
                num_fail_align_train += 1
            else:
                train_on_final_filtered.append(aligned_o)
                train_under_final_filtered.append(aligned_u)
                train_target_lengths.append(len(aligned_o))
        train_lags.append(lag)
    valid_lags = [l for l in train_lags if not np.isnan(l)]
    average_lag_train = int(np.round(np.mean(valid_lags))) if valid_lags else 0
    print(f"Alignment failures during training: {num_fail_align_train}")
    print(f"Calculated average training lag: {average_lag_train} samples (from {len(valid_lags)} segments).")
    print(f"Created {len(train_on_final_filtered)} individually aligned training pairs.")

    # --- Step 4: Align FILTERED Test Signals INDIVIDUALLY ---
    print("\n" + "=" * 50 + " Step 4: Aligning FILTERED Test Signals Individually " + "=" * 50)
    test_lags = []
    test_on_final_filtered, test_under_final_filtered = [], []
    test_target_lengths = []
    num_fail_align_test = 0

    print("Aligning FILTERED test segments individually...")
    for i in range(min(len(test_on_filtered), len(test_under_filtered))):
        o_filt, u_filt = test_on_filtered[i], test_under_filtered[i]
        if len(o_filt) < 10 or len(u_filt) < 10:
            lag = np.nan;
            num_fail_align_test += 1
        else:
            aligned_u, aligned_o, lag = align_segments_cross_correlation(u_filt, o_filt,
                                                                         max_lag_samples=int(SAMPLING_RATE * 1))
            if len(aligned_u) == 0:
                lag = np.nan;
                num_fail_align_test += 1
            else:
                test_on_final_filtered.append(aligned_o)
                test_under_final_filtered.append(aligned_u)
                test_target_lengths.append(len(aligned_o))
        test_lags.append(lag)

    valid_lags_test = [l for l in test_lags if not np.isnan(l)]
    average_lag_test = int(np.round(np.mean(valid_lags_test))) if valid_lags_test else 0
    print(f"Alignment failures during testing: {num_fail_align_test}")
    print(f"Calculated average test lag: {average_lag_test} samples (from {len(valid_lags_test)} segments).")
    print(f"Created {len(test_on_final_filtered)} individually aligned test pairs.")

    print(f"Final FILTERED training pairs: {len(train_on_final_filtered)}")
    print(f"Final FILTERED testing pairs: {len(test_on_final_filtered)}")
    if not train_under_final_filtered or not train_on_final_filtered: print(
        "Error: No valid FILTERED training segments remain after alignment."); exit()
    if not test_under_final_filtered or not test_on_final_filtered: print(
        "Warning: No valid FILTERED test segments remain after alignment. Evaluation might be empty.")

    # --- Step 5: Hyperparameter Tuning Loop (Hammerstein N-L) ---
    print("\n" + "=" * 70);
    print(f"Step 5: Hyperparameter Tuning for {MODEL_NAME} (Filt->Filt)");
    print("=" * 70)
    all_results = [];
    best_result = {'M': 0, 'Orders': [], 'Alpha': 0, 'AmpErr': 999.0, 'Corr': 0.0}
    best_nonlinear_coeffs = None;
    best_train_predicted_scaled = None;
    best_test_predicted_scaled = None
    best_v_mean = 0;
    best_v_std = 1;
    # 为最佳模型保存 L-Block
    best_frf_model_L_Block = None
    best_nfft_used = 0
    best_test_intermediate_w = None

    # --- N-L Hyperparameter Loop ---
    total_combinations = len(VOLTERRA_MEMORY_DEPTH_TO_TRY) * len(ORDERS_TO_TRY) * len(N_ALPHA_TO_TRY)
    current_combination = 0

    #  三重循环, 流程改为 N -> L
    for M in VOLTERRA_MEMORY_DEPTH_TO_TRY:
        for orders in ORDERS_TO_TRY:
            for n_alpha in N_ALPHA_TO_TRY:
                current_combination += 1
                orders_str = str(orders).replace(' ', '')
                param_str = f"M={M}, O={orders_str}, A={n_alpha}"
                print(f"\n--- Testing Combination {current_combination}/{total_combinations}: {param_str} ---")

                # Train N-Block (Volterra) (N-L: Input 'x' -> Target 'y') ---
                nonlinear_coeffs_current, v_mean_current, v_std_current = train_volterra_diag_nblock(
                    train_under_final_filtered,  #  N-Block 输入是 x
                    train_on_final_filtered,  # <--- N-Block 目标是 y
                    M,
                    orders,
                    regularization_alpha=n_alpha
                )
                if nonlinear_coeffs_current is None:
                    print(f"N-Block training failed for {param_str}. Skipping.")
                    all_results.append((M, orders_str, n_alpha, 999.0, 0.0))
                    continue

                # Apply N-Block (Volterra) -> Get Intermediate 'w' ---
                print(f"Applying N-Block to get intermediate signal 'w'...")
                train_intermediate_w = apply_volterra_diag_nblock(
                    train_under_final_filtered,
                    nonlinear_coeffs_current, M,
                    orders, v_mean_current, v_std_current
                )
                test_intermediate_w = apply_volterra_diag_nblock(
                    test_under_final_filtered,
                    nonlinear_coeffs_current, M,
                    orders, v_mean_current, v_std_current
                )

                # --- L-Block (FRF) Training (Dynamic, inside loop) ---
                # (L-Block 现在训练 'w' -> 'y')
                print("Training L-Block (FRF) on N-Block output 'w'...")
                min_len_train_final_list_w = [len(s) for s in train_intermediate_w if len(s) > 0]
                if not min_len_train_final_list_w:
                    print("Error: No valid intermediate 'w' segments from N-Block. Skipping.")
                    all_results.append((M, orders_str, n_alpha, 999.0, 0.0))
                    continue

                min_len_train_final_w = min(min_len_train_final_list_w)
                # 根据 'w' 的长度动态计算 NFFT
                NFFT_L_Block = int(2 ** np.ceil(np.log2(min_len_train_final_w)))
                if NFFT_L_Block < 256: NFFT_L_Block = 256

                try:
                    H1_freq, H2_freq, Hv_freq, frf_freqs, nfft_used = estimate_frf(
                        train_intermediate_w,  #  L-Block 输入是 'w'
                        train_on_final_filtered,  # <--- L-Block 目标是 'y'
                        SAMPLING_RATE,
                        NFFT_L_Block
                    )
                    estimators = {'H1': H1_freq, 'H2': H2_freq, 'Hv_geometric_mean': Hv_freq}
                    frf_model_L_Block = estimators[CHOSEN_FRF_ESTIMATOR]
                    if frf_model_L_Block is None or len(frf_model_L_Block) == 0:
                        raise ValueError("FRF L-Block estimation failed.")
                except Exception as e:
                    print(f"L-Block (FRF) training failed: {e}. Skipping.")
                    all_results.append((M, orders_str, n_alpha, 999.0, 0.0))
                    continue

                # --- Apply L-Block (FRF) (Dynamic) -> Get Final Output ---
                train_predicted_raw_L = reconstruct_signal_from_frf(
                    train_intermediate_w,  #  L-Block 输入是 'w'
                    frf_model_L_Block, nfft_used,
                    train_target_lengths  # 目标长度列表保持不变
                )
                test_predicted_raw_L = reconstruct_signal_from_frf(
                    test_intermediate_w,  # <--- L-Block 输入是 'w'
                    frf_model_L_Block, nfft_used,
                    test_target_lengths  # 目标长度列表保持不变
                )


                # Part A: 计算 *一个* 全局缩放因子 (仅使用训练集, 基于 L-Block 的最终输出)
                segment_scaling_factors = []
                num_bad_train_scale = 0
                # [!! MODIFIED !!]
                min_len_scale = min(len(train_predicted_raw_L), len(train_on_final_filtered))

                for i in range(min_len_scale):
                    seg_pred = train_predicted_raw_L[i]
                    seg_target = train_on_final_filtered[i]

                    if (seg_pred is None or seg_target is None or
                            len(seg_pred) < 2 or len(seg_target) < 2 or
                            not np.all(np.isfinite(seg_pred)) or not np.all(np.isfinite(seg_target))):
                        num_bad_train_scale += 1
                        continue

                    std_input = np.std(seg_pred)
                    std_target = np.std(seg_target)

                    if std_input > 1e-6 and std_target > 1e-6:
                        scaling_factor_seg = std_target / std_input
                        scaling_factor_seg = np.clip(scaling_factor_seg, 0.01, 100.0)
                        if np.isfinite(scaling_factor_seg):
                            segment_scaling_factors.append(scaling_factor_seg)
                        else:
                            num_bad_train_scale += 1
                    else:
                        num_bad_train_scale += 1

                valid_scaling_factors = [f for f in segment_scaling_factors if np.isfinite(f)]
                avg_scaling_factor = np.mean(valid_scaling_factors) if valid_scaling_factors else 1.0
                if not np.isfinite(avg_scaling_factor):
                    avg_scaling_factor = 1.0
                print(
                    f"Calculated average scaling factor: {avg_scaling_factor:.4f} (from {len(valid_scaling_factors)} valid segments, {num_bad_train_scale} failed)")

                # Part B: 将 *同一个* 全局因子应用到 *训练集*
                train_predicted_scaled_current = []

                for i in range(len(train_predicted_raw_L)):
                    seg_pred_train = train_predicted_raw_L[i]
                    if seg_pred_train is None or len(seg_pred_train) < 2 or not np.all(np.isfinite(seg_pred_train)):
                        train_predicted_scaled_current.append(np.array([]))
                        continue
                    scaled_seg_train = seg_pred_train * avg_scaling_factor
                    if not np.all(np.isfinite(scaled_seg_train)):
                        train_predicted_scaled_current.append(np.zeros_like(scaled_seg_train))
                    else:
                        train_predicted_scaled_current.append(scaled_seg_train)
                if len(train_predicted_scaled_current) < len(train_on_final_filtered):
                    train_predicted_scaled_current.extend(
                        [np.array([])] * (len(train_on_final_filtered) - len(train_predicted_scaled_current)))

                # Part C: 将 *同一个* 全局因子应用到 *测试集*
                test_predicted_scaled_current = []

                for i in range(len(test_predicted_raw_L)):
                    seg_pred_test = test_predicted_raw_L[i]
                    if seg_pred_test is None or len(seg_pred_test) < 2 or not np.all(np.isfinite(seg_pred_test)):
                        test_predicted_scaled_current.append(np.array([]))
                        continue
                    scaled_seg_test = seg_pred_test * avg_scaling_factor
                    if not np.all(np.isfinite(scaled_seg_test)):
                        test_predicted_scaled_current.append(np.zeros_like(scaled_seg_test))
                    else:
                        test_predicted_scaled_current.append(scaled_seg_test)
                if len(test_predicted_scaled_current) < len(test_on_final_filtered):
                    test_predicted_scaled_current.extend(
                        [np.array([])] * (len(test_on_final_filtered) - len(test_predicted_scaled_current)))


                # --- Evaluate ---
                print(f"--- Evaluating {param_str} (vs FILTERED Ground Truth) ---")
                train_metrics_agg = calculate_and_print_metrics(train_on_final_filtered, train_predicted_scaled_current,
                                                                f"Training Set ({param_str})", crop_len=CROP_LENGTH)
                test_metrics_agg = calculate_and_print_metrics(test_on_final_filtered, test_predicted_scaled_current,
                                                               f"Test Set ({param_str})", crop_len=CROP_LENGTH)
                test_amp_err = test_metrics_agg.get("Main P-T Amp Err (%)", 999.0);
                test_corr = test_metrics_agg.get("Correlation", 0.0)
                if not np.isfinite(test_amp_err): test_amp_err = 999.0
                if not np.isfinite(test_corr): test_corr = 0.0
                all_results.append((M, orders_str, n_alpha, test_amp_err, test_corr))

                # --- Update Best Model (Based on Test AmpErr) ---
                if test_amp_err < best_result['AmpErr']:
                    print(f"*** New Best Model Found! Test AmpErr={test_amp_err:.2f}% ({param_str}) ***")
                    best_result = {'M': M, 'Orders': orders, 'Alpha': n_alpha, 'AmpErr': test_amp_err,
                                   'Corr': test_corr}
                    best_nonlinear_coeffs = nonlinear_coeffs_current
                    best_v_mean = v_mean_current
                    best_v_std = v_std_current
                    best_train_predicted_scaled = train_predicted_scaled_current
                    best_test_predicted_scaled = test_predicted_scaled_current

                    # --- 保存最佳 L-Block 和中间信号 ---
                    best_frf_model_L_Block = frf_model_L_Block
                    best_nfft_used = nfft_used
                    best_test_intermediate_w = test_intermediate_w

    # --- End Hyperparameter Loops ---

    end_time = time.time()
    print(f"\nHyperparameter tuning finished in {end_time - start_time:.2f} seconds.")

    # --- Print Summary Table ---
    print("\n" + "=" * 90);
    print(f"Hyperparameter Tuning Summary ({MODEL_NAME})");
    print("=" * 90)
    print(
        f"{'M (N-Block)':<12} | {'Orders':<15} | {'N_Alpha':<10} | {'Test Amp Err (%)':<18} | {'Test Correlation':<18}")
    print("-" * 90)
    all_results.sort(key=lambda x: x[3])  # Sort by AmpErr
    for M, O, A, AmpErr, Corr in all_results: print(f"{M:<12} | {O:<15} | {A:<10.2f} | {AmpErr:<18.2f} | {Corr:<18.4f}")

    print("\n" + "=" * 90);
    print("Best Model Found");
    print("=" * 90)
    best_orders_str = str(best_result['Orders']).replace(' ', '')
    print(f"N-Block Model:    Volterra (Dynamic)")
    print(f"N-Block Memory M: {best_result['M']}")
    print(f"N-Block Orders:   {best_orders_str}")
    print(f"N-Block Alpha:    {best_result['Alpha']:.2f}")
    print(f"L-Block Model:    FRF ({CHOSEN_FRF_ESTIMATOR})")
    print(f"Test Correlation: {best_result['Corr']:.4f}")
    print(f"Test Main P-T Amp Err (%): {best_result['AmpErr']:.2f}%")

    # --- Step 6.5 & 7: Detailed Metrics and Visualization for BEST Model ---
    if best_test_predicted_scaled is None:
        print("\nNo valid model was trained. Skipping detailed metrics and visualization.")
    else:
        MODEL_NAME_BEST = f"Best {MODEL_NAME} (M={best_result['M']}, O={best_orders_str}, A={best_result['Alpha']:.2f})"

        # --- N-Block 诊断图 (在 Test 集上) ---
        print("\n" + "=" * 70);

        print(f"Step 6.5: Visualizing N-Block *Only* Output vs Target (Test Set)");
        print("=" * 70)
        plot_idx_test_lblock = 8  # 选择一个 Test 片段

        #  检查 'best_test_intermediate_w' 是否存在
        if 'best_test_intermediate_w' in locals() and best_test_intermediate_w is not None and \
                plot_idx_test_lblock < len(test_on_final_filtered) and \
                plot_idx_test_lblock < len(best_test_intermediate_w):

            #  plot_l_output 现在是 N-Block 的输出 'w'
            plot_l_output = best_test_intermediate_w[plot_idx_test_lblock]
            plot_l_target = test_on_final_filtered[plot_idx_test_lblock]  # 最终目标 (y)

            # 诊断 N-Block 的输出
            if len(plot_l_output) > 0 and len(plot_l_target) > 0:
                print(f"    N-Block Diagnostic (Seg {plot_idx_test_lblock}):")
                print(f"    N-Block Output ('w') StdDev: {np.std(plot_l_output):.4f}")
                print(f"    Target ('y') StdDev:         {np.std(plot_l_target):.4f}")

            if len(plot_l_output) == len(plot_l_target):

                plt.suptitle(f"N-Block *Only* Output ('w') vs Target ('y') (Test Segment {plot_idx_test_lblock})",
                             fontsize=16,
                             y=0.97)
                plt.subplot(2, 1, 1);
                time_axis_full_l = np.arange(len(plot_l_target)) / SAMPLING_RATE
                plt.plot(time_axis_full_l, plot_l_target, label="Target (FILTERED 'Before')", color='darkred',
                         linewidth=1.5)

                plt.plot(time_axis_full_l, plot_l_output, label="N-Block Output ('w')", color='purple', linestyle='--',
                         alpha=0.8)
                plt.title("Full Time Series");
                plt.ylabel("Amplitude");
                plt.legend();
                plt.grid(True, alpha=0.5)
                plt.subplot(2, 1, 2)
                if len(plot_l_target) >= CROP_LENGTH:
                    zoom_start_l = (len(plot_l_target) - CROP_LENGTH) // 2;
                    zoom_end_l = zoom_start_l + CROP_LENGTH;
                    time_axis_zoom_l = np.arange(zoom_start_l, zoom_end_l)
                    plot_l_target_zoom = plot_l_target[zoom_start_l:zoom_end_l];
                    plot_l_output_zoom = plot_l_output[zoom_start_l:zoom_end_l]
                    plt.plot(time_axis_zoom_l, plot_l_target_zoom, label="Target (FILTERED 'Before')", color='darkred',
                             linewidth=1.5)
                    # [!! MODIFIED !!]
                    plt.plot(time_axis_zoom_l, plot_l_output_zoom, label="N-Block Output ('w')", color='purple',
                             linestyle='--', alpha=0.8)
                    plt.title(f"Zoomed View (Middle {CROP_LENGTH} Samples)");
                    plt.xlabel("Sample Index");
                    plt.ylabel("Amplitude");
                    plt.legend();
                    plt.grid(True, alpha=0.5)
                plt.tight_layout(rect=[0, 0, 1, 0.95]);
                plt.show()
            else:
                print(f"Skipping N-Block plot for segment {plot_idx_test_lblock}: Length mismatch.")
        else:
            print(
                f"Skipping N-Block plot: Test index {plot_idx_test_lblock} out of bounds or 'best_test_intermediate_w' not found.")

        print("\n" + "=" * 50 + f" Step 7: Per-Segment Metrics ({MODEL_NAME_BEST}, Filt->Filt)" + "=" * 50)
        all_train_metrics = [];
        all_test_metrics = []
        print("Calculating Training segment metrics for best model...")
        for i in range(len(train_on_final_filtered)): metrics = calculate_segment_metrics(train_on_final_filtered[i],
                                                                                          best_train_predicted_scaled[
                                                                                              i] if i < len(
                                                                                              best_train_predicted_scaled) else None,
                                                                                          crop_len=CROP_LENGTH); all_train_metrics.append(
            metrics)
        print(f"Calculated {len(all_train_metrics)} training segment metrics.")
        print("Calculating Test segment metrics for best model...")
        for i in range(len(test_on_final_filtered)):
            metrics = calculate_segment_metrics(test_on_final_filtered[i], best_test_predicted_scaled[i] if i < len(
                best_test_predicted_scaled) else None, crop_len=CROP_LENGTH)
            all_test_metrics.append(metrics)
            corr_str = f"{metrics['Correlation']:.4f}";
            mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}";
            rmse_str = f"{metrics['RMSE']:.4f}";
            main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%"
            print(
                f"  Test Segment #{i:03d}: Corr={corr_str}, MAE={mae_str}, RMSE={rmse_str}, Main P-T Amp Err={main_pt_err_str}")

        print("\n" + "=" * 70);
        print(f"Step 8: Visualizing Final Results for {MODEL_NAME_BEST} (Filt->Filt)");
        print("=" * 70)
        plot_idx_train = 60  # Example train segment
        if (0 <= plot_idx_train < len(train_on_final_filtered) and 0 <= plot_idx_train < len(
                best_train_predicted_scaled) and 0 <= plot_idx_train < len(train_under_final_filtered)):
            plot_truth_filt = train_on_final_filtered[plot_idx_train];
            plot_recon = best_train_predicted_scaled[plot_idx_train];
            plot_input_filt = train_under_final_filtered[plot_idx_train]
            plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon))
            plt.figure(figsize=(20, 12));
            plt.suptitle(f"Training Set ({MODEL_NAME_BEST}) - Segment {plot_idx_train}", fontsize=20, y=0.98)
            plt.subplot(2, 1, 2)
            if len(plot_truth_filt) >= CROP_LENGTH:
                zoom_start = (len(plot_truth_filt) - CROP_LENGTH) // 2;
                zoom_end = zoom_start + CROP_LENGTH;
                time_axis_zoom = np.arange(zoom_start, zoom_end)
                plot_truth_zoom = plot_truth_filt[zoom_start:zoom_end];
                plot_recon_zoom = plot_recon[zoom_start:zoom_end] if plot_recon_valid and len(
                    plot_recon) >= zoom_end else np.full(CROP_LENGTH, np.nan);
                plot_input_filt_zoom = plot_input_filt[zoom_start:zoom_end]
                plt.plot(time_axis_zoom, plot_truth_zoom, label="Target: FILTERED 'Before'", color='black',
                         linewidth=1.5, zorder=3);
                plt.plot(time_axis_zoom, plot_recon_zoom, label=f'Reconstructed Signal', color='dodgerblue',
                         linestyle='--', linewidth=1.5, zorder=2)
                plt.plot(time_axis_zoom, plot_input_filt_zoom, label="Input: FILTERED 'After'", color='green',
                         linestyle=':', linewidth=1.0, zorder=1, alpha=0.7)
                true_peaks_idx, true_troughs_idx = get_peak_indices(plot_truth_zoom, fs=SAMPLING_RATE);
                pred_peaks_idx, pred_troughs_idx = get_peak_indices(plot_recon_zoom, fs=SAMPLING_RATE)
                plt.scatter(time_axis_zoom[true_peaks_idx], plot_truth_zoom[true_peaks_idx], color='red', marker='o',
                            s=50, label='True Peaks', zorder=5, alpha=0.8);
                plt.scatter(time_axis_zoom[true_troughs_idx], plot_truth_zoom[true_troughs_idx], color='red',
                            marker='x', s=50, label='True Troughs', zorder=5, alpha=0.8)
                plt.scatter(time_axis_zoom[pred_peaks_idx], plot_recon_zoom[pred_peaks_idx], color='lime', marker='o',
                            s=30, label='Pred Peaks', zorder=4, alpha=0.8);
                plt.scatter(time_axis_zoom[pred_troughs_idx], plot_recon_zoom[pred_troughs_idx], color='lime',
                            marker='x', s=30, label='Pred Troughs', zorder=4, alpha=0.8)
                if 0 <= plot_idx_train < len(all_train_metrics):
                    metrics = all_train_metrics[plot_idx_train]
                    corr_str = f"{metrics['Correlation']:.4f}";
                    mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}";
                    rmse_str = f"{metrics['RMSE']:.4f}";
                    main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%"
                    metrics_text = f"Segment Metrics (Cropped, Local Extrema P-T):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain P-T Amp Err: {main_pt_err_str}"
                    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=11,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                plt.title(f'Zoomed-in View (Middle {CROP_LENGTH} Samples)', fontsize=16);
                plt.xlabel('Sample Index');
                plt.ylabel('Amplitude');
                plt.legend(loc='upper right', fontsize=9);
                plt.grid(True)
            plt.subplot(2, 1, 1);
            time_axis_full = np.arange(len(plot_truth_filt)) / SAMPLING_RATE
            plt.plot(time_axis_full, plot_truth_filt, label="Target: FILTERED 'Before'", color='black', linewidth=1.5);
            plt.plot(time_axis_full, plot_recon if plot_recon_valid else np.full_like(plot_truth_filt, np.nan),
                     label=f'Reconstructed Signal', color='dodgerblue', linestyle='--', linewidth=1.5)
            plt.plot(time_axis_full, plot_input_filt, label="Input: FILTERED 'After'", color='green', linestyle=':',
                     linewidth=1.0, alpha=0.7)
            plt.title('Full Time Series (Aligned & Filtered)', fontsize=16);
            plt.xlabel('Time (s)');
            plt.ylabel('Signal Amplitude');
            plt.legend();
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 1, 0.95]);
            plt.show()
        else:
            print(f"Skipping train plot: Index {plot_idx_train} out of bounds.")

        plot_idx_test = 8  # Example test segment
        if (best_test_predicted_scaled and 0 <= plot_idx_test < len(test_on_final_filtered) and
                0 <= plot_idx_test < len(best_test_predicted_scaled) and
                0 <= plot_idx_test < len(test_under_final_filtered)):

            plot_truth_filt = test_on_final_filtered[plot_idx_test];
            plot_recon = best_test_predicted_scaled[plot_idx_test];
            plot_input_filt = test_under_final_filtered[plot_idx_test]
            plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon))
            plt.figure(figsize=(20, 12));
            plt.suptitle(f"Test Set ({MODEL_NAME_BEST}) - Segment {plot_idx_test}", fontsize=20, y=0.98)
            plt.subplot(2, 1, 2)
            if len(plot_truth_filt) >= CROP_LENGTH:
                zoom_start = (len(plot_truth_filt) - CROP_LENGTH) // 2;
                zoom_end = zoom_start + CROP_LENGTH
                time_axis_zoom = np.arange(zoom_start, zoom_end)
                plot_truth_zoom = plot_truth_filt[zoom_start:zoom_end];
                plot_recon_zoom = plot_recon[zoom_start:zoom_end] if plot_recon_valid and len(
                    plot_recon) >= zoom_end else np.full(CROP_LENGTH, np.nan);
                plot_input_filt_zoom = plot_input_filt[zoom_start:zoom_end]
                plt.plot(time_axis_zoom, plot_truth_zoom, label="Target: FILTERED 'Before'", color='darkred',
                         linewidth=1.5, zorder=3);
                plt.plot(time_axis_zoom, plot_recon_zoom, label=f'Reconstructed Signal', color='darkgreen',
                         linestyle='--', linewidth=1.5, zorder=2)
                plt.plot(time_axis_zoom, plot_input_filt_zoom, label="Input: FILTERED 'After'", color='blue',
                         linestyle=':', linewidth=1.0, zorder=1, alpha=0.7)
                true_peaks_idx, true_troughs_idx = get_peak_indices(plot_truth_zoom, fs=SAMPLING_RATE);
                pred_peaks_idx, pred_troughs_idx = get_peak_indices(plot_recon_zoom, fs=SAMPLING_RATE)
                plt.scatter(time_axis_zoom[true_peaks_idx], plot_truth_zoom[true_peaks_idx], color='red', marker='o',
                            s=50, label='True Peaks', zorder=5, alpha=0.8);
                plt.scatter(time_axis_zoom[true_troughs_idx], plot_truth_zoom[true_troughs_idx], color='red',
                            marker='x', s=50, label='True Troughs', zorder=5, alpha=0.8)
                plt.scatter(time_axis_zoom[pred_peaks_idx], plot_recon_zoom[pred_peaks_idx], color='lime', marker='o',
                            s=30, label='Pred Peaks', zorder=4, alpha=0.8);
                plt.scatter(time_axis_zoom[pred_troughs_idx], plot_recon_zoom[pred_troughs_idx], color='lime',
                            marker='x', s=30, label='Pred Troughs', zorder=4, alpha=0.8)
                if 0 <= plot_idx_test < len(all_test_metrics):
                    metrics = all_test_metrics[plot_idx_test]
                    corr_str = f"{metrics['Correlation']:.4f}";
                    mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}";
                    rmse_str = f"{metrics['RMSE']:.4f}";
                    main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%"
                    metrics_text = f"Segment Metrics (Cropped, Local Extrema P-T):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain P-T Amp Err: {main_pt_err_str}"
                    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=11,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                plt.title(f'Zoomed-in View (Middle {CROP_LENGTH} Samples)', fontsize=16);
                plt.xlabel('Sample Index');
                plt.ylabel('Amplitude');
                plt.legend(loc='upper right', fontsize=9);
                plt.grid(True)
            plt.subplot(2, 1, 1)
            time_axis_full = np.arange(len(plot_truth_filt)) / SAMPLING_RATE
            plt.plot(time_axis_full, plot_truth_filt, label="Target: FILTERED 'Before'", color='darkred',
                     linewidth=1.5);
            plt.plot(time_axis_full, plot_recon if plot_recon_valid else np.full_like(plot_truth_filt, np.nan),
                     label=f'Reconstructed Signal', color='darkgreen', linestyle='--', linewidth=1.5)
            plt.plot(time_axis_full, plot_input_filt, label="Input: FILTERED 'After'", color='blue', linestyle=':',
                     linewidth=1.0, alpha=0.7)
            plt.title('Full Time Series (Aligned & Filtered)', fontsize=16);
            plt.xlabel('Time (s)');
            plt.ylabel('Signal Amplitude');
            plt.legend();
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 1, 0.95]);
            plt.show()
        else:
            print(f"Skipping test plot: Index {plot_idx_test} out of bounds or data invalid.")

    print("\n" + "=" * 70);
    print("All Processes Completed");
    print("=" * 70)