import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error  # 导入 RMSE 计算
from sklearn.linear_model import Ridge  # <--- 【新增】导入用于 N-Block 训练

# =============================================================================
# 1. Core Configuration
# =============================================================================
SAMPLING_RATE = 100
CHANNEL_TO_ANALYZE = 0
SAMPLES_PER_SEGMENT = 1000

# --- 【修改】Hybrid FFT + Simple N 模型参数 ---
# L-Block (Hybrid FFT part)
FIXED_GAMMA = 0.35
BEST_CUTOFF_HZ = 15.0
# N-Block (Simple Non-linearity part)
N_POLY_TERMS = 2  # N-Block 的项数 (例如: 2 表示 v, v^3)
N_ALPHA = 0.1  # N-Block 的 L2 惩罚强度 (设置一个较小的值)
# --- 结束修改 ---

# --- Exponential Attenuation Parameters ---
CORNER_FREQ = 6.0
ATTENUATION_FACTOR = 0.5
RANDOM_SEED = 42

# --- Noise Parameters ---
FIXED_NOISE_STD = 0.05

# --- Metric Configuration ---
CROP_LENGTH = 800
CLIP_THRESHOLD_STD_FACTOR = 20  # 裁剪阈值因子
MAIN_PEAK_WINDOW_SEC = 0.75  # <<<--- 时间窗口半径 (秒) - 用于版本1的主峰筛选

# --- Peak Finding Parameters ---
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3
PEAK_PROMINENCE_STD_FACTOR = 0.1
PEAK_MIN_DISTANCE_SEC = 0.1

# --- Data File Paths ---
TRAIN_ON_BED_RAW_FILE = r"./data/vibration_analysis.npy"
TRAIN_UNDER_BED_SAVE_FILE = "train_under_bed_exponential_attenuation.npy"
TEST_ON_BED_RAW_FILE = r"./data/BSG.npy"
TEST_UNDER_BED_SAVE_FILE = "test_under_bed_exponential_attenuation.npy"


# =============================================================================
# 2. Core Helper Functions
# =============================================================================
# (load_and_segment_signal, generate_under_bed_signals,
#  align_segments_cross_correlation, align_with_fixed_lag - 无修改)
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


# =============================================================================
# 3. 【修改】Hybrid FFT + Simple N 模型函数
# =============================================================================

# --- L-Block Functions ---
def estimate_frf_for_phase(input_segments, output_segments, fs):
    min_len_list = [len(s) for s in input_segments if len(s) > 0] + [len(s) for s in output_segments if len(s) > 0]
    if not min_len_list: return np.array([]), np.array([])
    min_len = min(min_len_list);
    if min_len < 2: return np.array([]), np.array([])
    nfft = int(2 ** np.floor(np.log2(min_len)))
    if nfft < 2: return np.array([]), np.array([])
    S_xy_sum, S_xx_sum = np.zeros(nfft // 2 + 1, dtype=complex), np.zeros(nfft // 2 + 1, dtype=complex);
    valid_pairs = 0
    for i in range(len(input_segments)):
        if len(input_segments[i]) < nfft or len(output_segments[i]) < nfft: continue
        inp_seg, out_seg = input_segments[i][:nfft], output_segments[i][:nfft]
        if np.std(inp_seg) < 1e-9 or np.std(out_seg) < 1e-9: continue
        try:
            f, Pxy = signal.csd(out_seg, inp_seg, fs=fs, nperseg=nfft);
            _, Pxx = signal.welch(inp_seg, fs=fs, nperseg=nfft)
            if not np.all(np.isfinite(Pxy)) or not np.all(np.isfinite(Pxx)): continue
            S_xy_sum += Pxy;
            S_xx_sum += Pxx;
            valid_pairs += 1
        except ValueError:
            continue
    if valid_pairs == 0: return np.array([]), np.array([])
    H1 = (S_xy_sum / valid_pairs) / (S_xx_sum / valid_pairs + 1e-9)
    if not np.all(np.isfinite(H1)): return f, np.ones_like(H1)
    return f, H1


def apply_hybrid_fft_model(input_segments, fs, gamma, cutoff_hz, frf_model_freqs, frf_model_H):
    """
    应用 Hybrid FFT 模型 (现在作为 L-Block)
    """
    processed = [];
    num_bad_segments = 0
    if len(frf_model_freqs) == 0 or len(frf_model_H) == 0: return [np.zeros_like(seg) if len(seg) > 0 else np.array([])
                                                                   for seg in input_segments]
    unwrapped_angle = np.unwrap(np.angle(frf_model_H))
    for i, seg in enumerate(input_segments):
        if len(seg) == 0: processed.append(np.array([])); continue
        n = len(seg);
        seg_fft = fft(seg);
        freqs = fftfreq(n, 1 / fs);
        abs_freqs = np.abs(freqs)
        magnitude_curve = np.power(abs_freqs, gamma, where=abs_freqs > 1e-9, out=np.ones_like(abs_freqs));
        magnitude_curve[freqs == 0] = 1.0
        freq_1hz_idx = np.abs(freqs - 1.0).argmin();
        norm_factor = 1.0
        if freq_1hz_idx > 0 and magnitude_curve[freq_1hz_idx] > 1e-9: norm_factor = magnitude_curve[freq_1hz_idx]
        if norm_factor > 1e-9:
            magnitude_curve /= norm_factor
        else:
            norm_factor = 1.0
        if norm_factor > 1e-9:
            cutoff_gain = np.power(cutoff_hz, gamma) / norm_factor
        else:
            cutoff_gain = np.power(cutoff_hz, gamma)
        magnitude_curve[abs_freqs > cutoff_hz] = cutoff_gain
        phase_response = np.interp(abs_freqs, frf_model_freqs, unwrapped_angle, left=unwrapped_angle[0],
                                   right=unwrapped_angle[-1])
        H_hybrid = magnitude_curve * np.exp(1j * phase_response);
        equalized_fft = seg_fft * H_hybrid
        processed_seg = np.real(ifft(equalized_fft))
        if not np.all(np.isfinite(processed_seg)): processed_seg = np.zeros_like(seg); num_bad_segments += 1
        processed.append(processed_seg)
    if num_bad_segments > 0: print(
        f"\nWarning: {num_bad_segments} segments encountered non-finite values during Hybrid FFT application.")
    return processed


# --- 【新增】N-Block Functions ---
def train_simple_nonlinearity(intermediate_segments, target_segments, num_poly_terms, regularization_alpha=0.1):
    print(f"\nTraining Simple N-Block with {num_poly_terms} terms, Alpha={regularization_alpha}...")
    valid_intermediate = []
    valid_targets = []
    for i in range(len(intermediate_segments)):
        inter = intermediate_segments[i]  # v(k) from Hybrid FFT
        target = target_segments[i]  # y(k) original on-bed
        min_len = min(len(inter), len(target))
        if min_len > 0:
            valid_intermediate.append(inter[:min_len])
            valid_targets.append(target[:min_len])
    if not valid_intermediate or not valid_targets:
        print("Error: No valid segments for N-block training.")
        return None

    v_concat = np.concatenate(valid_intermediate)
    y_concat = np.concatenate(valid_targets)
    Phi_N = np.zeros((len(v_concat), num_poly_terms))
    for i in range(num_poly_terms):
        power = 2 * i + 1  # 奇次幂: 1, 3, 5, ...
        print(f"  Adding polynomial term v^{power}")
        with np.errstate(over='ignore', invalid='ignore'):
            term_vec = np.power(v_concat, power)
            if not np.all(np.isfinite(term_vec)):
                print(f"Warning: Overflow/invalid value in power {power} during N-block training. Clipping.")
                term_vec = np.clip(term_vec, -1e20, 1e20)
                term_vec[~np.isfinite(term_vec)] = 0
            Phi_N[:, i] = term_vec
    print("Solving linear system for N-Block coefficients...")
    try:
        model_N = Ridge(alpha=regularization_alpha, fit_intercept=False)
        model_N.fit(Phi_N, y_concat)
        nonlinear_coeffs = model_N.coef_
        print(f"N-Block training complete. Learned {len(nonlinear_coeffs)} coefficients: {nonlinear_coeffs}")
        return nonlinear_coeffs
    except Exception as e:
        print(f"Error solving linear system for N block: {e}")
        return None


def apply_polynomial_nonlinearity(segments, coeffs):
    """
    应用静态非线性多项式 N(v)
    """
    processed_segments = []
    for v in segments:  # v 是来自 L-Block (Hybrid FFT) 的中间信号
        if len(v) == 0:
            processed_segments.append(np.array([]))
            continue

        y_pred = np.zeros_like(v, dtype=float)
        for i, c in enumerate(coeffs):
            power = 2 * i + 1  # 奇次幂: 1, 3, 5, ...
            if c != 0:
                with np.errstate(over='ignore', invalid='ignore'):
                    term = c * np.power(v, power)
                    if not np.all(np.isfinite(term)):
                        term = np.clip(term, -1e20, 1e20)
                        term[~np.isfinite(term)] = 0
                y_pred += term
        processed_segments.append(y_pred)
    return processed_segments


# =============================================================================
# 4. Metric Functions
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
    metrics = {
        "Correlation": np.nan,
        "Mean Absolute Error (MAE)": np.nan,
        "RMSE": np.nan,
        "Main P-T Amp Err (%)": np.nan
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
    result_pt = get_peak_to_peak_amplitudes(true_flat, fs=SAMPLING_RATE)
    if len(result_pt) != 4: return metrics
    true_amps, true_peak_indices, true_trough_indices, _ = result_pt
    if not true_amps: return metrics
    main_true_amps = []
    main_true_peak_indices = []
    main_true_trough_indices = []
    window_radius_samples = int(MAIN_PEAK_WINDOW_SEC * SAMPLING_RATE)
    if not true_amps: return metrics
    np_true_peak_indices = np.array(true_peak_indices)
    np_true_trough_indices = np.array(true_trough_indices)
    np_true_amps = np.array(true_amps)
    if len(np_true_peak_indices) == 0 or len(np_true_trough_indices) == 0: return metrics
    mid_indices = (np_true_peak_indices + np_true_trough_indices) / 2.0
    added_indices_set = set()
    for i in range(len(true_amps)):
        current_amp = np_true_amps[i]
        current_mid_idx = mid_indices[i]
        window_start_idx = current_mid_idx - window_radius_samples
        window_end_idx = current_mid_idx + window_radius_samples
        indices_in_window = np.where((mid_indices >= window_start_idx) & (mid_indices <= window_end_idx))[0]
        is_local_max = False
        if len(indices_in_window) > 0:
            max_amp_in_window = np.max(np_true_amps[indices_in_window])
            if np.isclose(current_amp, max_amp_in_window, atol=1e-9):
                is_local_max = True
        if is_local_max:
            if i not in added_indices_set:
                main_true_amps.append(current_amp)
                main_true_peak_indices.append(true_peak_indices[i])
                main_true_trough_indices.append(true_trough_indices[i])
                added_indices_set.add(i)
    if not main_true_amps: return metrics
    matched_main_true_amps = [];
    matched_main_pred_amps = []
    for i in range(len(main_true_amps)):
        pk_idx = main_true_peak_indices[i];
        tr_idx = main_true_trough_indices[i]
        if pk_idx >= len(pred_flat) or tr_idx >= len(pred_flat) or pk_idx < 0 or tr_idx < 0: continue
        true_main_amp_i = main_true_amps[i]
        pred_peak_val = pred_flat[pk_idx];
        pred_trough_val = pred_flat[tr_idx]
        if not np.isfinite(pred_peak_val) or not np.isfinite(pred_trough_val): continue
        pred_main_amp_i = np.abs(pred_peak_val - pred_trough_val)
        matched_main_true_amps.append(true_main_amp_i);
        matched_main_pred_amps.append(pred_main_amp_i)
    if matched_main_true_amps:
        true_arr = np.array(matched_main_true_amps);
        pred_arr = np.array(matched_main_pred_amps)
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
    if ground_truth_seg is None or predicted_seg is None or len(ground_truth_seg) == 0 or len(predicted_seg) == 0:
        return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                "Main P-T Amp Err (%)": np.nan}
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
                "Main P-T Amp Err (%)": np.nan}
    if len(true_flat) < SAVGOL_WINDOW:
        return {"Correlation": np.nan, "Mean Absolute Error (MAE)": np.nan, "RMSE": np.nan,
                "Main P-T Amp Err (%)": np.nan}
    return calculate_metrics_core(true_flat, pred_flat, verbose_clipping=True)


# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 70);
    print("Step 1/2: Loading and Generating Data");
    print("=" * 70)
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
    samples_to_remove = 1
    train_on_bed_working = [s[samples_to_remove:-samples_to_remove] for s in train_on_bed_segments_raw]
    train_under_bed_working = [s[samples_to_remove:-samples_to_remove] for s in train_under_bed_segments_raw]
    test_on_bed_working = [s[samples_to_remove:-samples_to_remove] for s in test_on_bed_segments_raw]
    test_under_bed_working = [s[samples_to_remove:-samples_to_remove] for s in test_under_bed_segments_raw]

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
    for o, u in zip(test_on_bed_working, test_under_bed_working): o_a, u_a = align_with_fixed_lag(o, u,
                                                                                                  average_lag); test_on_aligned.append(
        o_a); test_under_aligned.append(u_a)

    # --- 【修改】Step 4 & 5: Hybrid FFT + Simple N 模型 ---
    print("\n" + "=" * 70);
    print("Step 4: Training Phase Model on RAW data (for L-Block)");
    print("=" * 70)
    valid_train_indices = [i for i, seg in enumerate(train_on_aligned) if len(seg) > 0]
    frf_freqs, frf_H = estimate_frf_for_phase([train_under_aligned[i] for i in valid_train_indices],
                                              [train_on_aligned[i] for i in valid_train_indices], SAMPLING_RATE)
    if len(frf_freqs) == 0: raise ValueError("FRF phase model training failed.")
    print("FRF phase model training complete.")

    print("\n" + "=" * 70);
    print(f"Step 5: Hybrid FFT + Simple N Model Training & Prediction");
    print("=" * 70)

    # --- 5.1: Apply L-Block (Hybrid FFT) to get Intermediate Signal 'v' ---
    print("\n--- 5.1: Applying L-Block (Hybrid FFT) to get intermediate signal 'v' ---")
    print(f"(Using Gamma={FIXED_GAMMA}, Cutoff={BEST_CUTOFF_HZ} Hz)")
    # 应用 L-Block 到训练集输入
    train_intermediate_v = apply_hybrid_fft_model(train_under_aligned,
                                                  SAMPLING_RATE,
                                                  FIXED_GAMMA,
                                                  BEST_CUTOFF_HZ,
                                                  frf_freqs,
                                                  frf_H)
    print(f"Generated {len(train_intermediate_v)} intermediate training segments.")

    # --- 5.2: Train N-Block ---
    print("\n--- 5.2: Training N-Block (Non-linear Polynomial) ---")
    # 目标是 *原始未滤波* 的 on-bed 信号
    nonlinear_coeffs = train_simple_nonlinearity(train_intermediate_v,
                                                 train_on_aligned,  # <-- Target is RAW on-bed
                                                 N_POLY_TERMS,
                                                 regularization_alpha=N_ALPHA)
    if nonlinear_coeffs is None: raise ValueError("N-Block training failed.")

    # --- 5.3: Apply Full Hybrid FFT + Simple N Model (Raw Prediction) ---
    print("\n--- 5.3: Applying Full Model for Raw Prediction ---")
    # 应用 L-Block 到测试集输入
    test_intermediate_v = apply_hybrid_fft_model(test_under_aligned,
                                                 SAMPLING_RATE,
                                                 FIXED_GAMMA,
                                                 BEST_CUTOFF_HZ,
                                                 frf_freqs,
                                                 frf_H)
    # 应用 N-Block 到训练集的中间信号
    train_predicted_raw = apply_polynomial_nonlinearity(train_intermediate_v, nonlinear_coeffs)
    # 应用 N-Block 到测试集的中间信号
    test_predicted_raw = apply_polynomial_nonlinearity(test_intermediate_v, nonlinear_coeffs)
    print("Hybrid FFT + Simple N Model raw prediction complete.")

    # --- Robust Scaling ---
    print("\n" + "=" * 70);
    print(f"Step 5.5: Applying Robust Scaling");
    print("=" * 70)
    train_predicted = [];
    segment_scaling_factors = [];
    num_bad_train = 0
    # 注意: 缩放是基于 *原始未滤波* 的 'on-bed' 信号
    for i in range(len(train_predicted_raw)):
        seg_hybrid = train_predicted_raw[i];
        seg_target = train_on_aligned[i]  # <-- Target is train_on_aligned (RAW)
        if len(seg_hybrid) < 2 or len(seg_target) < 2 or not np.all(np.isfinite(seg_hybrid)) or not np.all(
            np.isfinite(seg_target)): train_predicted.append(np.array([])); num_bad_train += 1; continue
        std_input = np.std(seg_hybrid);
        std_target = np.std(seg_target);
        scaling_factor_seg = 1.0
        if std_input > 1e-6 and std_target > 1e-6: scaling_factor_seg = std_target / std_input
        scaling_factor_seg = np.clip(scaling_factor_seg, 0.01, 100.0)
        if not np.isfinite(scaling_factor_seg): scaling_factor_seg = 1.0
        scaled_seg = seg_hybrid * scaling_factor_seg
        if not np.all(np.isfinite(scaled_seg)):
            train_predicted.append(np.zeros_like(scaled_seg)); num_bad_train += 1
        else:
            train_predicted.append(scaled_seg); segment_scaling_factors.append(scaling_factor_seg)
    avg_scaling_factor = np.mean(segment_scaling_factors) if segment_scaling_factors else 1.0
    print(
        f"Calculated AVERAGE scaling factor from {len(segment_scaling_factors)} valid segments: {avg_scaling_factor:.4f}")

    test_predicted = [];
    num_bad_test = 0
    for i in range(len(test_predicted_raw)):
        seg_hybrid_test = test_predicted_raw[i]
        if len(seg_hybrid_test) < 2 or not np.all(np.isfinite(seg_hybrid_test)): test_predicted.append(
            np.array([])); num_bad_test += 1; continue
        scaled_seg_test = seg_hybrid_test * avg_scaling_factor
        if not np.all(np.isfinite(scaled_seg_test)):
            test_predicted.append(np.zeros_like(scaled_seg_test)); num_bad_test += 1
        else:
            test_predicted.append(scaled_seg_test)
    # --- End Scaling ---

    # --- Final Evaluation ---
    print("\n" + "=" * 70);
    print(f"Step 6: Final Evaluation (Hybrid FFT + Simple N)");
    print("=" * 70)
    model_name_str = f"Hybrid FFT + Simple N (G={FIXED_GAMMA}, T={N_POLY_TERMS})"  # <--- 更新模型名称

    print(f"\n--- Evaluating Training Set ({model_name_str}) vs. RAW ---")
    train_metrics_agg = calculate_and_print_metrics(train_on_aligned, train_predicted,
                                                    f"Training Set ({model_name_str}) vs. RAW", crop_len=CROP_LENGTH)

    print(f"\n--- Evaluating Test Set ({model_name_str}) vs. RAW ---")
    test_metrics_agg = calculate_and_print_metrics(test_on_aligned, test_predicted,
                                                   f"Test Set ({model_name_str}) vs. RAW", crop_len=CROP_LENGTH)

    print("\n" + "=" * 50 + f" Step 6.5: Per-Segment Metrics ({model_name_str}, Cropped)" + "=" * 50)
    all_train_metrics = [];
    all_test_metrics = []
    # Train Metrics
    for i in range(len(train_predicted)):
        gt_seg = train_on_aligned[i] if i < len(train_on_aligned) else None;
        pred_seg = train_predicted[i]
        metrics = calculate_segment_metrics(gt_seg, pred_seg, crop_len=CROP_LENGTH)
        all_train_metrics.append(metrics)
        corr_str = f"{metrics['Correlation']:.4f}" if np.isfinite(metrics['Correlation']) else "nan";
        mae_str = f"{metrics['Mean Absolute Error (MAE)']:.4f}" if np.isfinite(
            metrics['Mean Absolute Error (MAE)']) else "nan";
        rmse_str = f"{metrics['RMSE']:.4f}" if np.isfinite(metrics['RMSE']) else "nan";
        main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%" if np.isfinite(
            metrics['Main P-T Amp Err (%)']) else "nan%"
        print(
            f"  Train Segment #{i:03d}: Corr={corr_str}, MAE={mae_str}, RMSE={rmse_str}, Main P-T Amp Err={main_pt_err_str}")
    # Test Metrics
    for i in range(len(test_predicted)):
        gt_seg = test_on_aligned[i] if i < len(test_on_aligned) else None;
        pred_seg = test_predicted[i]
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

    # --- Visualization ---
    print("\n" + "=" * 70);
    print(f"Step 7: Visualizing Results for {model_name_str}");
    print("=" * 70)
    train_plot_idx = 120
    if 0 <= train_plot_idx < len(train_on_aligned) and 0 <= train_plot_idx < len(
            train_predicted) and 0 <= train_plot_idx < len(train_under_aligned):
        plot_truth = train_on_aligned[train_plot_idx];
        plot_recon = train_predicted[train_plot_idx];
        plot_atten = train_under_aligned[train_plot_idx]
        plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon)) and not np.all(plot_recon == 0)
        plot_recon_display = plot_recon if plot_recon_valid else np.full_like(plot_truth, np.nan)
        if not plot_recon_valid: print(f"\nNote: Training Segment #{train_plot_idx} is invalid.")
        plt.figure(figsize=(20, 12));
        plt.suptitle(f"Training Set Reconstruction ({model_name_str}) - Segment {train_plot_idx}", fontsize=20, y=0.98)
        plt.subplot(2, 1, 1);
        plt.plot(plot_truth, label='Original Raw On-Bed Signal (Aligned)', color='black', linewidth=2.5);
        plt.plot(plot_recon_display, label='Reconstructed Signal (NaN if invalid)', color='green', linewidth=2,
                 alpha=0.9);
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
                main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%" if np.isfinite(
                    metrics['Main P-T Amp Err (%)']) else "nan%"
                metrics_text = f"Segment Metrics (Cropped):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain P-T Amp Err: {main_pt_err_str}"
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

    test_plot_idx = 80
    if 0 <= test_plot_idx < len(test_on_aligned) and 0 <= test_plot_idx < len(
            test_predicted) and 0 <= test_plot_idx < len(test_under_aligned):
        plot_truth = test_on_aligned[test_plot_idx];
        plot_recon = test_predicted[test_plot_idx];
        plot_atten = test_under_aligned[test_plot_idx]
        plot_recon_valid = len(plot_recon) > 0 and np.all(np.isfinite(plot_recon)) and not np.all(plot_recon == 0)
        plot_recon_display = plot_recon if plot_recon_valid else np.full_like(plot_truth, np.nan)
        if not plot_recon_valid: print(f"\nNote: Test Segment #{test_plot_idx} is invalid.")
        plt.figure(figsize=(20, 12));
        plt.suptitle(f"Test Set Reconstruction ({model_name_str}) - Segment {test_plot_idx}", fontsize=20, y=0.98)
        plt.subplot(2, 1, 1);
        plt.plot(plot_truth, label='Original Raw On-Bed Signal (Aligned)', color='darkred', linewidth=2.5);
        plt.plot(plot_recon_display, label='Reconstructed Signal (NaN if invalid)', color='darkgreen', linewidth=2,
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
                main_pt_err_str = f"{metrics['Main P-T Amp Err (%)']:.2f}%" if np.isfinite(
                    metrics['Main P-T Amp Err (%)']) else "nan%"
                metrics_text = f"Segment Metrics (Cropped):\nCorr: {corr_str}\nMAE: {mae_str}\nRMSE: {rmse_str}\nMain P-T Amp Err: {main_pt_err_str}"
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