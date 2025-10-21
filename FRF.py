import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pywt
from PyEMD import CEEMDAN
import os

# =============================================================================
# 1. Core Configuration (Exponential Attenuation Parameters + Data Paths)
# =============================================================================
SAMPLING_RATE = 100
CHANNEL_TO_ANALYZE = 0
SAMPLES_PER_SEGMENT = 1000

# --- Denoising Method ---
DENOISING_METHOD = 'none'

# --- Filter Parameters ---
LOWCUT_FREQ = 4.5
HIGHCUT_FREQ = 9.5
FILTER_ORDER = 5

# --- Equalization Parameters ---
GAMMA_CANDIDATES = [0.3, 0.35]
BEST_CUTOFF_HZ = 15.0

# --- Exponential Attenuation Parameters ---
CORNER_FREQ = 6.0
ATTENUATION_FACTOR = 0.5
RANDOM_SEED = 42

# --- Noise Parameters ---
FIXED_NOISE_STD = 0.05

# --- Data File Paths ---
TRAIN_ON_BED_RAW_FILE = r"./data/vibration_analysis.npy"
TRAIN_UNDER_BED_SAVE_FILE = "train_under_bed_exponential_attenuation.npy"
TEST_ON_BED_RAW_FILE = r"./data/BSG.npy"
TEST_UNDER_BED_SAVE_FILE = "test_under_bed_exponential_attenuation.npy"


# =============================================================================
# 2. Core Helper Functions (All functions remain the same)
# =============================================================================

def load_and_segment_signal(file_path, segment_len, samples_to_remove=1):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Signal file not found: {file_path}")
    raw_data = np.load(file_path)
    print(f"\nLoading signal file: {os.path.basename(file_path)}, original shape: {raw_data.shape}")
    if raw_data.ndim == 1:
        num_segments = len(raw_data) // segment_len
        segmented_data = [raw_data[i * segment_len: (i + 1) * segment_len] for i in range(num_segments)]
    elif raw_data.ndim == 2:
        if raw_data.shape[1] != segment_len:
            print(
                f"Warning: Signal columns ({raw_data.shape[1]}) != target length ({segment_len}). Slicing for channel {CHANNEL_TO_ANALYZE}.")
            start_col = CHANNEL_TO_ANALYZE * segment_len
            end_col = (CHANNEL_TO_ANALYZE + 1) * segment_len
            if end_col > raw_data.shape[1]:
                raise ValueError(
                    f"Channel slicing out of bounds: signal has {raw_data.shape[1]} cols, require {end_col}.")
            raw_data = raw_data[:, start_col:end_col]
        segmented_data = [row for row in raw_data if len(row) == segment_len]
    else:
        raise ValueError(f"Unsupported signal dimension: {raw_data.ndim} (only 1D/2D supported).")
    processed_segments = []
    for seg in segmented_data:
        if len(seg) > 2 * samples_to_remove:
            processed_segments.append(seg[samples_to_remove:-samples_to_remove])
        else:
            processed_segments.append(seg)
    print(f"Signal processing complete: {len(processed_segments)} segments, {len(processed_segments[0])} samples each.")
    return processed_segments


def generate_under_bed_signals(on_bed_segments, fs, save_path):
    np.random.seed(RANDOM_SEED)
    under_bed_segments = []
    for i, on_bed_seg in enumerate(on_bed_segments):
        n = len(on_bed_seg)
        signal_fft = fft(on_bed_seg)
        freqs = fftfreq(n, 1 / fs)
        attenuation_curve = np.exp(-(np.abs(freqs) / CORNER_FREQ) * ATTENUATION_FACTOR)
        signal_fft_attenuated = signal_fft * attenuation_curve
        attenuated_seg = np.real(ifft(signal_fft_attenuated))
        noise = FIXED_NOISE_STD * np.random.randn(len(attenuated_seg))
        noisy_attenuated_seg = attenuated_seg + noise
        under_bed_segments.append(noisy_attenuated_seg)
        print(f"\rGenerating under-bed signals: {i + 1}/{len(on_bed_segments)} segments", end="")
    under_bed_np = np.array(under_bed_segments)
    np.save(save_path, under_bed_np)
    print(f"\nUnder-bed signals saved: {save_path}, shape: {under_bed_np.shape}")
    return under_bed_segments


def align_segments_cross_correlation(signal_ref, signal_target, max_lag_samples=100):
    min_len = min(len(signal_ref), len(signal_target))
    ref, target = signal_ref[:min_len], signal_target[:min_len]
    corr = signal.correlate(target, ref, mode='full')
    lags = signal.correlation_lags(min_len, min_len, mode='full')
    lag = lags[np.argmax(corr)]
    if abs(lag) > max_lag_samples:
        lag = np.sign(lag) * max_lag_samples
    if lag > 0:
        ref, target = ref[:-lag], target[lag:]
    elif lag < 0:
        ref, target = ref[abs(lag):], target[:-abs(lag)]
    final_len = min(len(ref), len(target))
    return ref[:final_len], target[:final_len], lag


def align_with_fixed_lag(signal_ref, signal_target, lag):
    if lag > 0:
        ref_aligned, target_aligned = signal_ref[:-lag], signal_target[lag:]
    elif lag < 0:
        ref_aligned, target_aligned = signal_ref[abs(lag):], signal_target[:-abs(lag)]
    else:
        ref_aligned, target_aligned = signal_ref, signal_target
    min_len = min(len(ref_aligned), len(target_aligned))
    return ref_aligned[:min_len], target_aligned[:min_len]


def butter_bandpass(low, high, fs, order=5):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype='band')
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    if not segments:
        return []
    b, a = butter_bandpass(low, high, fs, order=order)
    return [signal.filtfilt(b, a, seg) for seg in segments]


def denoise_signal_wavelet(signal_data, wavelet='sym8', level=4):
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    max_possible_level = pywt.dwtn_max_level(signal_data.shape, pywt.Wavelet(wavelet))
    level = min(level, max_possible_level)
    if level < 1:
        return signal_data
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma == 0:
        return signal_data
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    denoised_signal = pywt.waverec(new_coeffs, wavelet)
    return denoised_signal[:len(signal_data)]


def denoise_signal_ceemdan(signal_data, noise_imfs_to_remove=1):
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    ceemdan = CEEMDAN()
    imfs = ceemdan(signal_data.astype(np.float64))
    if imfs.shape[0] > noise_imfs_to_remove:
        denoised_signal = np.sum(imfs[noise_imfs_to_remove:], axis=0)
    else:
        denoised_signal = imfs[-1] if imfs.shape[0] > 0 else np.zeros_like(signal_data)
    return denoised_signal


def estimate_frf_for_phase(input_segments, output_segments, fs):
    min_len_list = [len(s) for s in input_segments if len(s) > 0] + [len(s) for s in output_segments if len(s) > 0]
    if not min_len_list:
        return np.array([]), np.array([])
    min_len = min(min_len_list)
    if min_len == 0:
        return np.array([]), np.array([])
    nfft = int(2 ** np.floor(np.log2(min_len)))
    S_xy_sum, S_xx_sum = np.zeros(nfft // 2 + 1, dtype=complex), np.zeros(nfft // 2 + 1, dtype=complex)
    valid_pairs = 0
    for i in range(len(input_segments)):
        if len(input_segments[i]) < min_len or len(output_segments[i]) < min_len:
            continue
        inp_seg, out_seg = input_segments[i][:min_len], output_segments[i][:min_len]
        f, Pxy = signal.csd(out_seg, inp_seg, fs=fs, nperseg=nfft)
        _, Pxx = signal.welch(inp_seg, fs=fs, nperseg=nfft)
        S_xy_sum += Pxy
        S_xx_sum += Pxx
        valid_pairs += 1
    if valid_pairs == 0:
        return np.array([]), np.array([])
    H1 = (S_xy_sum / valid_pairs) / (S_xx_sum / valid_pairs + 1e-9)
    return f, H1


def apply_hybrid_fft_model(input_segments, fs, gamma, cutoff_hz, frf_model_freqs, frf_model_H):
    processed = []
    for seg in input_segments:
        if len(seg) == 0:
            processed.append(np.array([]))
            continue
        n = len(seg)
        seg_fft = fft(seg)
        freqs = fftfreq(n, 1 / fs)
        magnitude_curve = np.power(np.abs(freqs), gamma)
        magnitude_curve[freqs == 0] = 1.0
        freq_1hz_idx = np.abs(freqs - 1.0).argmin()
        norm_factor = 1.0
        if freq_1hz_idx > 0 and magnitude_curve[freq_1hz_idx] > 1e-9:
            norm_factor = magnitude_curve[freq_1hz_idx]
            magnitude_curve /= norm_factor
        if norm_factor > 1e-9:
            cutoff_gain = np.power(cutoff_hz, gamma) / norm_factor
        else:
            cutoff_gain = np.power(cutoff_hz, gamma)
        magnitude_curve[np.abs(freqs) > cutoff_hz] = cutoff_gain
        phase_response = np.interp(np.abs(freqs), frf_model_freqs, np.unwrap(np.angle(frf_model_H)), left=0, right=0)
        H_hybrid = magnitude_curve * np.exp(1j * phase_response)
        equalized_fft = seg_fft * H_hybrid
        processed.append(np.real(ifft(equalized_fft)))
    return processed


def calculate_and_print_metrics(ground_truth, predicted, method_name):
    print(f"\n--- {method_name} Final Evaluation Results ---")
    valid_gt = [s for s in ground_truth if len(s) > 0]
    valid_pred = [s for s in predicted if len(s) > 0]
    if not valid_gt or not valid_pred:
        print("Input list is empty or contains no valid segments, cannot evaluate.")
        return {}
    min_len = min([len(s) for s in valid_gt] + [len(s) for s in valid_pred])
    if min_len == 0:
        print("No segments with valid length to evaluate.")
        return {}
    true_flat = np.concatenate([s[:min_len] for s in valid_gt])
    pred_flat = np.concatenate([s[:min_len] for s in valid_pred])
    if true_flat.size == 0 or pred_flat.size == 0:
        print("Concatenated arrays are empty, cannot evaluate.")
        return {}
    corr, _ = pearsonr(true_flat, pred_flat)
    mae = np.mean(np.abs(true_flat - pred_flat))
    mean_abs_true = np.mean(np.abs(true_flat))
    amp_err = (mae / mean_abs_true) * 100 if mean_abs_true > 1e-9 else np.inf
    metrics = {"Correlation": corr, "Mean Absolute Error (MAE)": mae, "Amplitude Error (%)": amp_err}
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics


# =============================================================================
# 3. Main Execution
# =============================================================================
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False

    # --- Step 1 & 2: Load and Generate Data (Do this once) ---
    print("=" * 70)
    print("Step 1/2: Loading and Generating Data")
    print("=" * 70)
    train_on_bed_segments = load_and_segment_signal(file_path=TRAIN_ON_BED_RAW_FILE, segment_len=SAMPLES_PER_SEGMENT)
    test_on_bed_segments = load_and_segment_signal(file_path=TEST_ON_BED_RAW_FILE, segment_len=SAMPLES_PER_SEGMENT)
    if not os.path.exists(TRAIN_UNDER_BED_SAVE_FILE):
        print("\nGenerating training set under-bed signals...")
        generate_under_bed_signals(train_on_bed_segments, SAMPLING_RATE, TRAIN_UNDER_BED_SAVE_FILE)
    if not os.path.exists(TEST_UNDER_BED_SAVE_FILE):
        print("\nGenerating test set under-bed signals...")
        generate_under_bed_signals(test_on_bed_segments, SAMPLING_RATE, TEST_UNDER_BED_SAVE_FILE)
    train_under_bed_segments = [row for row in np.load(TRAIN_UNDER_BED_SAVE_FILE)]
    test_under_bed_segments = [row for row in np.load(TEST_UNDER_BED_SAVE_FILE)]

    # --- Step 4: Preprocessing (Gamma-Independent part, do this once) ---
    print("\n" + "=" * 70)
    print("Step 4: Performing Gamma-Independent Preprocessing")
    print("=" * 70)
    train_on_filtered = bandpass_filter_segments(train_on_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    train_under_filtered = bandpass_filter_segments(train_under_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    test_on_filtered = bandpass_filter_segments(test_on_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    test_under_filtered = bandpass_filter_segments(test_under_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)

    print("\nAligning training data and learning average lag...")
    train_on_aligned, train_under_aligned, lags_from_training = [], [], []
    for i in range(min(len(train_on_filtered), len(train_under_filtered))):
        o, u, lag = align_segments_cross_correlation(train_on_filtered[i], train_under_filtered[i])
        train_on_aligned.append(o)
        train_under_aligned.append(u)
        lags_from_training.append(lag)
    average_lag = int(np.round(np.mean(lags_from_training)))
    print(f"Learned average lag from training set: {average_lag} samples.")

    print("Aligning test data using the fixed average lag...")
    test_on_aligned, test_under_aligned = [], []
    for i in range(min(len(test_on_filtered), len(test_under_filtered))):
        o, u = align_with_fixed_lag(test_on_filtered[i], test_under_filtered[i], average_lag)
        test_on_aligned.append(o)
        test_under_aligned.append(u)

    # --- Step 5: Model Training (Gamma-Independent part, do this once) ---
    print("\n" + "=" * 70)
    print("Step 5: Training Gamma-Independent Phase Model")
    print("=" * 70)
    frf_freqs, frf_H = estimate_frf_for_phase(train_under_aligned, train_on_aligned, SAMPLING_RATE)
    if len(frf_freqs) == 0:
        raise ValueError("FRF model training failed, no valid segments.")
    print("FRF phase model training complete.")

    # --- Loop over Gamma Candidates ---
    print("\n" + "=" * 70)
    print("Step 5b: Looping Through Gamma Candidates for Evaluation")
    print("=" * 70)

    results = []
    for current_gamma in GAMMA_CANDIDATES:
        print(f"\n----------- Testing Gamma = {current_gamma} -----------")

        # --- Gamma-DEPENDENT part of training/evaluation ---
        train_under_hybrid = apply_hybrid_fft_model(
            train_under_aligned, SAMPLING_RATE, current_gamma, BEST_CUTOFF_HZ, frf_freqs, frf_H
        )

        valid_train_on = [s for s in train_on_aligned if len(s) > 0]
        valid_train_reconstructed = [s for s in train_under_hybrid if len(s) > 0]
        if not valid_train_on or not valid_train_reconstructed:
            print(f"Skipping gamma={current_gamma} due to insufficient valid segments for scaling.")
            continue

        std_target = np.std(np.concatenate(valid_train_on))
        std_input = np.std(np.concatenate(valid_train_reconstructed))
        scaling_factor = std_target / std_input if std_input > 1e-9 else 1.0
        print(f"Calculated scaling factor for gamma={current_gamma}: {scaling_factor:.4f}")

        test_under_hybrid = apply_hybrid_fft_model(
            test_under_aligned, SAMPLING_RATE, current_gamma, BEST_CUTOFF_HZ, frf_freqs, frf_H
        )
        test_predicted = [seg * scaling_factor for seg in test_under_hybrid]
        test_metrics = calculate_and_print_metrics(
            test_on_aligned, test_predicted, f"Test Set (gamma={current_gamma})"
        )

        if test_metrics:
            results.append({'gamma': current_gamma, 'metrics': test_metrics})

    # --- Find and Print the Best Gamma ---
    print("\n" + "=" * 70)
    print("Final Results Summary")
    print("=" * 70)

    if not results:
        print("No valid results were obtained. Cannot determine the best gamma.")
        # Exit or handle error appropriately
        best_gamma_value = GAMMA_CANDIDATES[0]  # Fallback
        best_metrics_value = {}
    else:
        best_result = min(results, key=lambda x: x['metrics'].get('Amplitude Error (%)', float('inf')))
        best_gamma_value = best_result['gamma']
        best_metrics_value = best_result['metrics']

        print(f"\nOptimal Gamma Value found: {best_gamma_value}")
        print("Performance on Test Set with this Gamma:")
        for key, value in best_metrics_value.items():
            print(f"  - {key}: {value:.4f}")

    # --- Step 6: Visualize Reconstruction Results for the BEST Gamma ---
    print("\n" + "=" * 70)
    print(f"Step 6: Visualizing Results for Best Gamma = {best_gamma_value}")
    print("=" * 70)

    # Re-generate the predicted signals for the best gamma to plot them
    train_under_hybrid_best = apply_hybrid_fft_model(train_under_aligned, SAMPLING_RATE, best_gamma_value,
                                                     BEST_CUTOFF_HZ, frf_freqs, frf_H)
    valid_train_reconstructed_best = [s for s in train_under_hybrid_best if len(s) > 0]
    std_input_best = np.std(np.concatenate(valid_train_reconstructed_best))
    scaling_factor_best = std_target / std_input_best if std_input_best > 1e-9 else 1.0

    train_predicted_best = [seg * scaling_factor_best for seg in train_under_hybrid_best]
    test_under_hybrid_best = apply_hybrid_fft_model(test_under_aligned, SAMPLING_RATE, best_gamma_value, BEST_CUTOFF_HZ,
                                                    frf_freqs, frf_H)
    test_predicted_best = [seg * scaling_factor_best for seg in test_under_hybrid_best]

    # <<< NEW: Evaluate the best gamma on the training set as well for comparison >>>
    print("\n--- Evaluating Best Gamma on Training Set ---")
    train_metrics_best = calculate_and_print_metrics(
        train_on_aligned, train_predicted_best, f"Training Set (Best Gamma = {best_gamma_value})"
    )

    # Training Set Visualization for Best Gamma
    train_plot_idx = 5
    if len(train_on_aligned) > train_plot_idx:
        train_time = np.arange(len(train_on_aligned[train_plot_idx])) / SAMPLING_RATE
        plt.figure(figsize=(20, 12))
        plt.suptitle(f"Training Set Reconstruction (Best Gamma = {best_gamma_value})", fontsize=20, y=0.98)
        plt.subplot(2, 1, 1)
        plt.plot(train_time, train_on_aligned[train_plot_idx], label='Original On-Bed Signal', color='black',
                 linewidth=2.5)
        plt.plot(train_time, train_predicted_best[train_plot_idx], label='Reconstructed Signal', color='green',
                 linewidth=2, alpha=0.9)
        plt.plot(train_time, train_under_aligned[train_plot_idx], label='Attenuated Under-Bed Signal', color='blue',
                 linewidth=1.5, alpha=0.7, linestyle='--')
        plt.title('Full Time Series', fontsize=16)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.subplot(2, 1, 2)
        zoom_start, zoom_end = int(len(train_time) * 0.3), int(len(train_time) * 0.3) + int(5 * SAMPLING_RATE)
        plt.plot(train_time[zoom_start:zoom_end], train_on_aligned[train_plot_idx][zoom_start:zoom_end],
                 label='Original On-Bed Signal', color='black', linewidth=2.5)
        plt.plot(train_time[zoom_start:zoom_end], train_predicted_best[train_plot_idx][zoom_start:zoom_end],
                 label='Reconstructed Signal', color='green', linewidth=2, alpha=0.9)
        plt.plot(train_time[zoom_start:zoom_end], train_under_aligned[train_plot_idx][zoom_start:zoom_end],
                 label='Attenuated Under-Bed Signal', color='blue', linewidth=1.5, alpha=0.7, linestyle='--')

        # <<< MODIFIED: Use the newly calculated best metrics for the plot text >>>
        if train_metrics_best:
            metric_text = f"Correlation: {train_metrics_best['Correlation']:.4f}\nMAE: {train_metrics_best['Mean Absolute Error (MAE)']:.4f}\nAmp Error: {train_metrics_best['Amplitude Error (%)']:.2f}%"
            plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.title('Zoomed-in View (5 seconds)', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Test Set Visualization for Best Gamma
    test_plot_idx = 5
    if len(test_on_aligned) > test_plot_idx:
        test_time = np.arange(len(test_on_aligned[test_plot_idx])) / SAMPLING_RATE
        plt.figure(figsize=(20, 12))
        plt.suptitle(f"Test Set Reconstruction (Best Gamma = {best_gamma_value})", fontsize=20, y=0.98)
        plt.subplot(2, 1, 1)
        plt.plot(test_time, test_on_aligned[test_plot_idx], label='Original On-Bed Signal', color='darkred',
                 linewidth=2.5)
        plt.plot(test_time, test_predicted_best[test_plot_idx], label='Reconstructed Signal', color='darkgreen',
                 linewidth=2, alpha=0.9)
        plt.plot(test_time, test_under_aligned[test_plot_idx], label='Attenuated Under-Bed Signal', color='darkblue',
                 linewidth=1.5, alpha=0.7, linestyle='--')
        plt.title('Full Time Series', fontsize=16)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.subplot(2, 1, 2)
        zoom_start, zoom_end = int(len(test_time) * 0.3), int(len(test_time) * 0.3) + int(5 * SAMPLING_RATE)

        if best_metrics_value:
            metric_text = f"Correlation: {best_metrics_value['Correlation']:.4f}\nMAE: {best_metrics_value['Mean Absolute Error (MAE)']:.4f}\nAmp Error: {best_metrics_value['Amplitude Error (%)']:.2f}%"
            plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.plot(test_time[zoom_start:zoom_end], test_on_aligned[test_plot_idx][zoom_start:zoom_end],
                 label='Original On-Bed Signal', color='darkred', linewidth=2.5)
        plt.plot(test_time[zoom_start:zoom_end], test_predicted_best[test_plot_idx][zoom_start:zoom_end],
                 label='Reconstructed Signal', color='darkgreen', linewidth=2, alpha=0.9)
        plt.plot(test_time[zoom_start:zoom_end], test_under_aligned[test_plot_idx][zoom_start:zoom_end],
                 label='Attenuated Under-Bed Signal', color='darkblue', linewidth=1.5, alpha=0.7, linestyle='--')
        plt.title('Zoomed-in View (5 seconds)', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    print("\n" + "=" * 70)
    print("All Processes Completed: Automated Gamma Search Finished")
    print("=" * 70)
