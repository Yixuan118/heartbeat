import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from PyEMD import EEMD

# =============================================================================
# 1. Configuration
# =============================================================================
SAMPLING_RATE = 100
SEGMENT_LENGTH_SEC = 10
SAMPLES_PER_SEGMENT = int(SAMPLING_RATE * SEGMENT_LENGTH_SEC)
CHANNEL_TO_ANALYZE = 0

# --- Filter Parameters ---
LOWCUT_FREQ = 4.5
HIGHCUT_FREQ = 9.5
FILTER_ORDER = 5

# --- Method Parameters ---
CHOSEN_FRF_ESTIMATOR = 'Hv_geometric_mean'
FFT_DENOISE_THRESHOLD_RATIO = 0.35

# --- File Paths ---
TRAIN_ON_BED_FILE = r"./data/raw_signal_before_2025-10-04T161532_2025-10-04T162821.npy"
TRAIN_UNDER_BED_FILE = r"./data/raw_signal_after_2025-10-04T161532_2025-10-04T162821.npy"
TEST_ON_BED_FILE = r"./data/raw_signal_before_2025-09-15T233703_2025-09-15T234233.npy"
TEST_UNDER_BED_FILE = r"./data/raw_signal_after_2025-09-15T233703_2025-09-15T234233.npy"


# =============================================================================
# 2. Helper Functions
# =============================================================================
def load_and_preprocess_single_channel(file_path, samples_to_remove, target_channel, expected_segment_len):
    """Loads and extracts a single channel's data from a .npy file."""
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    if data.ndim != 2 or data.shape[1] % expected_segment_len != 0:
        return None

    num_channels_in_file = data.shape[1] // expected_segment_len
    if num_channels_in_file <= target_channel:
        return None

    start_col = target_channel * expected_segment_len
    end_col = (target_channel + 1) * expected_segment_len
    selected_channel_segments = data[:, start_col:end_col]

    filename = file_path.split('\\')[-1]
    print(f"File {filename}: Loaded {len(selected_channel_segments)} segments of {SEGMENT_LENGTH_SEC}s each.")

    if selected_channel_segments.shape[1] > 2 * samples_to_remove:
        return selected_channel_segments[:, samples_to_remove:-samples_to_remove]
    else:
        return selected_channel_segments


def align_segments_cross_correlation(signal_ref, signal_target, max_lag_samples):
    """Aligns two signals using cross-correlation to find the optimal lag."""
    min_len = min(len(signal_ref), len(signal_target))
    signal_ref_trunc, signal_target_trunc = signal_ref[:min_len], signal_target[:min_len]

    if min_len == 0:
        return np.array([]), np.array([]), 0

    correlation = signal.correlate(signal_target_trunc, signal_ref_trunc, mode='full')
    lags = signal.correlation_lags(min_len, min_len, mode='full')
    lag = lags[np.argmax(correlation)]

    if abs(lag) > max_lag_samples:
        lag = np.sign(lag) * max_lag_samples

    if abs(lag) >= min_len:
        return np.array([]), np.array([]), lag

    if lag > 0:
        aligned_ref, aligned_target = signal_ref_trunc[:-lag], signal_target_trunc[lag:]
    elif lag < 0:
        aligned_ref, aligned_target = signal_ref_trunc[abs(lag):], signal_target_trunc[:-abs(lag)]
    else:
        aligned_ref, aligned_target = signal_ref_trunc, signal_target_trunc

    return aligned_ref, aligned_target, lag


def estimate_frf(input_segments, output_segments, fs, nfft):
    """Estimates the Frequency Response Function (FRF) using the Welch method."""
    if len(input_segments) != len(output_segments) or len(input_segments) == 0:
        raise ValueError("Input and output must have an equal number of non-empty segments.")

    segment_len = input_segments[0].shape[0]
    window = signal.windows.hann(segment_len)

    S_xx_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yy_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yx_sum = np.zeros(nfft // 2 + 1, dtype=complex)

    for i in range(len(input_segments)):
        x_win, y_win = input_segments[i] * window, output_segments[i] * window
        X_f, Y_f = fft(x_win, n=nfft), fft(y_win, n=nfft)
        X_f_half, Y_f_half = X_f[:nfft // 2 + 1], Y_f[:nfft // 2 + 1]

        S_xx_sum += np.conj(X_f_half) * X_f_half
        S_yy_sum += np.conj(Y_f_half) * Y_f_half
        S_yx_sum += Y_f_half * np.conj(X_f_half)

    n_avg = len(input_segments)
    S_xx_avg, S_yy_avg, S_yx_avg = S_xx_sum / n_avg, S_yy_sum / n_avg, S_yx_sum / n_avg

    epsilon = 1e-12 * np.max(np.abs(S_xx_avg))

    H1_freq = S_yx_avg / (S_xx_avg + epsilon)
    H2_freq = S_yy_avg / (np.conj(S_yx_avg) + epsilon)
    Hv_freq = np.sqrt(H1_freq * H2_freq)
    Hv_freq = np.nan_to_num(Hv_freq)

    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    return Hv_freq, freqs


def reconstruct_signal_from_frf(input_segments, frf_model, nfft, segment_len):
    """Reconstructs the output signal using the FRF and input signal."""
    reconstructed_segments = []
    H_full = np.concatenate((frf_model, np.conj(frf_model[-2:0:-1])))

    for x_segment in input_segments:
        y_pred_freq = H_full * fft(x_segment, n=nfft)
        y_pred_time = ifft(y_pred_freq)
        reconstructed_segments.append(np.real(y_pred_time[:segment_len]))

    return reconstructed_segments


def butter_bandpass(low, high, fs, order=5):
    """Designs a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype='band')
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    """Applies a zero-phase bandpass filter to a list of signal segments."""
    if segments is None:
        return None
    b, a = butter_bandpass(low, high, fs, order=order)
    return [signal.filtfilt(b, a, seg) for seg in segments]


def fft_denoise(signal_data, threshold_ratio):
    """Removes noise from a signal by thresholding its Fourier spectrum."""
    fft_coeffs = fft(signal_data)
    spectrum = np.abs(fft_coeffs)
    threshold = threshold_ratio * np.max(spectrum)
    fft_coeffs[spectrum < threshold] = 0
    denoised_signal = np.real(ifft(fft_coeffs))
    return denoised_signal


def apply_eemd_pipeline(segments, fft_threshold_ratio):
    """Processes a list of signal segments using an EEMD and FFT denoising pipeline."""
    processed_segments = []
    eemd = EEMD()
    eemd.trials = 50
    for i, seg in enumerate(segments):
        print(f"\rProcessing segment {i + 1}/{len(segments)} with EEMD...", end="")
        seg_centered = seg - np.mean(seg)
        IMFs = eemd.eemd(seg_centered)
        filtered_emd = seg_centered - IMFs[0]  # Subtract first IMF (high-frequency noise)
        denoised_signal = fft_denoise(filtered_emd, threshold_ratio=fft_threshold_ratio)
        processed_segments.append(denoised_signal)
    print("\nEEMD processing complete.")
    return processed_segments


def calculate_metrics(ground_truth, predicted, method_name):
    """Calculates and prints performance metrics between ground truth and predicted signals."""
    print(f"\n--- {method_name} ---")
    true_flat = np.concatenate(ground_truth)
    pred_flat = np.concatenate(predicted)

    min_len = min(len(true_flat), len(pred_flat))
    true_flat, pred_flat = true_flat[:min_len], pred_flat[:min_len]

    if np.std(true_flat) == 0 or np.std(pred_flat) == 0:
        corr = np.nan
    else:
        corr, _ = pearsonr(true_flat, pred_flat)

    mae = np.mean(np.abs(true_flat - pred_flat))
    mse = np.mean((true_flat - pred_flat) ** 2)
    mean_abs_true = np.mean(np.abs(true_flat))
    amplitude_error = (mae / mean_abs_true if mean_abs_true > 1e-9 else np.inf) * 100

    metrics = {"Correlation": corr, "MAE": mae, "MSE": mse, "Amplitude Error (%)": amplitude_error}
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics


# =============================================================================
# 3. Main Program Execution
# =============================================================================
if __name__ == "__main__":
    print("--- Final Comparison: FRF Reconstruction vs. EEMD Denoising ---")

    # --- Step 1: Load and Preprocess Real Data ---
    print("\n" + "=" * 20 + " Step 1: Loading and Preprocessing Data " + "=" * 20)
    train_on_raw = load_and_preprocess_single_channel(TRAIN_ON_BED_FILE, 1, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)
    train_under_raw = load_and_preprocess_single_channel(TRAIN_UNDER_BED_FILE, 1, CHANNEL_TO_ANALYZE,
                                                         SAMPLES_PER_SEGMENT)
    test_on_raw = load_and_preprocess_single_channel(TEST_ON_BED_FILE, 1, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)
    test_under_raw = load_and_preprocess_single_channel(TEST_UNDER_BED_FILE, 1, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)

    if any(x is None for x in [train_on_raw, train_under_raw, test_on_raw, test_under_raw]):
        print("\nError: Data loading failed. Terminating program.")
        exit()

    train_on_filtered = bandpass_filter_segments(train_on_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, FILTER_ORDER)
    train_under_filtered = bandpass_filter_segments(train_under_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE,
                                                    FILTER_ORDER)
    test_on_filtered = bandpass_filter_segments(test_on_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, FILTER_ORDER)
    test_under_filtered = bandpass_filter_segments(test_under_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE,
                                                   FILTER_ORDER)

    # --- Align and finalize training data ---
    n_train = min(len(train_on_filtered), len(train_under_filtered))
    aligned_train_under_list, aligned_train_on_list = [], []
    for i in range(n_train):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(train_under_filtered[i], train_on_filtered[i],
                                                                   int(SAMPLING_RATE * 1))
        if len(aligned_u) > 0:
            aligned_train_under_list.append(aligned_u)
            aligned_train_on_list.append(aligned_o)

    min_len_train = min(len(s) for s in aligned_train_on_list) if aligned_train_on_list else 0
    train_under_aligned = [s[:min_len_train] for s in aligned_train_under_list]
    train_on_aligned = [s[:min_len_train] for s in aligned_train_on_list]

    # --- Align and finalize test data ---
    n_test = min(len(test_on_filtered), len(test_under_filtered))
    aligned_test_under_list, aligned_test_on_list = [], []
    for i in range(n_test):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(test_under_filtered[i], test_on_filtered[i],
                                                                   int(SAMPLING_RATE * 1))
        if len(aligned_u) > 0:
            aligned_test_under_list.append(aligned_u)
            aligned_test_on_list.append(aligned_o)

    min_len_test = min(len(s) for s in aligned_test_on_list) if aligned_test_on_list else 0
    test_under_aligned = [s[:min_len_test] for s in aligned_test_under_list]
    test_on_aligned = [s[:min_len_test] for s in aligned_test_on_list]

    if not train_under_aligned or not test_on_aligned:
        print("Error: Data is empty after alignment.")
        exit()

    NFFT = int(2 ** np.ceil(np.log2(max(min_len_train, 256))))
    print("\nData preprocessing complete.")

    # --- Step 2: Execute Both Methods ---
    # Method 1: FRF Reconstruction
    print("\n" + "=" * 20 + " Executing Method 1: FRF Reconstruction... " + "=" * 20)
    frf_model, _ = estimate_frf(train_under_aligned, train_on_aligned, SAMPLING_RATE, NFFT)

    # Apply to training set
    train_reconstructed_frf_raw = reconstruct_signal_from_frf(train_under_aligned, frf_model, NFFT, min_len_train)
    coherent_gain_correction = 0.5
    train_reconstructed_frf = [seg * coherent_gain_correction for seg in train_reconstructed_frf_raw]

    # Apply to test set
    test_reconstructed_frf_raw = reconstruct_signal_from_frf(test_under_aligned, frf_model, NFFT, min_len_test)
    test_reconstructed_frf = [seg * coherent_gain_correction for seg in test_reconstructed_frf_raw]
    print("FRF reconstruction complete.")

    # Method 2: EEMD Denoising
    print("\n" + "=" * 20 + " Executing Method 2: EEMD Denoising... " + "=" * 20)
    print("\nDenoising training set...")
    train_denoised_eemd = apply_eemd_pipeline(train_on_aligned, FFT_DENOISE_THRESHOLD_RATIO)

    print("\nDenoising test set...")
    test_denoised_eemd = apply_eemd_pipeline(test_on_aligned, FFT_DENOISE_THRESHOLD_RATIO)

    # --- Step 3: Evaluate and Compare ---
    print("\n" + "=" * 25 + " Final Evaluation and Comparison " + "=" * 25)
    print("\n" + "*" * 15 + " Training Set Results " + "*" * 15)
    metrics_frf_train = calculate_metrics(train_on_aligned, train_reconstructed_frf,
                                          "Method 1: FRF Reconstruction (Train)")
    metrics_eemd_train = calculate_metrics(train_on_aligned, train_denoised_eemd, "Method 2: EEMD Denoising (Train)")

    print("\n" + "*" * 15 + " Test Set Results " + "*" * 15)
    metrics_frf_test = calculate_metrics(test_on_aligned, test_reconstructed_frf, "Method 1: FRF Reconstruction (Test)")
    metrics_eemd_test = calculate_metrics(test_on_aligned, test_denoised_eemd, "Method 2: EEMD Denoising (Test)")

    # --- Step 4: Visualize the Comparison ---
    print("\n--- Analysis complete. Generating comparison plots... ---")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Training Set Visualization
    plot_idx_train = 10
    if len(train_on_aligned) > plot_idx_train:
        time_axis_train = np.arange(len(train_on_aligned[plot_idx_train])) / SAMPLING_RATE
        plt.figure(figsize=(18, 8))
        plt.plot(time_axis_train, train_on_aligned[plot_idx_train], label='Original On-Bed Signal (Noisy)',
                 color='black', alpha=0.6)
        plt.plot(time_axis_train, train_reconstructed_frf[plot_idx_train],
                 label=f'Method 1: FRF (Corr: {metrics_frf_train["Correlation"]:.3f})', color='deepskyblue',
                 linestyle='--')
        plt.plot(time_axis_train, train_denoised_eemd[plot_idx_train],
                 label=f'Method 2: EEMD (Corr: {metrics_eemd_train["Correlation"]:.3f})', color='green', linestyle='-.',
                 linewidth=2)
        plt.title(f'Method Comparison (Training Set Sample #{plot_idx_train})', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Test Set Visualization
    plot_idx_test = 10
    if len(test_on_aligned) > plot_idx_test:
        time_axis_test = np.arange(len(test_on_aligned[plot_idx_test])) / SAMPLING_RATE
        plt.figure(figsize=(18, 8))
        plt.plot(time_axis_test, test_on_aligned[plot_idx_test], label='Original On-Bed Signal (Noisy)', color='black',
                 alpha=0.6)
        plt.plot(time_axis_test, test_reconstructed_frf[plot_idx_test],
                 label=f'Method 1: FRF (Corr: {metrics_frf_test["Correlation"]:.3f})', color='deepskyblue',
                 linestyle='--')
        plt.plot(time_axis_test, test_denoised_eemd[plot_idx_test],
                 label=f'Method 2: EEMD (Corr: {metrics_eemd_test["Correlation"]:.3f})', color='green', linestyle='-.',
                 linewidth=2)
        plt.title(f'Method Comparison (Test Set Sample #{plot_idx_test})', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
