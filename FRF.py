import numpy as np
from scipy.fft import fft, ifft
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# =============================================================================
# 1. Configuration
# =============================================================================
SAMPLING_RATE = 100  # Hz
SEGMENT_LENGTH_SEC = 10  # seconds
SAMPLES_PER_SEGMENT = int(SAMPLING_RATE * SEGMENT_LENGTH_SEC)
CHANNEL_TO_ANALYZE = 0

# --- File Paths ---
TRAIN_ON_BED_FILE = r"./data/raw_signal_before_2025-10-04T161532_2025-10-04T162821.npy"
TRAIN_UNDER_BED_FILE = r"./data/raw_signal_after_2025-10-04T161532_2025-10-04T162821.npy"
TEST_ON_BED_FILE = r"./data/raw_signal_before_2025-09-15T233703_2025-09-15T234233.npy"
TEST_UNDER_BED_FILE = r"./data/raw_signal_after_2025-09-15T233703_2025-09-15T234233.npy"

# --- Bandpass Filter Parameters ---
LOWCUT_FREQ = 5  # Hz
HIGHCUT_FREQ = 9.5  # Hz
FILTER_ORDER = 5

# --- FRF Estimator Choice ---
# Options: 'H1', 'H2', 'Hv_geometric_mean'
CHOSEN_FRF_ESTIMATOR = 'Hv_geometric_mean'


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

    # Basic validation of the loaded data structure
    if data.ndim != 2 or data.shape[1] % expected_segment_len != 0:
        print(f"Warning: Data in {file_path} has an unexpected shape.")
        return None

    num_channels_in_file = data.shape[1] // expected_segment_len
    if num_channels_in_file <= target_channel:
        print(f"Warning: Target channel {target_channel} not available in {file_path}.")
        return None

    # Extract the specified channel
    start_col = target_channel * expected_segment_len
    end_col = (target_channel + 1) * expected_segment_len
    selected_channel_segments = data[:, start_col:end_col]

    print(f"Loaded {len(selected_channel_segments)} segments from {file_path.split('\\')[-1]}.")

    # Remove a few samples from each end to avoid edge effects from other processing
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

    # Cap the lag to a maximum allowed value
    if abs(lag) > max_lag_samples:
        lag = np.sign(lag) * max_lag_samples

    if abs(lag) >= min_len:
        return np.array([]), np.array([]), lag

    # Apply the lag to align the signals
    if lag > 0:
        aligned_ref, aligned_target = signal_ref_trunc[:-lag], signal_target_trunc[lag:]
    elif lag < 0:
        aligned_ref, aligned_target = signal_ref_trunc[abs(lag):], signal_target_trunc[:-abs(lag)]
    else:
        aligned_ref, aligned_target = signal_ref_trunc, signal_target_trunc

    return aligned_ref, aligned_target, lag


def estimate_frf(input_segments, output_segments, fs, nfft):
    """Estimates the Frequency Response Function (FRF) H1, H2, Hv, and coherence."""
    if len(input_segments) != len(output_segments) or len(input_segments) == 0:
        raise ValueError("Input and output must have an equal number of non-empty segments.")

    segment_len = input_segments[0].shape[0]
    window = signal.windows.hann(segment_len)

    S_xx_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yy_sum = np.zeros(nfft // 2 + 1, dtype=complex)
    S_yx_sum = np.zeros(nfft // 2 + 1, dtype=complex)

    # Calculate averaged auto- and cross-power spectral densities
    for i in range(len(input_segments)):
        x_win, y_win = input_segments[i] * window, output_segments[i] * window
        X_f, Y_f = fft(x_win, n=nfft), fft(y_win, n=nfft)

        X_f_half = X_f[:nfft // 2 + 1]
        Y_f_half = Y_f[:nfft // 2 + 1]

        S_xx_sum += np.conj(X_f_half) * X_f_half
        S_yy_sum += np.conj(Y_f_half) * Y_f_half
        S_yx_sum += Y_f_half * np.conj(X_f_half)

    n_avg = len(input_segments)
    S_xx_avg = S_xx_sum / n_avg
    S_yy_avg = S_yy_sum / n_avg
    S_yx_avg = S_yx_sum / n_avg

    # Add small epsilon to prevent division by zero
    epsilon = 1e-12

    # Calculate FRF estimators and coherence
    H1_freq = S_yx_avg / (S_xx_avg + epsilon)
    H2_freq = S_yy_avg / (np.conj(S_yx_avg) + epsilon)
    Hv_freq = np.sqrt(H1_freq * H2_freq)
    Hv_freq = np.nan_to_num(Hv_freq)  # Replace NaN/inf with 0

    coherence = (np.abs(S_yx_avg) ** 2) / (S_xx_avg * S_yy_avg + epsilon)
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)

    return H1_freq, H2_freq, Hv_freq, coherence, freqs


def reconstruct_signal_from_frf(input_segments, H_freq_estimator, nfft, segment_len):
    """Reconstructs the output signal using the FRF and input signal."""
    reconstructed_segments = []
    # Create the full frequency response from the half-spectrum
    H_full = np.concatenate((H_freq_estimator, np.conj(H_freq_estimator[-2:0:-1])))

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
    b, a = butter_bandpass(low, high, fs, order)
    return [signal.filtfilt(b, a, seg) for seg in segments]


def calculate_metrics(true_segments, pred_segments):
    """Calculates correlation, MAE, MSE, and Amplitude Error between true and predicted signals."""
    true_flat = np.concatenate(true_segments)
    pred_flat = np.concatenate(pred_segments)

    # Avoid division by zero if a signal is flat
    if np.std(true_flat) == 0 or np.std(pred_flat) == 0:
        corr = np.nan
    else:
        corr, _ = pearsonr(true_flat, pred_flat)

    mae = np.mean(np.abs(true_flat - pred_flat))
    mse = np.mean((true_flat - pred_flat) ** 2)
    mean_abs_true = np.mean(np.abs(true_flat))

    # Avoid division by zero for amplitude error
    amplitude_error = (mae / mean_abs_true) if mean_abs_true > 1e-9 else np.inf

    return {"Correlation": corr, "MAE": mae, "MSE": mse, "Amplitude Error (%)": amplitude_error * 100}


# =============================================================================
# 3. Main Execution
# =============================================================================
if __name__ == "__main__":
    print(f"--- Analysis Start (Using {SEGMENT_LENGTH_SEC}s segments) ---")

    # --- Step 1: Load Data ---
    print("\nStep 1: Loading and preprocessing data...")
    train_on_raw = load_and_preprocess_single_channel(TRAIN_ON_BED_FILE, 1, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)
    train_under_raw = load_and_preprocess_single_channel(TRAIN_UNDER_BED_FILE, 1, CHANNEL_TO_ANALYZE,
                                                         SAMPLES_PER_SEGMENT)
    test_on_raw = load_and_preprocess_single_channel(TEST_ON_BED_FILE, 1, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)
    test_under_raw = load_and_preprocess_single_channel(TEST_UNDER_BED_FILE, 1, CHANNEL_TO_ANALYZE, SAMPLES_PER_SEGMENT)

    if any(x is None for x in [train_on_raw, train_under_raw, test_on_raw, test_under_raw]):
        print("\nError: One or more data files failed to load. Terminating program.")
        exit()

    # --- Step 2: Filter Data ---
    print("\nStep 2: Applying bandpass filter to all signals...")
    train_on_filtered = bandpass_filter_segments(train_on_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, FILTER_ORDER)
    train_under_filtered = bandpass_filter_segments(train_under_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE,
                                                    FILTER_ORDER)
    test_on_filtered = bandpass_filter_segments(test_on_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, FILTER_ORDER)
    test_under_filtered = bandpass_filter_segments(test_under_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE,
                                                   FILTER_ORDER)

    # --- Step 3: Align Data Segments ---
    print("\nStep 3: Aligning filtered signals using cross-correlation...")
    n_train_segments = min(len(train_on_filtered), len(train_under_filtered))
    n_test_segments = min(len(test_on_filtered), len(test_under_filtered))

    # Align training data
    aligned_train_under, aligned_train_on = [], []
    for i in range(n_train_segments):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(train_under_filtered[i], train_on_filtered[i],
                                                                   max_lag_samples=int(SAMPLING_RATE * 1))
        if len(aligned_u) > 0:
            aligned_train_under.append(aligned_u)
            aligned_train_on.append(aligned_o)

    # Align testing data
    aligned_test_under, aligned_test_on = [], []
    for i in range(n_test_segments):
        aligned_u, aligned_o, _ = align_segments_cross_correlation(test_under_filtered[i], test_on_filtered[i],
                                                                   max_lag_samples=int(SAMPLING_RATE * 1))
        if len(aligned_u) > 0:
            aligned_test_under.append(aligned_u)
            aligned_test_on.append(aligned_o)

    if not aligned_train_under or not aligned_test_on:
        print("Error: No data remains after alignment. Exiting.")
        exit()

    # --- Step 4: Truncate segments to a uniform length after alignment ---
    min_len_train = min(len(s) for s in aligned_train_under)
    train_under_final = [s[:min_len_train] for s in aligned_train_under]
    train_on_final = [s[:min_len_train] for s in aligned_train_on]

    min_len_test = min(len(s) for s in aligned_test_under)
    test_under_final = [s[:min_len_test] for s in aligned_test_under]
    test_on_final = [s[:min_len_test] for s in aligned_test_on]

    print(f"\nData prepared: Using {len(train_on_final)} training segments and {len(test_on_final)} test segments.")

    # Determine NFFT for FFT calculations (power of 2 greater than segment length)
    NFFT = int(2 ** np.ceil(np.log2(max(min_len_train, 256))))

    # --- Step 5: Estimate FRF and Evaluate Model ---
    print("\nStep 5: Modeling and evaluating the FRF...")
    print("Estimating transfer function from training data...")
    H1, H2, Hv, coherence, freqs = estimate_frf(train_under_final, train_on_final, SAMPLING_RATE, NFFT)

    estimators = {'H1': H1, 'H2': H2, 'Hv_geometric_mean': Hv}
    transfer_function = estimators[CHOSEN_FRF_ESTIMATOR]

    # Evaluate on Training Set
    train_reconstructed = reconstruct_signal_from_frf(train_under_final, transfer_function, NFFT, min_len_train)
    metrics_train = calculate_metrics(train_on_final, train_reconstructed)
    print("\n--- Training Set Metrics (FRF Model) ---")
    for key, value in metrics_train.items():
        print(f"{key}: {value:.4f}")

    # Evaluate on Test Set
    test_reconstructed = reconstruct_signal_from_frf(test_under_final, transfer_function, NFFT, min_len_test)
    metrics_test = calculate_metrics(test_on_final, test_reconstructed)
    print("\n--- Test Set Metrics (FRF Model) ---")
    for key, value in metrics_test.items():
        print(f"{key}: {value:.4f}")

    # --- Step 6: Visualization ---
    print("\n--- Analysis complete. Generating plots... ---")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Visualize Training Set Result
    plot_idx_train = 30  # Select a sample index to plot
    if len(train_on_final) > plot_idx_train:
        time_axis_train = np.arange(len(train_on_final[plot_idx_train])) / SAMPLING_RATE
        plt.figure(figsize=(15, 7))
        plt.plot(time_axis_train, train_on_final[plot_idx_train], label='Ground Truth (On-Bed)', color='black',
                 linewidth=2)
        plt.plot(time_axis_train, train_reconstructed[plot_idx_train],
                 label=f'FRF Reconstruction (Corr: {metrics_train["Correlation"]:.3f})',
                 color='dodgerblue', linestyle='--')
        plt.title(f'Model Reconstruction on Training Sample #{plot_idx_train}', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Visualize Test Set Result
    plot_idx_test = 30  # Select a sample index to plot
    if len(test_on_final) > plot_idx_test:
        time_axis_test = np.arange(len(test_on_final[plot_idx_test])) / SAMPLING_RATE
        plt.figure(figsize=(15, 7))
        plt.plot(time_axis_test, test_on_final[plot_idx_test], label='Ground Truth (On-Bed)', color='black',
                 linewidth=2)
        plt.plot(time_axis_test, test_reconstructed[plot_idx_test],
                 label=f'FRF Reconstruction (Corr: {metrics_test["Correlation"]:.3f})',
                 color='dodgerblue', linestyle='--')
        plt.title(f'Model Reconstruction on Test Sample #{plot_idx_test}', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

