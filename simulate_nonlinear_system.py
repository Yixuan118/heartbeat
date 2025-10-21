import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. Core Configuration
# =============================================================================
SAMPLING_RATE = 100
CHANNEL_TO_ANALYZE = 0
SAMPLES_PER_SEGMENT = 1000

# --- Filter & Model Parameters ---
LOWCUT_FREQ = 4.5
HIGHCUT_FREQ = 9.5
FILTER_ORDER = 5
VOLTERRA_MEMORY_DEPTH = 15

# --- Manual Post-processing Scale Factor ---
MANUAL_SCALE_FACTOR = 1

# --- Attenuation Parameters ---
CORNER_FREQ = 6.0
ATTENUATION_FACTOR = 0.5
NOISE_LEVEL = 0.1
RANDOM_SEED = 42

# --- Data File Paths ---
TRAIN_ON_BED_RAW_FILE = r"./data/vibration_analysis.npy"
TRAIN_UNDER_BED_SAVE_FILE = "train_under_bed_nonlinear_attenuation.npy"

TEST_ON_BED_RAW_FILE = r"./data/BSG.npy"
TEST_UNDER_BED_SAVE_FILE = "test_under_bed_nonlinear_attenuation.npy"


# =============================================================================
# 2. Helper Functions
# =============================================================================

def load_and_segment_signal(file_path, segment_len, samples_to_remove=1):
    if not os.path.exists(file_path): print(f"Warning: Signal file not found: {file_path}."); return []
    try:
        raw_data = np.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}."); return []
    print(f"\nLoading: {os.path.basename(file_path)}, shape: {raw_data.shape}")
    if raw_data.ndim == 1:
        num_segments = len(raw_data) // segment_len
        segmented_data = [raw_data[i * segment_len:(i + 1) * segment_len] for i in range(num_segments)]
    elif raw_data.ndim == 2:
        if raw_data.shape[1] != segment_len:
            start_col, end_col = CHANNEL_TO_ANALYZE * segment_len, (CHANNEL_TO_ANALYZE + 1) * segment_len
            if end_col > raw_data.shape[1]: print(f"Error: Channel slicing out of bounds."); return []
            raw_data = raw_data[:, start_col:end_col]
        segmented_data = [row for row in raw_data if len(row) == segment_len]
    else:
        print(f"Unsupported dimension: {raw_data.ndim}.");
        return []
    processed_segments = [seg[samples_to_remove:-samples_to_remove] for seg in segmented_data if
                          len(seg) > 2 * samples_to_remove]
    if not processed_segments: print("Warning: No valid segments found."); return []
    print(f"Processing complete: {len(processed_segments)} segments.")
    return processed_segments


def generate_under_bed_signals(on_bed_segments, fs, save_path):
    if not on_bed_segments: return []
    np.random.seed(RANDOM_SEED)
    under_bed_segments = []
    for i, on_bed_seg in enumerate(on_bed_segments):
        n = len(on_bed_seg);
        signal_fft = fft(on_bed_seg);
        freqs = fftfreq(n, 1 / fs)
        attenuation_curve = np.exp(-(np.abs(freqs) / CORNER_FREQ) * ATTENUATION_FACTOR)
        attenuated_seg = np.real(ifft(signal_fft * attenuation_curve))
        if np.random.rand() < 0.25:
            clip_threshold = np.random.uniform(1.5, 2.5) * np.std(attenuated_seg)
            attenuated_seg = np.clip(attenuated_seg, -clip_threshold, clip_threshold)
        compression_factor = np.random.uniform(1.0, 2.5)
        max_val = np.max(np.abs(attenuated_seg))
        if max_val > 0: attenuated_seg = np.tanh(attenuated_seg / max_val * compression_factor) * max_val
        signal_std = np.std(attenuated_seg);
        noise = NOISE_LEVEL * np.random.randn(len(attenuated_seg)) * signal_std
        under_bed_segments.append(attenuated_seg + noise)
        print(f"\rGenerating under-bed signals: {i + 1}/{len(on_bed_segments)}", end="")
    under_bed_np = np.array(under_bed_segments);
    np.save(save_path, under_bed_np)
    print(f"\nSaved: {save_path}, shape: {under_bed_np.shape}")
    return under_bed_segments


def align_segments_cross_correlation(signal_ref, signal_target, max_lag_samples=100):
    min_len = min(len(signal_ref), len(signal_target));
    ref, target = signal_ref[:min_len], signal_target[:min_len]
    corr = signal.correlate(target, ref, mode='full');
    lags = signal.correlation_lags(min_len, min_len, mode='full')
    lag = lags[np.argmax(corr)]
    if abs(lag) > max_lag_samples: lag = np.sign(lag) * max_lag_samples
    if lag > 0:
        ref, target = ref[:-lag], target[lag:]
    elif lag < 0:
        ref, target = ref[abs(lag):], target[:-abs(lag)]
    final_len = min(len(ref), len(target));
    return ref[:final_len], target[:final_len]


def butter_bandpass(low, high, fs, order=5):
    nyquist = 0.5 * fs;
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype='band');
    return b, a


def bandpass_filter_segments(segments, low, high, fs, order=5):
    if not segments: return []
    b, a = butter_bandpass(low, high, fs, order=order)
    return [signal.filtfilt(b, a, seg) for seg in segments]


def train_volterra_model(input_segments, output_segments, memory_depth):
    print(f"\nTraining 2nd order Volterra model with memory M={memory_depth}...")
    if not input_segments or not output_segments: print("Error: Training segments are empty."); return None
    x = np.concatenate(input_segments);
    y = np.concatenate(output_segments)
    M = memory_depth
    if len(x) < M: raise ValueError("Signal length is smaller than model memory depth.")
    num_coeffs = M + M * M;
    num_samples = len(x) - M + 1
    Phi = np.zeros((num_samples, num_coeffs))
    for n in range(num_samples):
        x_delayed = x[n + M - 1:n - 1:-1] if n > 0 else x[n + M - 1::-1]
        Phi[n, :M] = x_delayed;
        Phi[n, M:] = np.outer(x_delayed, x_delayed).flatten()
    y_target = y[M - 1:]
    print("Solving linear system for Volterra kernels...")
    kernel, _, _, _ = np.linalg.lstsq(Phi, y_target, rcond=None)
    print(f"Volterra model training complete. Learned {len(kernel)} coefficients.")
    return kernel


def apply_volterra_model(input_segments, kernel, memory_depth):
    print("Applying trained Volterra model...")
    if kernel is None: print("Cannot apply model: kernel is None."); return [np.array([]) for _ in input_segments]
    M = memory_depth;
    num_coeffs = M + M * M
    if len(kernel) != num_coeffs: raise ValueError(f"Kernel size mismatch.")
    reconstructed_segments = []
    for x in input_segments:
        if len(x) < M: reconstructed_segments.append(np.array([])); continue
        num_samples = len(x) - M + 1
        Phi_test = np.zeros((num_samples, num_coeffs))
        for n in range(num_samples):
            x_delayed = x[n + M - 1:n - 1:-1] if n > 0 else x[n + M - 1::-1]
            Phi_test[n, :M] = x_delayed;
            Phi_test[n, M:] = np.outer(x_delayed, x_delayed).flatten()
        y_pred = Phi_test @ kernel
        y_padded = np.pad(y_pred, (M - 1, 0), 'constant')
        reconstructed_segments.append(y_padded)
    return reconstructed_segments


# ========== 【MODIFIED】Metrics function with cropping capability ==========
def calculate_and_print_metrics(ground_truth, predicted, method_name, crop_len=None):
    title = f"\n--- {method_name} Final Evaluation Results ---"
    if crop_len:
        title += f" (Cropped to middle {crop_len} samples)"
    print(title)

    valid_gt = [s for s in ground_truth if len(s) > 0]
    valid_pred = [s for s in predicted if len(s) > 0]

    if not valid_gt or not valid_pred:
        print("Input list is empty, cannot evaluate.");
        return {}

    min_len = min([len(s) for s in valid_gt] + [len(s) for s in valid_pred])
    if min_len == 0:
        print("No segments with valid length to evaluate.");
        return {}

    if crop_len and min_len >= crop_len:
        start = (min_len - crop_len) // 2
        end = start + crop_len
        true_flat = np.concatenate([s[start:end] for s in valid_gt])
        pred_flat = np.concatenate([s[start:end] for s in valid_pred])
    else:
        true_flat = np.concatenate([s[:min_len] for s in valid_gt])
        pred_flat = np.concatenate([s[:min_len] for s in valid_pred])

    if true_flat.size == 0 or pred_flat.size == 0:
        print("Concatenated arrays are empty, cannot evaluate.");
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
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False
    print("=" * 70);
    print("Reconstruction with Manual Post-processing Calibration");
    print("=" * 70)

    # Step 1 & 2: Load and Generate Data
    print("\n" + "=" * 50 + " Step 1&2: Loading and Generating Data " + "=" * 50)
    train_on_bed_segments = load_and_segment_signal(file_path=TRAIN_ON_BED_RAW_FILE, segment_len=SAMPLES_PER_SEGMENT)
    if os.path.exists(TRAIN_UNDER_BED_SAVE_FILE) and train_on_bed_segments:
        train_under_bed_segments = [row for row in np.load(TRAIN_UNDER_BED_SAVE_FILE)]
    else:
        train_under_bed_segments = generate_under_bed_signals(on_bed_segments=train_on_bed_segments, fs=SAMPLING_RATE,
                                                              save_path=TRAIN_UNDER_BED_SAVE_FILE)

    test_on_bed_segments = load_and_segment_signal(file_path=TEST_ON_BED_RAW_FILE, segment_len=SAMPLES_PER_SEGMENT)
    if os.path.exists(TEST_UNDER_BED_SAVE_FILE) and test_on_bed_segments:
        test_under_bed_segments = [row for row in np.load(TEST_UNDER_BED_SAVE_FILE)]
    else:
        test_under_bed_segments = generate_under_bed_signals(on_bed_segments=test_on_bed_segments, fs=SAMPLING_RATE,
                                                             save_path=TEST_UNDER_BED_SAVE_FILE)

    NUM_TEST_SAMPLES = 100
    if len(test_on_bed_segments) > NUM_TEST_SAMPLES:
        print(f"\nTruncating test set to the first {NUM_TEST_SAMPLES} segments.")
        test_on_bed_segments = test_on_bed_segments[:NUM_TEST_SAMPLES]
        test_under_bed_segments = test_under_bed_segments[:NUM_TEST_SAMPLES]

    # Step 3: Preprocessing
    print("\n" + "=" * 50 + " Step 3: Signal Preprocessing " + "=" * 50)
    train_on_filtered = bandpass_filter_segments(train_on_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    train_under_filtered = bandpass_filter_segments(train_under_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    test_on_filtered = bandpass_filter_segments(test_on_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    test_under_filtered = bandpass_filter_segments(test_under_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    train_on_aligned, train_under_aligned = [], []
    if train_on_filtered and train_under_filtered:
        for i in range(min(len(train_on_filtered), len(train_under_filtered))):
            o, u = align_segments_cross_correlation(train_on_filtered[i], train_under_filtered[i]);
            train_on_aligned.append(o);
            train_under_aligned.append(u)
    test_on_aligned, test_under_aligned = [], []
    if test_on_filtered and test_under_filtered:
        for i in range(min(len(test_on_filtered), len(test_under_filtered))):
            o, u = align_segments_cross_correlation(test_on_filtered[i], test_under_filtered[i]);
            test_on_aligned.append(o);
            test_under_aligned.append(u)
    print(f"Preprocessing complete: {len(train_on_aligned)} training, {len(test_on_aligned)} test segments aligned.")

    # Step 4: Volterra Model Training & Raw Prediction
    print("\n" + "=" * 50 + " Step 4: Volterra Model Training & Raw Prediction " + "=" * 50)
    volterra_kernel = train_volterra_model(train_under_aligned, train_on_aligned, VOLTERRA_MEMORY_DEPTH)
    train_predicted_raw = apply_volterra_model(train_under_aligned, volterra_kernel, VOLTERRA_MEMORY_DEPTH)
    test_predicted_raw = apply_volterra_model(test_under_aligned, volterra_kernel, VOLTERRA_MEMORY_DEPTH)

    # Step 5: Manual Calibration and Final Evaluation
    print("\n" + "=" * 50 + " Step 5: Manual Calibration & Final Evaluation " + "=" * 50)
    print(f"Applying manual scale factor of {MANUAL_SCALE_FACTOR} to all predictions.")

    train_predicted_calibrated = [seg * MANUAL_SCALE_FACTOR for seg in train_predicted_raw]
    test_predicted_calibrated = [seg * MANUAL_SCALE_FACTOR for seg in test_predicted_raw]

    # --- 【MODIFIED】Calculate and print metrics for both full and cropped signals ---
    print("\n--- METRICS ON FULL SIGNAL ---")
    calculate_and_print_metrics(train_on_aligned, train_predicted_calibrated, "Training Set (Full)")
    calculate_and_print_metrics(test_on_aligned, test_predicted_calibrated, "Test Set (Full)")

    print("\n--- METRICS ON CROPPED (MIDDLE 800) SIGNAL ---")
    CROP_LENGTH = 800
    train_metrics = calculate_and_print_metrics(train_on_aligned, train_predicted_calibrated, "Training Set (Cropped)",
                                                crop_len=CROP_LENGTH)
    test_metrics = calculate_and_print_metrics(test_on_aligned, test_predicted_calibrated, "Test Set (Cropped)",
                                               crop_len=CROP_LENGTH)

    # Step 6: Visualizing Final Results
    print("\n" + "=" * 50 + " Step 6: Visualizing Final Results " + "=" * 50)
    train_plot_idx = 5
    if len(train_on_aligned) > train_plot_idx:
        plt.figure(figsize=(20, 10));
        plt.suptitle(f"Training Set - Manual Calibration (x{MANUAL_SCALE_FACTOR}) (Segment #{train_plot_idx})",
                     fontsize=18)
        plt.plot(train_on_aligned[train_plot_idx], label='Original On-Bed Signal', color='black', linewidth=2)
        plt.plot(train_predicted_calibrated[train_plot_idx], label='Reconstructed Signal (Calibrated)', color='green',
                 alpha=0.8)
        plt.plot(train_under_aligned[train_plot_idx], label='Distorted Under-Bed Signal', color='blue', alpha=0.5,
                 linestyle='--')
        plt.legend();
        plt.title('Training Set Comparison');
        plt.xlabel('Sample');
        plt.ylabel('Amplitude');
        plt.grid(True)
        plt.show()

    test_plot_idx = 5
    if len(test_on_aligned) > test_plot_idx:
        plt.figure(figsize=(20, 10));
        plt.suptitle(f"Test Set - Manual Calibration (x{MANUAL_SCALE_FACTOR}) (Segment #{test_plot_idx})", fontsize=18)
        plt.plot(test_on_aligned[test_plot_idx], label='Original On-Bed Signal', color='black', linewidth=2)
        plt.plot(test_predicted_calibrated[test_plot_idx], label='Reconstructed Signal (Calibrated)', color='red',
                 alpha=0.8)
        plt.plot(test_under_aligned[test_plot_idx], label='Distorted Under-Bed Signal', color='blue', alpha=0.5,
                 linestyle='--')
        plt.legend();
        plt.title('Test Set Comparison');
        plt.xlabel('Sample');
        plt.ylabel('Amplitude');
        plt.grid(True)
        plt.show()

    print("\n" + "=" * 70);
    print("All Processes Completed");
    print("=" * 70)
