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
SAMPLES_PER_SEGMENT = 1000  # Segment length for both training/test sets

# --- Denoising Method ---
DENOISING_METHOD = 'none'

# --- Filter Parameters ---
LOWCUT_FREQ = 0.8
HIGHCUT_FREQ = 20.0
FILTER_ORDER = 5

# --- Equalization Parameters ---
BEST_GAMMA = 0.8
BEST_CUTOFF_HZ = 15.0

# --- Exponential Attenuation Parameters (Linear Part) ---
CORNER_FREQ = 6.0  # Attenuation corner frequency (Hz)
ATTENUATION_FACTOR = 0.5  # Attenuation factor, higher value means faster high-frequency decay
NOISE_LEVEL = 0.1  # Fixed noise level
RANDOM_SEED = 42  # Seed for reproducible noise generation

# --- Data File Paths ---
TRAIN_ON_BED_RAW_FILE = r"D:\UGA\heartbeat_system(1)\vibration_analysis.npy"
TRAIN_UNDER_BED_SAVE_FILE = "train_under_bed_nonlinear_attenuation.npy"  # New filename

TEST_ON_BED_RAW_FILE = r"D:\UGA\DataDemo-main\bsg_simulation\data\BSG_train_rr_noise.npy"
TEST_UNDER_BED_SAVE_FILE = "test_under_bed_nonlinear_attenuation.npy"  # New filename


# =============================================================================
# 2. Core Helper Functions (Linear + Non-linear Attenuation Logic)
# =============================================================================

def load_and_segment_signal(file_path, segment_len, samples_to_remove=1):
    """Unified function to load and segment signals"""
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
    """
    【已修改】生成床下信号，包含固定的线性衰减和随机的非线性失真。
    """
    np.random.seed(RANDOM_SEED)

    under_bed_segments = []
    for i, on_bed_seg in enumerate(on_bed_segments):
        # --- 1. 固定的线性衰减 ---
        n = len(on_bed_seg)
        signal_fft = fft(on_bed_seg)
        freqs = fftfreq(n, 1 / fs)
        attenuation_curve = np.exp(-(np.abs(freqs) / CORNER_FREQ) * ATTENUATION_FACTOR)
        attenuated_seg = np.real(ifft(signal_fft * attenuation_curve))

        # --- 2. 随机的非线性失真 ---
        # a. 硬饱和/削波 (有一定概率发生)
        if np.random.rand() < 0.25:  # 25% 的概率发生削波
            clip_threshold = np.random.uniform(1.5, 2.5) * np.std(attenuated_seg)
            attenuated_seg = np.clip(attenuated_seg, -clip_threshold, clip_threshold)

        # b. 软饱和/压缩 (每次都施加，但强度随机)
        compression_factor = np.random.uniform(1.0, 2.5)  # 1.0接近线性, 2.5压缩效应很强
        max_val = np.max(np.abs(attenuated_seg))
        if max_val > 0:
            normalized_signal = attenuated_seg / max_val
            compressed_signal = np.tanh(normalized_signal * compression_factor)
            attenuated_seg = compressed_signal * max_val

        # --- 3. 添加固定模式的噪声 ---
        signal_std = np.std(attenuated_seg)  # 在失真后重新计算标准差
        noise = NOISE_LEVEL * np.random.randn(len(attenuated_seg)) * signal_std
        noisy_attenuated_seg = attenuated_seg + noise

        under_bed_segments.append(noisy_attenuated_seg)
        print(f"\rGenerating under-bed signals with non-linearity: {i + 1}/{len(on_bed_segments)} segments", end="")

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
    return ref[:final_len], target[:final_len]


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
    print("=" * 70)
    print("Signal Loading & Processing with NON-LINEAR Attenuation")
    print("=" * 70)

    # --- Step 1: Process Training Set ---
    print("\n" + "=" * 50 + " Step 1: Processing Training Set " + "=" * 50)
    train_on_bed_segments = load_and_segment_signal(
        file_path=TRAIN_ON_BED_RAW_FILE,
        segment_len=SAMPLES_PER_SEGMENT
    )

    if os.path.exists(TRAIN_UNDER_BED_SAVE_FILE):
        print(f"\nTraining set under-bed signal file exists, loading directly: {TRAIN_UNDER_BED_SAVE_FILE}")
        train_under_bed_segments = [row for row in np.load(TRAIN_UNDER_BED_SAVE_FILE)]
    else:
        print("\nGenerating training set under-bed signals (Linear Attenuation + Non-linear Distortion)...")
        train_under_bed_segments = generate_under_bed_signals(
            on_bed_segments=train_on_bed_segments,
            fs=SAMPLING_RATE,
            save_path=TRAIN_UNDER_BED_SAVE_FILE
        )

    # --- Step 2: Process Test Set ---
    print("\n" + "=" * 50 + " Step 2: Processing Test Set " + "=" * 50)
    test_on_bed_segments = load_and_segment_signal(
        file_path=TEST_ON_BED_RAW_FILE,
        segment_len=SAMPLES_PER_SEGMENT
    )

    if os.path.exists(TEST_UNDER_BED_SAVE_FILE):
        print(f"\nTest set under-bed signal file exists, loading directly: {TEST_UNDER_BED_SAVE_FILE}")
        test_under_bed_segments = [row for row in np.load(TEST_UNDER_BED_SAVE_FILE)]
    else:
        print("\nGenerating test set under-bed signals (Linear Attenuation + Non-linear Distortion)...")
        test_under_bed_segments = generate_under_bed_signals(
            on_bed_segments=test_on_bed_segments,
            fs=SAMPLING_RATE,
            save_path=TEST_UNDER_BED_SAVE_FILE
        )

    # --- Step 3: Verify Attenuation Effect ---
    print("\n" + "=" * 50 + " Step 3: Verifying Attenuation Effect " + "=" * 50)
    # Note: The plot will only show the LINEAR part of the attenuation.
    # The non-linear effects are not easily visualized on a frequency curve.
    train_on_sample = train_on_bed_segments[0]
    train_under_sample = train_under_bed_segments[0]
    test_on_sample = test_on_bed_segments[0]
    test_under_sample = test_under_bed_segments[0]


    def calc_psd(sig, fs, nperseg=512):
        freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
        return freqs, psd


    def get_attenuation_curve(fs, max_freq=30):
        n = int(fs * 2)
        freqs = fftfreq(n, 1 / fs)
        abs_freqs = np.abs(freqs)
        mask = (abs_freqs >= 0) & (abs_freqs <= max_freq)

        sorted_indices = np.argsort(abs_freqs[mask])
        plot_freqs = abs_freqs[mask][sorted_indices]

        attenuation_curve = np.exp(-(plot_freqs / CORNER_FREQ) * ATTENUATION_FACTOR)

        return plot_freqs, attenuation_curve


    train_freqs, train_psd_on = calc_psd(train_on_sample, SAMPLING_RATE)
    _, train_psd_under = calc_psd(train_under_sample, SAMPLING_RATE)
    test_freqs, test_psd_on = calc_psd(test_on_sample, SAMPLING_RATE)
    _, test_psd_under = calc_psd(test_under_sample, SAMPLING_RATE)
    atten_freqs, atten_curve = get_attenuation_curve(SAMPLING_RATE)

    plt.figure(figsize=(16, 14))

    # Subplot 1: Exponential Attenuation Curve (Linear Part)
    plt.subplot(3, 1, 1)
    plt.plot(atten_freqs, atten_curve, color='purple', linewidth=3)
    plt.axvline(x=CORNER_FREQ, color='orange', linestyle='--', linewidth=1.5,
                label=f'Corner Frequency ({CORNER_FREQ} Hz)')
    plt.title('Exponential Frequency Attenuation Curve (Linear Part of Simulation)', fontsize=16)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Attenuation Factor', fontsize=12)
    plt.xlim(0, 30)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True)

    # Subplot 2: Training Set PSD Comparison
    plt.subplot(3, 1, 2)
    plt.semilogy(train_freqs, train_psd_on, label='Training Set: On-Bed Signal', color='darkred', linewidth=2.5)
    plt.semilogy(train_freqs, train_psd_under, label='Training Set: Under-Bed Signal (Attenuated + Distorted)',
                 color='darkblue', linewidth=2,
                 alpha=0.8)
    plt.axvline(x=CORNER_FREQ, color='orange', linestyle='--', linewidth=1.5)
    plt.title('Training Set - Power Spectral Density Comparison', fontsize=16)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power/Frequency (dB/Hz)', fontsize=12)
    plt.xlim(0, 30)
    plt.ylim(bottom=1e-12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Subplot 3: Test Set PSD Comparison
    plt.subplot(3, 1, 3)
    plt.semilogy(test_freqs, test_psd_on, label='Test Set: On-Bed Signal', color='darkred', linewidth=2.5)
    plt.semilogy(test_freqs, test_psd_under, label='Test Set: Under-Bed Signal (Attenuated + Distorted)',
                 color='darkblue', linewidth=2, alpha=0.8)
    plt.axvline(x=CORNER_FREQ, color='orange', linestyle='--', linewidth=1.5)
    plt.title('Test Set - Power Spectral Density Comparison', fontsize=16)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power/Frequency (dB/Hz)', fontsize=12)
    plt.xlim(0, 30)
    plt.ylim(bottom=1e-12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # --- Step 4 & 5 & 6... (The rest of the script remains unchanged)
    # ... (The following code is identical to the previous version)

    # --- Step 4: Signal Preprocessing ---
    print("\n" + "=" * 50 + " Step 4: Signal Preprocessing " + "=" * 50)
    if DENOISING_METHOD == 'wavelet':
        train_under_denoised = [denoise_signal_wavelet(seg) for seg in train_under_bed_segments]
        test_under_denoised = [denoise_signal_wavelet(seg) for seg in test_under_bed_segments]
        print("Applied Wavelet Denoising.")
    elif DENOISING_METHOD == 'ceemdan':
        print("Applying CEEMDAN Denoising...")
        train_under_denoised = [denoise_signal_ceemdan(seg) for seg in train_under_bed_segments]
        test_under_denoised = [denoise_signal_ceemdan(seg) for seg in test_under_bed_segments]
        print("CEEMDAN Denoising complete.")
    else:
        train_under_denoised = train_under_bed_segments
        test_under_denoised = test_under_bed_segments
        print("No additional denoising applied.")

    train_on_filtered = bandpass_filter_segments(train_on_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    train_under_filtered = bandpass_filter_segments(train_under_denoised, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    test_on_filtered = bandpass_filter_segments(test_on_bed_segments, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)
    test_under_filtered = bandpass_filter_segments(test_under_denoised, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE)

    train_on_aligned, train_under_aligned = [], []
    n_train = min(len(train_on_filtered), len(train_under_filtered))
    for i in range(n_train):
        o, u = align_segments_cross_correlation(train_on_filtered[i], train_under_filtered[i])
        train_on_aligned.append(o)
        train_under_aligned.append(u)

    test_on_aligned, test_under_aligned = [], []
    n_test = min(len(test_on_filtered), len(test_under_filtered))
    for i in range(n_test):
        o, u = align_segments_cross_correlation(test_on_filtered[i], test_under_filtered[i])
        test_on_aligned.append(o)
        test_under_aligned.append(u)

    print(
        f"Preprocessing complete: {len(train_on_aligned)} training segments aligned, {len(test_on_aligned)} test segments aligned.")

    # Step 5: Model Training & Evaluation
    print("\n" + "=" * 50 + " Step 5: Model Training & Evaluation " + "=" * 50)
    frf_freqs, frf_H = estimate_frf_for_phase(train_under_aligned, train_on_aligned, SAMPLING_RATE)
    if len(frf_freqs) == 0:
        raise ValueError("FRF model training failed, no valid segments.")
    print("FRF phase model training complete.")

    train_under_hybrid = apply_hybrid_fft_model(train_under_aligned, SAMPLING_RATE, BEST_GAMMA, BEST_CUTOFF_HZ,
                                                frf_freqs, frf_H)
    valid_train_on = [s for s in train_on_aligned if len(s) > 0]
    valid_train_under = [s for s in train_under_hybrid if len(s) > 0]
    std_target = np.std(np.concatenate(valid_train_on))
    std_input = np.std(np.concatenate(valid_train_under))
    scaling_factor = std_target / std_input if std_input > 1e-9 else 1.0
    print(f"Global scaling factor: {scaling_factor:.4f}")

    train_predicted = [seg * scaling_factor for seg in train_under_hybrid]
    train_metrics = calculate_and_print_metrics(train_on_aligned, train_predicted, "Training Set - Hybrid FFT Model")

    test_under_hybrid = apply_hybrid_fft_model(test_under_aligned, SAMPLING_RATE, BEST_GAMMA, BEST_CUTOFF_HZ, frf_freqs,
                                               frf_H)
    test_predicted = [seg * scaling_factor for seg in test_under_hybrid]
    test_metrics = calculate_and_print_metrics(test_on_aligned, test_predicted, "Test Set (BSG) - Hybrid FFT Model")

    # Step 6: Visualize Reconstruction Results
    print("\n" + "=" * 50 + " Step 6: Visualizing Reconstruction Results " + "=" * 50)
    # Training Set Visualization
    train_plot_idx = 5
    if len(train_on_aligned) > train_plot_idx and len(train_under_aligned) > train_plot_idx and len(
            train_predicted) > train_plot_idx:
        train_time = np.arange(len(train_on_aligned[train_plot_idx])) / SAMPLING_RATE
        plt.figure(figsize=(20, 12))
        plt.suptitle(f"Training Set - Signal Reconstruction (Segment #{train_plot_idx})", fontsize=20, y=0.98)

        plt.subplot(2, 1, 1)
        plt.plot(train_time, train_on_aligned[train_plot_idx], label='Original On-Bed Signal', color='black',
                 linewidth=2.5)
        plt.plot(train_time, train_predicted[train_plot_idx], label='Reconstructed Signal', color='green', linewidth=2,
                 alpha=0.9)
        plt.plot(train_time, train_under_aligned[train_plot_idx], label='Attenuated Under-Bed Signal', color='blue',
                 linewidth=1.5,
                 alpha=0.7, linestyle='--')
        plt.title('Full Time Series', fontsize=16)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.subplot(2, 1, 2)
        zoom_start = int(len(train_time) * 0.3)
        zoom_end = zoom_start + int(5 * SAMPLING_RATE)
        zoom_time = train_time[zoom_start:zoom_end]
        plt.plot(zoom_time, train_on_aligned[train_plot_idx][zoom_start:zoom_end], label='Original On-Bed Signal',
                 color='black',
                 linewidth=2.5)
        plt.plot(zoom_time, train_predicted[train_plot_idx][zoom_start:zoom_end], label='Reconstructed Signal',
                 color='green',
                 linewidth=2, alpha=0.9)
        plt.plot(zoom_time, train_under_aligned[train_plot_idx][zoom_start:zoom_end],
                 label='Attenuated Under-Bed Signal',
                 color='blue', linewidth=1.5, alpha=0.7, linestyle='--')
        metric_text = f'Correlation: {train_metrics["Correlation"]:.4f}\nMAE: {train_metrics["Mean Absolute Error (MAE)"]:.4f}\nAmplitude Error: {train_metrics["Amplitude Error (%)"]:.2f}%'
        plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.title('Zoomed-in View (5 seconds)', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Test Set Visualization
    test_plot_idx = 5
    if len(test_on_aligned) > test_plot_idx and len(test_under_aligned) > test_plot_idx and len(
            test_predicted) > test_plot_idx:
        test_time = np.arange(len(test_on_aligned[test_plot_idx])) / SAMPLING_RATE
        plt.figure(figsize=(20, 12))
        plt.suptitle(f"Test Set - Signal Reconstruction (Segment #{test_plot_idx})", fontsize=20, y=0.98)

        plt.subplot(2, 1, 1)
        plt.plot(test_time, test_on_aligned[test_plot_idx], label='Original On-Bed Signal', color='darkred',
                 linewidth=2.5)
        plt.plot(test_time, test_predicted[test_plot_idx], label='Reconstructed Signal', color='darkgreen', linewidth=2,
                 alpha=0.9)
        plt.plot(test_time, test_under_aligned[test_plot_idx], label='Attenuated Under-Bed Signal', color='darkblue',
                 linewidth=1.5,
                 alpha=0.7, linestyle='--')
        plt.title('Full Time Series', fontsize=16)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.subplot(2, 1, 2)
        zoom_start = int(len(test_time) * 0.3)
        zoom_end = zoom_start + int(5 * SAMPLING_RATE)
        zoom_time = test_time[zoom_start:zoom_end]
        plt.plot(zoom_time, test_on_aligned[test_plot_idx][zoom_start:zoom_end], label='Original On-Bed Signal',
                 color='darkred',
                 linewidth=2.5)
        plt.plot(zoom_time, test_predicted[test_plot_idx][zoom_start:zoom_end], label='Reconstructed Signal',
                 color='darkgreen',
                 linewidth=2, alpha=0.9)
        plt.plot(zoom_time, test_under_aligned[test_plot_idx][zoom_start:zoom_end], label='Attenuated Under-Bed Signal',
                 color='darkblue', linewidth=1.5, alpha=0.7, linestyle='--')
        metric_text = f'Correlation: {test_metrics["Correlation"]:.4f}\nMAE: {test_metrics["Mean Absolute Error (MAE)"]:.4f}\nAmplitude Error: {test_metrics["Amplitude Error (%)"]:.2f}%'
        plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title('Zoomed-in View (5 seconds)', fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    print("\n" + "=" * 70)
    print("All Processes Completed: Using Non-Linear Attenuation Model")
    print("=" * 70)