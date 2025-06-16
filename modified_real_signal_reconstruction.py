# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal.windows import gaussian
from scipy.signal import butter, filtfilt, find_peaks
import warnings
from PyEMD import EEMD
from scipy.fft import fft, ifft
from sklearn.model_selection import train_test_split  # Import for data splitting

# 设置中文显示
warnings.filterwarnings("ignore")  # 屏蔽所有警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 参数配置 ==========
FS = 100  # 采样率 (Hz)
PAD_LENGTH = 200  # 信号填充长度
WARMUP_LENGTH = 5 * FS  # 预热数据长度 (5秒)
N_CHANNEL_RESPONSE = 1000  # 通道响应长度
WINDOW_SIZE_TROUGH_SEARCH = 30  # 谷点搜索窗口大小
MIN_PEAK_DISTANCE = 50  # 最小峰间距
MIN_PEAK_VALLEY_DIFF = 0.15  # 最小峰谷差值
PEAK_PAIRING_TIME_WINDOW_SAMPLES = 15  # 峰谷配对时间窗口（样本数）
METRICS_ACCURACY_TOLERANCE = 0.1  # 性能评估容差
TEST_SET_SIZE = 0.2  # 20% of the data will be used for testing


# ========== 信号处理工具函数 ==========
def pad_signal(signal, pad_length=PAD_LENGTH):
    """信号填充函数"""
    if len(signal) == 0:
        return np.array([])
    if len(signal) < pad_length:
        return np.pad(signal, (pad_length, pad_length), mode='edge')
    return np.concatenate((signal[pad_length - 1::-1], signal, signal[-1:-pad_length - 1:-1]))


def process_signal_with_padding(signal, process_fn, pad_length=PAD_LENGTH):
    """带填充的信号处理"""
    if len(signal) == 0:
        return np.array([])
    padded = pad_signal(signal, pad_length)
    if len(padded) == 0:
        return np.array([])
    processed = process_fn(padded)
    if len(processed) < 2 * pad_length:
        return processed[pad_length:max(pad_length, len(processed) - pad_length)]
    return processed[pad_length:-pad_length]


def highpass_filter(data, cutoff=0.5, fs=FS, order=4):
    """高通滤波器"""
    if len(data) <= order * 3: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


def extract_respiration(data, cutoff=0.5, fs=FS, order=4):
    """提取呼吸成分"""
    if len(data) <= order * 3: return np.zeros_like(data)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def wavelet_denoise(data, wavelet='db4', level=4):
    """小波去噪"""
    if len(data) < pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet)) + 1 or level <= 0: return data
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
    except ValueError:
        return data
    if not coeffs or not coeffs[-1].any(): return data
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745) if np.median(np.abs(coeffs[-1])) > 1e-9 else 1e-9
    threshold = sigma * np.sqrt(2 * np.log(len(data))) if len(data) > 1 else 0
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') if c is not None and len(c) > 0 else c for c
                                in coeffs[1:]]
    try:
        denoised_data = pywt.waverec(new_coeffs, wavelet, mode='symmetric')
    except ValueError:
        return data
    return denoised_data[:len(data)]


def fft_denoise(signal, threshold=0.05):
    """FFT去噪"""
    if len(signal) == 0: return np.array([])
    fft_signal = fft(signal)
    fft_magnitude = np.abs(fft_signal)
    max_mag = np.max(fft_magnitude) if len(fft_magnitude) > 0 else 0
    if max_mag == 0: return signal
    mask = fft_magnitude < threshold * max_mag
    fft_signal[mask] = 0
    return np.real(ifft(fft_signal))


def apply_eemd_and_denoise(signal, low_freq_delete=4):
    """EEMD去噪"""
    if len(signal) < 10: return signal
    try:
        eemd = EEMD()
        eemd.trials = 50
        IMFs = eemd.eemd(signal)
        if len(IMFs) == 0: return signal
        signal_filtered_emd = signal - IMFs[0]
        num_imfs_to_remove = min(low_freq_delete, len(IMFs) - 1)
        start_index = max(1, len(IMFs) - num_imfs_to_remove)
        for i in range(start_index, len(IMFs)):
            signal_filtered_emd -= IMFs[i]
        return signal_filtered_emd
    except Exception:
        return signal


def kalman_filter(data, process_noise=0.1, measurement_noise=0.01):
    """卡尔曼滤波"""
    n = len(data)
    if n == 0: return np.array([])
    x_hat, P = np.zeros(n), np.ones(n)
    Q, R = process_noise, measurement_noise
    x_hat[0], P[0] = data[0], 1.0
    for t in range(1, n):
        x_hat_minus = x_hat[t - 1]
        P_minus = P[t - 1] + Q
        K = P_minus / (P_minus + R) if (P_minus + R) != 0 else 0
        x_hat[t] = x_hat_minus + K * (data[t] - x_hat_minus)
        P[t] = (1 - K) * P_minus
    return x_hat


def rls_filter(R, S, filter_length=100, lambda_=0.99, delta=0.001):
    """RLS自适应滤波器"""
    n = len(R)
    S_reconstructed = np.zeros(n)
    if n < filter_length or filter_length <= 0: return S_reconstructed
    w = np.zeros(filter_length)
    P = np.eye(filter_length) / delta
    for i in range(filter_length, n):
        x = R[i - filter_length:i][::-1]
        pi = np.dot(P, x)
        gamma = 1 / (lambda_ + np.dot(x.T, pi))
        error = S[i] - np.dot(w, x)
        w += gamma * error * pi
        P = (P - gamma * np.outer(pi, pi)) / lambda_
        S_reconstructed[i] = np.dot(w, x)
    return S_reconstructed


def wiener_deconvolution(R, H, K=0.02):
    """维纳反卷积"""
    len_R = len(R)
    if len_R == 0: return np.array([])
    len_H = len(H)
    H = np.array([1.0]) if len_H == 0 else H
    H_fft = np.fft.fft(H, n=len_R)
    R_fft = np.fft.fft(R)
    H_conj = np.conj(H_fft)
    H_power_spectrum = np.abs(H_fft) ** 2
    denominator = H_power_spectrum + K
    denominator[np.abs(denominator) < 1e-10] = 1e-10
    S_fft = (H_conj / denominator) * R_fft
    return np.fft.ifft(S_fft).real


# ========== 用于公平评估的修改后函数 ==========
def smooth_and_correct_for_test(S_reconstructed, window_size_gauss=25):
    """
    MODIFIED for testing: Performs smoothing ONLY.
    Removes the amplitude correction part that used the original signal, to prevent data leakage.
    """
    len_reconstructed = len(S_reconstructed)
    if len_reconstructed == 0:
        return np.array([])

    if window_size_gauss > 1 and window_size_gauss <= len_reconstructed:
        window = gaussian(window_size_gauss, std=max(1, window_size_gauss / 6))
        pad_len = window_size_gauss // 2
        padded = np.pad(S_reconstructed, (pad_len,), mode='reflect')
        S_smoothed = np.convolve(padded, window / window.sum(), mode='valid')
        if len(S_smoothed) != len_reconstructed:
            return S_reconstructed  # Return original if something goes wrong
        return S_smoothed
    else:
        return S_reconstructed.copy()


def add_respiration_component(target_signals_filtered, source_for_respiration):
    """
    MODIFIED for testing: Adds respiration extracted from a specified source signal.
    During testing, this source should be the beddot signal, not the ground-truth BCG.
    """
    is_list_input = isinstance(target_signals_filtered, list)
    num_samples = len(target_signals_filtered)
    reconstructed_with_resp_list = []
    if num_samples != len(source_for_respiration):
        print("警告: 目标信号和呼吸源信号数量不匹配。")
        return target_signals_filtered

    for i in range(num_samples):
        target_filt = target_signals_filtered[i]
        source_sig = source_for_respiration[i]

        # Extract respiration from the provided source signal
        respiration_component = extract_respiration(source_sig)

        len_target = len(target_filt)
        len_resp = len(respiration_component)
        final_resp_comp = np.zeros(len_target)
        if len_resp > 0:
            if len_resp >= len_target:
                final_resp_comp = respiration_component[:len_target]
            else:
                final_resp_comp = np.pad(respiration_component, (0, len_target - len_resp), mode='edge')
        reconstructed_with_resp_list.append(target_filt + final_resp_comp)

    if not is_list_input:
        try:
            return np.array(reconstructed_with_resp_list)
        except ValueError:
            pass  # Fallback to list
    return reconstructed_with_resp_list


# ========== 数据加载、模型训练、重建和评估函数 ==========
def load_data():
    """加载数据，若文件缺失则生成随机数据"""
    print("加载所有数据用于训练和评估...")
    try:
        all_bcg_signals = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-06-15T113000_2025-06-15T113200.npy')
        all_beddot_signals = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-06-15T113000_2025-06-15T113200.npy')
        print(f"成功加载数据: BCG 形状 {all_bcg_signals.shape}, BedDot 形状 {all_beddot_signals.shape}")
        min_samples = min(all_bcg_signals.shape[0], all_beddot_signals.shape[0])
        if min_samples < all_bcg_signals.shape[0] or min_samples < all_beddot_signals.shape[0]:
            print(f"警告: BCG和BedDot样本数不一致，使用最小样本数: {min_samples}")
        all_bcg_signals = all_bcg_signals[:min_samples]
        all_beddot_signals = all_beddot_signals[:min_samples]
    except FileNotFoundError:
        print("警告: 数据文件未找到！生成随机数据进行演示。")
        num_samples_total = 100
        signal_len = 1000
        all_bcg_signals = np.random.rand(num_samples_total, signal_len) * 2 - 1
        all_beddot_signals = np.random.rand(num_samples_total, signal_len) * 2 - 1
        for i in range(all_bcg_signals.shape[0]):
            common_heartbeat = np.sin(np.linspace(0, 10 * np.pi, signal_len) * np.random.uniform(0.9, 1.1)) * 0.5
            respiration = np.sin(np.linspace(0, 2 * np.pi, signal_len) * np.random.uniform(0.2, 0.4)) * 0.3
            bcg_noise = np.random.normal(0, 0.1, signal_len)
            beddot_noise = np.random.normal(0, 0.15, signal_len)
            attenuation = np.random.uniform(0.5, 0.8)
            delay = np.random.randint(-3, 3)
            beddot_common = np.roll(common_heartbeat, delay) * attenuation
            all_bcg_signals[i, :] = common_heartbeat + respiration + bcg_noise
            all_beddot_signals[i, :] = beddot_common + respiration * attenuation * 0.8 + beddot_noise
        print(f"生成随机数据: BCG 形状 {all_bcg_signals.shape}, BedDot 形状 {all_beddot_signals.shape}")
    return all_bcg_signals, all_beddot_signals


def preprocess_signals(signals, process_fn, pad_length=PAD_LENGTH):
    """信号预处理"""
    processed_signals = []
    for sig in signals:
        if sig.ndim > 1:
            sig = sig.flatten()
        processed_sig = process_signal_with_padding(sig, process_fn, pad_length)
        processed_signals.append(processed_sig)
    first_len = len(processed_signals[0]) if processed_signals else 0
    if all(len(s) == first_len for s in processed_signals):
        return np.array(processed_signals)
    else:
        return processed_signals


def train_model(all_bcg_filtered, all_beddot_denoised, lambda_values=[0.95, 0.99, 0.999],
                filter_length_values=[50, 100, 200]):
    """训练RLS模型"""
    min_mae = float('inf')
    best_lambda, best_fl = lambda_values[0], filter_length_values[0]
    num_samples_actual = len(all_bcg_filtered)
    if num_samples_actual == 0: return best_lambda, best_fl
    for lambda_ in lambda_values:
        for fl in filter_length_values:
            current_maes = []
            for i in range(num_samples_actual):
                S_ref, R_in = all_bcg_filtered[i], all_beddot_denoised[i]
                if len(R_in) < fl or len(S_ref) != len(R_in): continue
                reconstructed_single_S = rls_filter(R_in, S_ref, filter_length=fl, lambda_=lambda_)
                valid_len = len(S_ref) - fl
                if valid_len > 0:
                    current_maes.append(np.mean(np.abs(S_ref[fl:] - reconstructed_single_S[fl:])))
            if not current_maes: continue
            mae = np.mean(current_maes)
            print(f"  参数: lambda_={lambda_}, fl={fl}, 平均MAE={mae:.6f}")
            if mae < min_mae:
                min_mae = mae
                best_lambda, best_fl = lambda_, fl
    print(f"\n训练完成。最佳参数对应的最小平均MAE: {min_mae:.6f}")
    return best_lambda, best_fl


def estimate_channel_response(all_bcg_filtered, all_beddot_denoised, N_resp=N_CHANNEL_RESPONSE):
    """估计通道响应"""
    H_avg = np.zeros(N_resp, dtype=complex)
    count = 0
    num_samples = len(all_bcg_filtered)
    if num_samples == 0:
        H_avg[0] = 1.0 if N_resp > 0 else None
        return H_avg
    for i in range(num_samples):
        S_bcg, R_beddot = all_bcg_filtered[i], all_beddot_denoised[i]
        if len(S_bcg) == 0 or len(R_beddot) == 0 or len(S_bcg) != len(R_beddot): continue
        fft_S_bcg = np.fft.fft(S_bcg)
        fft_R_beddot = np.fft.fft(R_beddot)
        denominator = fft_S_bcg.copy()
        zero_mask = np.abs(denominator) < 1e-9
        denominator[zero_mask] = 1e-9 * np.exp(1j * np.angle(denominator[zero_mask]))
        temp_H_fft = fft_R_beddot / denominator
        temp_H_time = np.fft.ifft(temp_H_fft)
        current_H = np.zeros(N_resp, dtype=complex)
        len_temp_H = len(temp_H_time)
        current_H[:min(len_temp_H, N_resp)] = temp_H_time[:min(len_temp_H, N_resp)]
        H_avg += current_H
        count += 1
    if count > 0:
        H_avg /= count
    else:
        H_avg[0] = 1.0 if N_resp > 0 else None
    return H_avg.real


def reconstruct_signals_combined(all_input_R_processed, all_reference_S_original_filtered, H_channel_R_eq_H_S,
                                 rls_lambda, rls_fl):
    """信号重建"""
    num_samples = len(all_input_R_processed)
    reconstructed_S_filtered_list = []
    if num_samples == 0: return []
    for i in range(num_samples):
        R_proc_single, S_ref_filt_single = all_input_R_processed[i], all_reference_S_original_filtered[i]
        if len(R_proc_single) == 0 or len(S_ref_filt_single) == 0:
            reconstructed_S_filtered_list.append(np.zeros_like(S_ref_filt_single))
            continue
        rls_output_S_est = rls_filter(R_proc_single, S_ref_filt_single, filter_length=rls_fl, lambda_=rls_lambda)
        wiener_output_S_est = wiener_deconvolution(R_proc_single, H_channel_R_eq_H_S)
        min_len = min(len(rls_output_S_est), len(wiener_output_S_est), len(S_ref_filt_single))
        if min_len == 0:
            reconstructed_S_filtered_list.append(np.zeros_like(S_ref_filt_single))
            continue
        rls_o, wiener_o = rls_output_S_est[:min_len], wiener_output_S_est[:min_len]
        S_ref_o = S_ref_filt_single[:min_len]
        combined_S_est_temp = np.zeros(min_len)
        convergence_point = min(rls_fl, min_len)
        if convergence_point > 0:
            combined_S_est_temp[:convergence_point] = wiener_o[:convergence_point]
        if min_len > convergence_point:
            alpha, beta = 0.7, 0.3
            combined_S_est_temp[convergence_point:] = alpha * rls_o[convergence_point:] + beta * wiener_o[
                                                                                                 convergence_point:]

        final_output_S_est_temp = smooth_and_correct_for_test(combined_S_est_temp)

        final_output_S_est = np.zeros_like(S_ref_filt_single)
        len_to_copy = min(len(final_output_S_est_temp), len(final_output_S_est))
        final_output_S_est[:len_to_copy] = final_output_S_est_temp[:len_to_copy]
        reconstructed_S_filtered_list.append(final_output_S_est)
    try:
        return np.array(reconstructed_S_filtered_list)
    except ValueError:
        return reconstructed_S_filtered_list


def calculate_scalar_performance_metrics(true_values, pred_values, tolerance=METRICS_ACCURACY_TOLERANCE,
                                         metric_name="标量"):
    """计算标量性能指标"""
    results = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'SMAPE': np.nan, 'Accuracy': np.nan,
               'Correlation': np.nan}
    try:
        true_values, pred_values = np.array(true_values).flatten(), np.array(pred_values).flatten()
        if len(true_values) == 0 or len(pred_values) == 0 or len(true_values) != len(pred_values): return results
        abs_error = np.abs(true_values - pred_values)
        results['MAE'] = np.mean(abs_error)
        results['RMSE'] = np.sqrt(np.mean((true_values - pred_values) ** 2))
        transformed_true, transformed_pred = log_transform(true_values), log_transform(pred_values)
        abs_transformed_true = np.abs(transformed_true)
        epsilon = 1e-9
        valid_mask_mape = abs_transformed_true > epsilon
        if np.sum(valid_mask_mape) > 0:
            results['MAPE'] = np.mean(
                np.abs(transformed_true[valid_mask_mape] - transformed_pred[valid_mask_mape]) / abs_transformed_true[
                    valid_mask_mape]) * 100
        denominator_smape = (np.abs(true_values) + np.abs(pred_values)) / 2 + epsilon
        results['SMAPE'] = np.mean(np.abs(true_values - pred_values) / denominator_smape) * 100
        if np.sum(valid_mask_mape) > 0:
            results['Accuracy'] = np.mean(
                np.abs(transformed_true[valid_mask_mape] - transformed_pred[valid_mask_mape]) <= tolerance *
                abs_transformed_true[valid_mask_mape]) * 100
        if len(true_values) > 1 and np.var(true_values) > epsilon and np.var(pred_values) > epsilon:
            results['Correlation'] = np.corrcoef(true_values, pred_values)[0, 1]
    except Exception as e:
        print(f"计算 {metric_name} 指标出错: {e}")
    return results


def log_transform(x): return np.sign(x) * np.log1p(np.abs(x))


def calculate_overall_signal_metrics(all_true_signals, all_pred_signals, signal_type="整体信号"):
    """计算整体信号指标"""
    all_mae, all_rmse, all_corrs, all_smape, all_point_percentage_errors, all_accuracy_points = [], [], [], [], [], []
    num_signals = len(all_true_signals)
    if num_signals == 0 or num_signals != len(all_pred_signals): return {}
    processed_count = 0
    for i in range(num_signals):
        true_s, pred_s = np.array(all_true_signals[i]).flatten(), np.array(all_pred_signals[i]).flatten()
        if len(true_s) == 0 or len(pred_s) == 0 or len(true_s) != len(pred_s): continue
        processed_count += 1
        all_mae.append(np.mean(np.abs(true_s - pred_s)))
        all_rmse.append(np.sqrt(np.mean((true_s - pred_s) ** 2)))
        epsilon = 1e-9
        if len(true_s) > 1 and np.var(true_s) > epsilon and np.var(pred_s) > epsilon:
            try:
                corr = np.corrcoef(true_s, pred_s)[0, 1]
                if not np.isnan(corr): all_corrs.append(corr)
            except Exception:
                pass
        denominator_smape = (np.abs(true_s) + np.abs(pred_s)) / 2 + epsilon
        all_smape.append(np.mean(np.abs(true_s - pred_s) / denominator_smape) * 100)
        transformed_true, transformed_pred = log_transform(true_s), log_transform(pred_s)
        abs_transformed_true = np.abs(transformed_true)
        valid_mask = abs_transformed_true > epsilon
        if np.sum(valid_mask) > 0:
            all_point_percentage_errors.extend(
                np.abs(transformed_true[valid_mask] - transformed_pred[valid_mask]) / abs_transformed_true[valid_mask])
            all_accuracy_points.extend(
                np.abs(transformed_true[valid_mask] - transformed_pred[valid_mask]) <= METRICS_ACCURACY_TOLERANCE *
                abs_transformed_true[valid_mask])
    print(f"基于 {processed_count} 个有效信号对计算指标。")
    return {'MAE': np.nanmean(all_mae) if all_mae else np.nan, 'RMSE': np.nanmean(all_rmse) if all_rmse else np.nan,
            'MAPE': np.nanmean(all_point_percentage_errors) * 100 if all_point_percentage_errors else np.nan,
            'SMAPE': np.nanmean(all_smape) if all_smape else np.nan,
            'Correlation': np.nanmean(all_corrs) if all_corrs else np.nan,
            'Accuracy': np.nanmean(all_accuracy_points) * 100 if all_accuracy_points else np.nan}


def find_main_peak_valley_pairs(signal, min_peak_dist=MIN_PEAK_DISTANCE, trough_search_window=WINDOW_SIZE_TROUGH_SEARCH,
                                min_p2v_diff=MIN_PEAK_VALLEY_DIFF, prominence_threshold=0.05):
    """优化波峰检测"""
    signal = np.array(signal).flatten()
    if len(signal) < min_peak_dist: return []
    peaks, _ = find_peaks(signal, distance=min_peak_dist, prominence=prominence_threshold, height=min_p2v_diff * 0.5)
    valleys, _ = find_peaks(-signal)
    peak_valley_pairs = []
    if len(peaks) == 0 or len(valleys) == 0: return []
    for peak_idx in peaks:
        max_diff_found, best_valley_candidate = -float('inf'), -1
        all_possible_valleys = valleys[
            (valleys > peak_idx - trough_search_window) & (valleys < peak_idx + trough_search_window)]
        if len(all_possible_valleys) == 0: continue
        for v_idx in all_possible_valleys:
            current_diff = signal[peak_idx] - signal[v_idx]
            if current_diff >= min_p2v_diff and current_diff > max_diff_found:
                max_diff_found, best_valley_candidate = current_diff, v_idx
        if best_valley_candidate != -1:
            if not any(abs(best_valley_candidate - v) < min_peak_dist / 2 for _, v in peak_valley_pairs):
                peak_valley_pairs.append((peak_idx, best_valley_candidate))
    return sorted(peak_valley_pairs, key=lambda x: x[0])


def pair_peak_valley_data_for_height_diff(true_signal, pred_signal, true_peak_valley_pairs, pred_peak_valley_pairs,
                                          time_window_samples=PEAK_PAIRING_TIME_WINDOW_SAMPLES):
    """峰谷对配对"""
    paired_true_diffs, paired_pred_diffs = [], []
    if not true_peak_valley_pairs or not pred_peak_valley_pairs: return np.array(paired_true_diffs), np.array(
        paired_pred_diffs)
    used_pred_indices = [False] * len(pred_peak_valley_pairs)
    for true_peak, true_valley in true_peak_valley_pairs:
        best_match_k, min_dist = -1, float('inf')
        for k, (pred_peak, pred_valley) in enumerate(pred_peak_valley_pairs):
            if not used_pred_indices[k]:
                dist = abs(true_peak - pred_peak) + abs(true_valley - pred_valley)
                if dist < min_dist and abs(true_peak - pred_peak) <= time_window_samples:
                    min_dist, best_match_k = dist, k
        if best_match_k != -1:
            pred_peak, pred_valley = pred_peak_valley_pairs[best_match_k]
            if all(0 <= idx < len(sig) for idx, sig in
                   [(true_peak, true_signal), (true_valley, true_signal), (pred_peak, pred_signal),
                    (pred_valley, pred_signal)]):
                paired_true_diffs.append(true_signal[true_peak] - true_signal[true_valley])
                paired_pred_diffs.append(pred_signal[pred_peak] - pred_signal[pred_valley])
                used_pred_indices[best_match_k] = True
    return np.array(paired_true_diffs), np.array(paired_pred_diffs)


def plot_comparison(original_bcg, original_beddot, reconstructed_bcg_with_resp, sample_idx=0, fs=FS):
    """绘制信号比较图"""
    if not (0 <= sample_idx < len(original_bcg)): return
    true_s, beddot_s, pred_s = np.array(original_bcg[sample_idx]).flatten(), np.array(
        original_beddot[sample_idx]).flatten(), np.array(reconstructed_bcg_with_resp[sample_idx]).flatten()
    min_len = min(len(true_s), len(beddot_s), len(pred_s))
    if min_len == 0: return
    true_s, beddot_s, pred_s = true_s[:min_len], beddot_s[:min_len], pred_s[:min_len]
    t = np.arange(0, min_len / fs, 1 / fs)
    plt.figure(figsize=(15, 5))
    plt.plot(t, true_s, label='原始BCG (Ground Truth)', alpha=0.8, color='blue', linewidth=1.5)
    plt.plot(t, beddot_s, label='BedDot信号 (输入)', alpha=0.7, color='green', linewidth=1.0)
    plt.plot(t, pred_s, label='重建信号 (输出)', linewidth=1.5, linestyle='--', color='red')
    plt.title(f'测试集信号比较 (样本 {sample_idx})', fontsize=14)
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度')
    plt.legend();
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout();
    plt.show()


def plot_peak_valley(signal, peak_valley_pairs, title, color='blue', fs=FS):
    """绘制峰谷对"""
    signal = np.array(signal).flatten()
    if len(signal) == 0: return
    t = np.arange(0, len(signal) / fs, 1 / fs)
    plt.plot(t, signal, label='信号', color=color, linewidth=1.0)
    plotted_peak_label, plotted_valley_label = False, False
    for peak_idx, valley_idx in peak_valley_pairs:
        if not (0 <= peak_idx < len(signal) and 0 <= valley_idx < len(signal)): continue
        peak_label, valley_label = ('J峰', '谷点') if not plotted_peak_label else ("", "")
        plt.plot(t[peak_idx], signal[peak_idx], 'o', markersize=6, color='red', label=peak_label)
        plt.plot(t[valley_idx], signal[valley_idx], 'x', markersize=6, color='purple', label=valley_label)
        plotted_peak_label = plotted_valley_label = True
        plt.vlines(t[peak_idx], signal[valley_idx], signal[peak_idx], color='gray', linestyle=':', alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label: plt.legend(by_label.values(), by_label.keys())
    plt.title(title);
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度')
    plt.grid(True, linestyle='--', alpha=0.6)


def print_metrics_dict(title, metrics_dict):
    """打印指标字典"""
    print(f"\n--- {title} ---")
    if not metrics_dict:
        print("  指标字典为空。")
        return
    for key in ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'Accuracy', 'Correlation']:
        val = metrics_dict.get(key, np.nan)
        print(f"  {key}: {val:.4f}" if not np.isnan(val) else f"  {key}: N/A")


def pair_peak_data(true_signal, pred_signal, true_peaks, pred_peaks, time_window_samples=15, fs=FS):
    """
    Pairs peaks between true and predicted signals to evaluate amplitude and IBI.
    This function is crucial for a fair feature-based comparison.
    """
    paired_true_amps, paired_pred_amps = [], []
    paired_true_ibis, paired_pred_ibis = [], []

    if len(true_peaks) < 2 or len(pred_peaks) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    used_pred_indices = [False] * len(pred_peaks)
    true_peak_to_pred_peak_map = {}

    for i, true_peak_idx_val in enumerate(true_peaks):
        best_match_k, min_dist = -1, float('inf')
        for k, pred_peak_idx_val in enumerate(pred_peaks):
            if not used_pred_indices[k]:
                dist = abs(true_peak_idx_val - pred_peak_idx_val)
                if dist < min_dist and dist <= time_window_samples:
                    min_dist, best_match_k = dist, k

        if best_match_k != -1:
            used_pred_indices[best_match_k] = True
            pred_peak_idx_val_matched = pred_peaks[best_match_k]
            true_peak_to_pred_peak_map[i] = best_match_k

            if 0 <= true_peak_idx_val < len(true_signal) and 0 <= pred_peak_idx_val_matched < len(pred_signal):
                paired_true_amps.append(true_signal[true_peak_idx_val])
                paired_pred_amps.append(pred_signal[pred_peak_idx_val_matched])

    sorted_true_indices_in_map = sorted(true_peak_to_pred_peak_map.keys())

    for i in range(len(sorted_true_indices_in_map) - 1):
        true_map_idx1 = sorted_true_indices_in_map[i]
        true_map_idx2 = sorted_true_indices_in_map[i + 1]

        pred_map_idx1 = true_peak_to_pred_peak_map[true_map_idx1]
        pred_map_idx2 = true_peak_to_pred_peak_map[true_map_idx2]

        true_loc1, true_loc2 = true_peaks[true_map_idx1], true_peaks[true_map_idx2]
        pred_loc1, pred_loc2 = pred_peaks[pred_map_idx1], pred_peaks[pred_map_idx2]

        true_ibi = (true_loc2 - true_loc1) / fs
        pred_ibi = (pred_loc2 - pred_loc1) / fs

        paired_true_ibis.append(true_ibi)
        paired_pred_ibis.append(pred_ibi)

    return np.array(paired_true_amps), np.array(paired_pred_amps), np.array(paired_true_ibis), np.array(
        paired_pred_ibis)


# ========== 主程序 (已重构和优化) ==========
def main():
    """主程序: 包含训练/测试分离逻辑和幅度校正"""
    # 1. 加载和分割数据
    all_bcg, all_beddot = load_data()
    if len(all_bcg) == 0:
        print("错误: 未能加载或生成数据，程序退出。")
        return

    indices = np.arange(len(all_bcg))
    train_indices, test_indices = train_test_split(indices, test_size=TEST_SET_SIZE, random_state=42)

    bcg_train, bcg_test = all_bcg[train_indices], all_bcg[test_indices]
    beddot_train, beddot_test = all_beddot[train_indices], all_beddot[test_indices]
    print(f"\n数据已分割: {len(bcg_train)}个训练样本, {len(bcg_test)}个测试样本。")

    # =================================================================
    # 2. 训练阶段 (仅使用训练数据)
    # =================================================================
    print("\n--- 开始训练阶段 (仅使用训练数据) ---")

    # 2.1 预处理训练数据
    print("预处理训练数据...")
    bcg_train_filtered = preprocess_signals(bcg_train, highpass_filter)
    denoise_fn = lambda x: kalman_filter(fft_denoise(wavelet_denoise(apply_eemd_and_denoise(x))))
    beddot_train_denoised = preprocess_signals(beddot_train, lambda sig: denoise_fn(highpass_filter(sig)))
    print("训练数据预处理完成。")

    # 2.2 训练模型
    print("\n在训练集上训练RLS模型...")
    best_lambda, best_fl = train_model(bcg_train_filtered, beddot_train_denoised)
    print(f"\n训练得到的最佳RLS参数: lambda={best_lambda}, filter_length={best_fl}")

    print("\n在训练集上估计平均通道响应...")
    H_avg_channel = estimate_channel_response(bcg_train_filtered, beddot_train_denoised, N_resp=N_CHANNEL_RESPONSE)
    print("模型主要部分训练完成。")

    # =================================================================
    # 2.3 计算幅度缩放因子
    # =================================================================
    print("\n计算幅度缩放因子 (基于训练集)...")
    reconstructed_train_filtered_for_scaling = reconstruct_signals_combined(
        beddot_train_denoised, bcg_train_filtered, H_avg_channel, best_lambda, best_fl
    )
    std_original_train = [np.std(s) for s in bcg_train_filtered if len(s) > 0 and np.std(s) > 1e-9]
    std_reconstructed_train = [np.std(s) for s in reconstructed_train_filtered_for_scaling if
                               len(s) > 0 and np.std(s) > 1e-9]

    if std_reconstructed_train and std_original_train:
        avg_std_original = np.mean(std_original_train)
        avg_std_reconstructed = np.mean(std_reconstructed_train)
        amplitude_scaling_factor = avg_std_original / avg_std_reconstructed
        print(f"训练集上: 原始滤波信号平均Std={avg_std_original:.4f}, 重建信号平均Std={avg_std_reconstructed:.4f}")
        print(f"计算得到的幅度缩放因子: {amplitude_scaling_factor:.4f}")
    else:
        amplitude_scaling_factor = 1.0
        print("警告: 无法计算有效的幅度缩放因子，将使用默认值 1.0。")
    print("幅度校准完成。")

    # =================================================================
    # 3. 测试阶段 (使用测试数据和已训练的模型/参数)
    # =================================================================
    print(f"\n--- 开始测试阶段 (使用{len(bcg_test)}个样本) ---")
    reconstructed_S_final_with_resp_test = []
    reconstructed_S_estimates_filtered_test = []

    for i in range(len(bcg_test)):
        # a. 创建预热数据 (避免使用训练集数据)
        if i == 0:
            warmup_beddot = beddot_test[i][:WARMUP_LENGTH][::-1]
            warmup_bcg = bcg_test[i][:WARMUP_LENGTH][::-1]  # 仅为创建同结构参考信号，实际不用于重建
        else:
            warmup_beddot = beddot_test[i - 1][-WARMUP_LENGTH:]
            warmup_bcg = bcg_test[i - 1][-WARMUP_LENGTH:]

        # b. 创建用于处理的扩展信号
        extended_beddot = np.concatenate((warmup_beddot, beddot_test[i]))
        extended_bcg_ref = np.concatenate((warmup_bcg, bcg_test[i]))

        # c. 处理单个扩展信号
        extended_beddot_denoised = denoise_fn(highpass_filter(extended_beddot))
        extended_bcg_ref_filtered = highpass_filter(extended_bcg_ref)

        # d. 重建滤波后信号
        reconstructed_extended_list = reconstruct_signals_combined(
            [extended_beddot_denoised], [extended_bcg_ref_filtered], H_avg_channel, best_lambda, best_fl
        )
        reconstructed_extended_filtered = np.array(reconstructed_extended_list)[0]
        clean_reconstructed_filtered = reconstructed_extended_filtered[WARMUP_LENGTH:]

        # e. 幅度校正与呼吸波合成
        # e.1. 应用在训练阶段计算出的幅度缩放因子
        scaled_reconstructed_filtered = clean_reconstructed_filtered * amplitude_scaling_factor

        # e.2. 从BedDot信号中独立提取呼吸波
        respiration_component = extract_respiration(extended_beddot)[WARMUP_LENGTH:]

        # e.3. 确保所有片段长度一致
        target_len = len(bcg_test[i])
        final_scaled_filtered = np.zeros(target_len)
        final_resp = np.zeros(target_len)

        len_to_copy_filt = min(target_len, len(scaled_reconstructed_filtered))
        final_scaled_filtered[:len_to_copy_filt] = scaled_reconstructed_filtered[:len_to_copy_filt]

        len_to_copy_resp = min(target_len, len(respiration_component))
        final_resp[:len_to_copy_resp] = respiration_component[:len_to_copy_resp]

        # e.4. 合成最终信号
        final_with_resp = final_scaled_filtered + final_resp

        # f. 收集用于评估的结果
        reconstructed_S_estimates_filtered_test.append(final_scaled_filtered)
        reconstructed_S_final_with_resp_test.append(final_with_resp)

    print("\n--- 所有测试样本重建完成 ---")

    # ==============================================================================
    # 4. 评估阶段 (比较测试集真值和重建结果)
    # ==============================================================================
    print("\n--- 性能评估 (在测试集上) ---")
    # 4.1 整体信号指标
    overall_final_metrics = calculate_overall_signal_metrics(bcg_test, reconstructed_S_final_with_resp_test,
                                                             signal_type="原始BCG vs 最终重建BCG (测试集)")
    print_metrics_dict("原始BCG vs. 最终重建BCG (测试集)", overall_final_metrics)

    # 4.2 基于特征的指标
    bcg_test_filtered_benchmark = preprocess_signals(bcg_test, highpass_filter)
    all_true_height_diffs, all_pred_height_diffs = [], []
    all_paired_true_peak_amplitudes, all_paired_pred_peak_amplitudes = [], []
    all_paired_true_ibis, all_paired_pred_ibis = [], []

    for i in range(len(bcg_test)):
        original_signal_filt = bcg_test_filtered_benchmark[i]
        reconstructed_signal_filt = reconstructed_S_estimates_filtered_test[i]

        if len(original_signal_filt) < MIN_PEAK_DISTANCE or len(reconstructed_signal_filt) < MIN_PEAK_DISTANCE:
            continue

        # A. 峰谷高度差
        original_pv_pairs = find_main_peak_valley_pairs(original_signal_filt)
        reconstructed_pv_pairs = find_main_peak_valley_pairs(reconstructed_signal_filt)

        if original_pv_pairs and reconstructed_pv_pairs:
            p_true_diffs, p_pred_diffs = pair_peak_valley_data_for_height_diff(
                original_signal_filt, reconstructed_signal_filt, original_pv_pairs, reconstructed_pv_pairs
            )
            all_true_height_diffs.extend(p_true_diffs)
            all_pred_height_diffs.extend(p_pred_diffs)

        # B. J峰振幅和IBI
        original_peaks = sorted([p for p, v in original_pv_pairs])
        reconstructed_peaks = sorted([p for p, v in reconstructed_pv_pairs])

        if len(original_peaks) >= 2 and len(reconstructed_peaks) >= 2:
            p_true_amps, p_pred_amps, p_true_ibis, p_pred_ibis = pair_peak_data(
                original_signal_filt, reconstructed_signal_filt, original_peaks, reconstructed_peaks
            )
            all_paired_true_peak_amplitudes.extend(p_true_amps)
            all_paired_pred_peak_amplitudes.extend(p_pred_amps)
            all_paired_true_ibis.extend(p_true_ibis)
            all_paired_pred_ibis.extend(p_pred_ibis)

    # 计算并打印所有聚合特征的指标
    print_metrics_dict("峰谷高度差 指标 (测试集滤波信号对比)",
                       calculate_scalar_performance_metrics(all_true_height_diffs, all_pred_height_diffs,
                                                            metric_name="峰谷高度差"))
    print_metrics_dict("J峰振幅 指标 (测试集滤波信号对比)",
                       calculate_scalar_performance_metrics(all_paired_true_peak_amplitudes,
                                                            all_paired_pred_peak_amplitudes,
                                                            metric_name="J峰振幅"))
    print_metrics_dict("心跳间期(IBI) 指标 (测试集滤波信号对比)",
                       calculate_scalar_performance_metrics(all_paired_true_ibis, all_paired_pred_ibis,
                                                            metric_name="心跳间期(IBI)"))

    # =================================================================
    # 5. 可视化 (在测试集上)
    # =================================================================
    if len(bcg_test) > 0:
        print("\n绘制第一个测试样本的信号比较图...")
        plot_comparison(bcg_test, beddot_test, reconstructed_S_final_with_resp_test, sample_idx=0, fs=FS)

        print("\n为前几个测试样本绘制峰谷比较图...")
        num_plots = min(3, len(bcg_test))
        for i in range(num_plots):
            original_s_f_plot = bcg_test_filtered_benchmark[i]
            reconstructed_s_f_plot = reconstructed_S_estimates_filtered_test[i]
            if len(original_s_f_plot) == 0 or len(reconstructed_s_f_plot) == 0: continue

            original_pv_p = find_main_peak_valley_pairs(original_s_f_plot)
            reconstructed_pv_p = find_main_peak_valley_pairs(reconstructed_s_f_plot)

            plt.figure(figsize=(12, 7))
            plt.subplot(2, 1, 1)
            plot_peak_valley(original_s_f_plot, original_pv_p, f'测试样本 {i} - 原始滤波BCG', fs=FS, color='blue')
            plt.subplot(2, 1, 2)
            plot_peak_valley(reconstructed_s_f_plot, reconstructed_pv_p, f'测试样本 {i} - 重建滤波BCG', color='orange',
                             fs=FS)
            plt.tight_layout(pad=3.0)
            plt.show()


# 程序入口
if __name__ == "__main__":
    main()
