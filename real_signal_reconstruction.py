# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal.windows import gaussian
from scipy.signal import butter, filtfilt, find_peaks
import warnings
from PyEMD import EEMD
from scipy.fft import fft, ifft

warnings.filterwarnings("ignore") # 屏蔽所有警告
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# ========== 参数配置 ==========
FS = 100; PAD_LENGTH = 200; N_CHANNEL_RESPONSE = 1000; WINDOW_SIZE_TROUGH_SEARCH = 30
MIN_PEAK_DISTANCE = 50; MIN_PEAK_VALLEY_DIFF = 0.15; PEAK_PAIRING_TIME_WINDOW_SAMPLES = 15
METRICS_ACCURACY_TOLERANCE = 0.1

# ========== 信号处理工具函数 ==========
def pad_signal(signal, pad_length=PAD_LENGTH):
    """信号填充函数"""
    if len(signal) == 0: return np.array([])
    if len(signal) < pad_length: return np.pad(signal, (pad_length, pad_length), mode='edge')
    return np.concatenate((signal[pad_length - 1::-1], signal, signal[-1:-pad_length - 1:-1]))

def process_signal_with_padding(signal, process_fn, pad_length=PAD_LENGTH):
    """带填充的信号处理"""
    if len(signal) == 0: return np.array([])
    padded = pad_signal(signal, pad_length)
    if len(padded) == 0: return np.array([])
    processed = process_fn(padded)
    if len(processed) < 2 * pad_length:
        print(f"警告: 处理后的信号长度 {len(processed)} 小于 2*pad_length ({2*pad_length})。可能无法正确移除填充。")
        return processed[pad_length:max(pad_length, len(processed)-pad_length)]
    return processed[pad_length:-pad_length]

def highpass_filter(data, cutoff=0.3, fs=FS, order=4):
    """高通滤波器"""
    if len(data) <= order * 3: print(f"警告: 数据长度 {len(data)} 过短，无法应用高通滤波器。"); return data
    nyq = 0.5 * fs; normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def extract_respiration(data, cutoff=0.5, fs=FS, order=4):
    """提取呼吸成分"""
    if len(data) <= order * 3: print(f"警告: 数据长度 {len(data)} 过短，无法应用低通滤波器提取呼吸。"); return np.zeros_like(data)
    nyq = 0.5 * fs; normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def wavelet_denoise(data, wavelet='db4', level=3):
    """小波去噪"""
    if len(data) < pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet)) + 1 or level <= 0: print(f"警告: 数据长度 {len(data)} 过短或level设置不当，无法进行小波去噪。"); return data
    try: coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
    except ValueError as e: print(f"警告: 小波分解时出错 (数据长度: {len(data)}, level: {level}): {e}"); return data
    if not coeffs or not coeffs[-1].any(): return data
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745) if np.median(np.abs(coeffs[-1])) > 1e-9 else 1e-9
    threshold = sigma * np.sqrt(2 * np.log(len(data))) if len(data) > 1 else 0
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]: new_coeffs.append(pywt.threshold(c, threshold, mode='soft') if c is not None and len(c)>0 else c)
    try: denoised_data = pywt.waverec(new_coeffs, wavelet, mode='symmetric')
    except ValueError as e: print(f"警告: 小波重构时出错: {e}"); return data
    return denoised_data[:len(data)] # 确保输出长度与输入一致

def fft_denoise(signal, threshold=0.01):
    """FFT去噪"""
    if len(signal) == 0: return np.array([])
    fft_signal = fft(signal); fft_magnitude = np.abs(fft_signal)
    max_mag = np.max(fft_magnitude) if len(fft_magnitude) > 0 else 0
    if max_mag == 0: return signal
    mask = fft_magnitude < threshold * max_mag
    fft_signal[mask] = 0
    return np.real(ifft(fft_signal))

def apply_eemd_and_denoise(signal, low_freq_delete=4):
    """EEMD去噪"""
    if len(signal) < 10: print("警告: 信号太短，无法应用EEMD。"); return signal
    try:
        eemd = EEMD(); eemd.trials = 50; IMFs = eemd.eemd(signal)
        if len(IMFs) == 0: return signal
        signal_filtered_emd = signal - IMFs[0] # 去除最高频
        num_imfs_to_remove = min(low_freq_delete, len(IMFs) - 1)
        start_index_for_low_freq_removal = max(1, len(IMFs) - num_imfs_to_remove)
        for i in range(start_index_for_low_freq_removal, len(IMFs)): signal_filtered_emd -= IMFs[i]
        return signal_filtered_emd
    except Exception as e: print(f"EEMD处理时发生错误: {e}"); return signal

def kalman_filter(data, process_noise=1, measurement_noise=0.1):
    """卡尔曼滤波"""
    n = len(data);
    if n == 0: return np.array([])
    x_hat = np.zeros(n); P = np.ones(n); Q = process_noise; R = measurement_noise
    x_hat[0] = data[0]; P[0] = 1.0
    for t in range(1, n):
        x_hat_minus = x_hat[t - 1]; P_minus = P[t - 1] + Q
        K = P_minus / (P_minus + R) if (P_minus + R) != 0 else 0
        x_hat[t] = x_hat_minus + K * (data[t] - x_hat_minus)
        P[t] = (1 - K) * P_minus
    return x_hat

def nlms_adaptive_filter(R, S, filter_length=100, mu=0.01):
    """NLMS自适应滤波"""
    n = len(R); S_reconstructed = np.zeros(n)
    if n < filter_length or filter_length <= 0: print(f"警告: 数据长度 {n} 或滤波器长度 {filter_length} 不足，无法应用NLMS。"); return S_reconstructed
    h = np.zeros(filter_length)
    for i in range(filter_length, n):
        R_window = R[i - filter_length:i][::-1]
        S_reconstructed[i] = np.dot(h, R_window)
        error = S[i] - S_reconstructed[i]
        norm_sq = np.dot(R_window, R_window) + 1e-10
        h += (mu / norm_sq) * error * R_window
    return S_reconstructed

def wiener_deconvolution(R, H, K=0.001):
    """维纳反卷积"""
    len_R = len(R);
    if len_R == 0: return np.array([])
    len_H = len(H); H = np.array([1.0]) if len_H == 0 else H
    H_fft = np.fft.fft(H, n=len_R); R_fft = np.fft.fft(R)
    H_conj = np.conj(H_fft); H_power_spectrum = np.abs(H_fft)**2
    denominator = H_power_spectrum + K
    denominator[np.abs(denominator) < 1e-10] = 1e-10
    S_fft = (H_conj / denominator) * R_fft
    return np.fft.ifft(S_fft).real


def smooth_and_correct(S_reconstructed, S_original, window_size_gauss=0.1, segment_size=100):  # 默认 window_size_gauss 改为 5
    """信号平滑与校正 (优化：使用标准差缩放替代RMS校正)"""
    len_reconstructed = len(S_reconstructed);
    len_original = len(S_original)
    if len_reconstructed == 0: return np.array([])

    # 1. 高斯平滑 (如果需要)
    if window_size_gauss > 1 and window_size_gauss <= len_reconstructed:
        # 确保窗口大小为奇数
        if window_size_gauss % 2 == 0:
            window_size_gauss += 1
            print(f"警告: 高斯窗口大小调整为奇数: {window_size_gauss}")

        # 检查窗口大小是否仍然有效
        if window_size_gauss > len_reconstructed:
            print(f"警告: 调整后的高斯窗口大小 {window_size_gauss} 仍然大于信号长度 {len_reconstructed}，跳过平滑。")
            S_smoothed = S_reconstructed.copy()
        else:
            try:
                window = gaussian(window_size_gauss, std=max(1, window_size_gauss / 6))  # 确保 std >= 1
                pad_len = window_size_gauss // 2
                # 使用反射填充以减少边界效应
                padded = np.pad(S_reconstructed, (pad_len,), mode='reflect')
                S_smoothed = np.convolve(padded, window / window.sum(), mode='valid')
                # 确保输出长度与输入一致
                if len(S_smoothed) != len_reconstructed:
                    # 这通常发生在边缘情况下，尝试截断或填充
                    if len(S_smoothed) > len_reconstructed:
                        S_smoothed = S_smoothed[:len_reconstructed]
                    else:
                        # 如果输出更短，可能是卷积实现问题或非常小的输入
                        # 填充回原始长度可能不是最佳选择，这里选择保持原样并警告
                        print(f"警告: 平滑后信号长度 {len(S_smoothed)} 短于原始长度 {len_reconstructed}")
                        # 填充回去可能引入不连续，这里返回未平滑的
                        S_smoothed = S_reconstructed.copy()
            except Exception as e:
                print(f"警告: 高斯平滑时发生错误: {e}. 跳过平滑。")
                S_smoothed = S_reconstructed.copy()  # 出错则不平滑
    else:
        # 如果窗口大小无效或不需要平滑
        S_smoothed = S_reconstructed.copy()

    # 如果没有原始信号用于参考，则只返回平滑结果
    if len_original == 0 or len(S_original) != len(S_smoothed):
        print("警告: 无有效参考信号或长度不匹配，无法进行幅度校正和偏差校正。")
        return S_smoothed

    # 2. 基于标准差的幅度校正 (替换原来的RMS校正)
    std_original = np.std(S_original)
    std_smoothed = np.std(S_smoothed)
    epsilon = 1e-9

    S_scaled = S_smoothed  # 初始化
    if std_smoothed > epsilon:
        scale_factor = std_original / std_smoothed
        S_scaled = S_smoothed * scale_factor
        # print(f"调试: 原始Std={std_original:.4f}, 平滑Std={std_smoothed:.4f}, 缩放因子={scale_factor:.4f}") # 可选的调试信息
    else:
        # print("调试: 平滑后信号标准差接近零，跳过缩放。")
        pass  # 如果标准差为零，则不缩放

    # 3. 初始段偏差校正 (在幅度校正之后进行)
    # 使用与原始信号相同长度的段进行比较
    len_bias_corr = min(100, len(S_original), len(S_scaled))
    if len_bias_corr > 0:
        bias = np.mean(S_original[:len_bias_corr]) - np.mean(S_scaled[:len_bias_corr])
        S_corrected = S_scaled + bias
        # print(f"调试: 偏差校正值={bias:.4f}") # 可选的调试信息
    else:
        S_corrected = S_scaled  # 无法计算偏差

    return S_corrected


# ========== 数据加载与预处理 ==========
def load_data():
    """加载数据 - 使用所有数据进行训练和评估"""
    print("加载所有数据用于训练和评估...")
    try:
        all_bcg_signals = np.load(r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_mean_compressed_1_2025-06-03T083040_2025-06-03T083105.npy')
        all_beddot_signals = np.load(r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_mean_compressed_0.5_2025-06-03T091627_2025-06-03T091655.npy')
        print(f"成功加载数据: BCG 形状 {all_bcg_signals.shape}, BedDot 形状 {all_beddot_signals.shape}")
        min_samples = min(all_bcg_signals.shape[0], all_beddot_signals.shape[0])
        if min_samples < all_bcg_signals.shape[0] or min_samples < all_beddot_signals.shape[0]: print(f"警告: BCG和BedDot样本数不一致，将使用共同的最小样本数: {min_samples}")
        all_bcg_signals = all_bcg_signals[:min_samples]
        all_beddot_signals = all_beddot_signals[:min_samples]
    except FileNotFoundError:
        print("警告: 数据文件未找到！将生成随机数据进行演示。")
        num_samples_total = 100; signal_len = 1000
        all_bcg_signals = np.random.rand(num_samples_total, signal_len) * 2 - 1
        all_beddot_signals = np.random.rand(num_samples_total, signal_len) * 2 - 1
        for i in range(all_bcg_signals.shape[0]):
            common_heartbeat = np.sin(np.linspace(0, 10 * np.pi, signal_len) * np.random.uniform(0.9, 1.1)) * 0.5
            respiration = np.sin(np.linspace(0, 2 * np.pi, signal_len) * np.random.uniform(0.2, 0.4)) * 0.3
            bcg_noise = np.random.normal(0, 0.1, signal_len); beddot_noise = np.random.normal(0, 0.15, signal_len)
            attenuation = np.random.uniform(0.5, 0.8); delay = np.random.randint(-3, 3)
            beddot_common = np.roll(common_heartbeat, delay) * attenuation
            all_bcg_signals[i, :] = common_heartbeat + respiration + bcg_noise
            all_beddot_signals[i, :] = beddot_common + respiration * attenuation * 0.8 + beddot_noise
        print(f"生成随机数据: BCG 形状 {all_bcg_signals.shape}, BedDot 形状 {all_beddot_signals.shape}")
    return all_bcg_signals, all_beddot_signals

def preprocess_signals(signals, process_fn, pad_length=PAD_LENGTH):
    """信号预处理"""
    processed_signals = []
    for sig in signals:
        if sig.ndim > 1: print("警告: 输入信号是多维的，将尝试使用第一维。"); sig = sig.flatten()
        processed_sig = process_signal_with_padding(sig, process_fn, pad_length)
        processed_signals.append(processed_sig)
    first_len = len(processed_signals[0]) if processed_signals else 0
    if all(len(s) == first_len for s in processed_signals): return np.array(processed_signals)
    else: print("警告: 预处理后的信号长度不一致，将返回列表。"); return processed_signals

# ========== 模型训练 ==========
def train_model(all_bcg_filtered, all_beddot_denoised, mu_values=[0.01, 0.05], filter_length_values=[50, 100]):
    """训练NLMS模型 - 使用所有数据"""
    min_mae = float('inf'); best_mu, best_fl = mu_values[0], filter_length_values[0]
    print(f"\n在所有数据上训练NLMS模型，mu_values={mu_values}, filter_length_values={filter_length_values}...")
    num_samples_actual = len(all_bcg_filtered)
    if num_samples_actual == 0: print("警告: 没有有效的训练数据。"); return best_mu, best_fl
    for mu in mu_values:
        for fl in filter_length_values:
            current_maes = []
            for i in range(num_samples_actual):
                S_ref = all_bcg_filtered[i]; R_in = all_beddot_denoised[i]
                if len(R_in) < fl or len(S_ref) != len(R_in): continue # 跳过无效样本
                reconstructed_single_S = nlms_adaptive_filter(R_in, S_ref, fl, mu)
                valid_len = len(S_ref) - fl
                if valid_len > 0: current_maes.append(np.mean(np.abs(S_ref[fl:] - reconstructed_single_S[fl:])))
            if not current_maes: print(f"  参数: mu={mu}, fl={fl}, 无有效样本计算MAE。"); continue
            mae = np.mean(current_maes)
            print(f"  参数: mu={mu}, fl={fl}, 平均MAE={mae:.6f}")
            if mae < min_mae: min_mae = mae; best_mu = mu; best_fl = fl
    print(f"训练完成。训练期间找到的最佳参数对应的最小平均MAE: {min_mae:.6f}")
    return best_mu, best_fl

def estimate_channel_response(all_bcg_filtered, all_beddot_denoised, N_resp=N_CHANNEL_RESPONSE):
    """估计通道响应 - 使用所有数据"""
    H_avg = np.zeros(N_resp); count = 0; num_samples = len(all_bcg_filtered)
    print("\n使用所有数据估计平均通道响应 (H 使得 BedDot_processed ≈ H * BCG_filtered)...")
    if num_samples == 0: print("警告: 没有有效数据用于估计通道响应。"); H_avg[0] = 1.0 if N_resp > 0 else None; return H_avg
    for i in range(num_samples):
        S_bcg = all_bcg_filtered[i]; R_beddot = all_beddot_denoised[i]
        if len(S_bcg) == 0 or len(R_beddot) == 0 or len(S_bcg) != len(R_beddot): continue
        fft_S_bcg = np.fft.fft(S_bcg); fft_R_beddot = np.fft.fft(R_beddot)
        denominator = fft_S_bcg.copy(); zero_mask = np.abs(denominator) < 1e-9
        denominator[zero_mask] = 1e-9 * np.exp(1j * np.angle(denominator[zero_mask]))
        temp_H_fft = fft_R_beddot / denominator
        temp_H_time = np.fft.ifft(temp_H_fft).real
        current_H = np.zeros(N_resp); len_temp_H = len(temp_H_time)
        current_H[:min(len_temp_H, N_resp)] = temp_H_time[:min(len_temp_H, N_resp)]
        H_avg += current_H; count += 1
    if count > 0: H_avg /= count; print(f"平均通道响应估计完成 (基于 {count} 个有效样本)。")
    else: print("警告: 未能从任何有效样本估计H_avg，使用简单脉冲响应。"); H_avg[0] = 1.0 if N_resp > 0 else None
    return H_avg

# ========== 信号重建 ==========
def reconstruct_signals_combined(all_input_R_processed, all_reference_S_original_filtered, H_channel_R_eq_H_S, nlms_mu, nlms_fl):
    """信号重建 - 对所有数据进行"""
    num_samples = len(all_input_R_processed); reconstructed_S_filtered_list = []
    print("\n对所有信号进行重建 (结合NLMS和维纳滤波)...")
    if num_samples == 0: print("警告: 没有信号可供重建。"); return []
    for i in range(num_samples):
        R_proc_single = all_input_R_processed[i]; S_ref_filt_single = all_reference_S_original_filtered[i]
        if len(R_proc_single) == 0 or len(S_ref_filt_single) == 0:
            print(f"警告: 样本 {i} 输入信号为空，无法重建。"); reconstructed_S_filtered_list.append(np.zeros_like(S_ref_filt_single)); continue
        nlms_output_S_est = nlms_adaptive_filter(R_proc_single, S_ref_filt_single, nlms_fl, nlms_mu)
        wiener_output_S_est = wiener_deconvolution(R_proc_single, H_channel_R_eq_H_S)
        min_len = min(len(nlms_output_S_est), len(wiener_output_S_est), len(S_ref_filt_single))
        if min_len == 0: print(f"警告: 样本 {i} 处理后信号长度为0，无法组合。"); reconstructed_S_filtered_list.append(np.zeros_like(S_ref_filt_single)); continue
        nlms_o = nlms_output_S_est[:min_len]; wiener_o = wiener_output_S_est[:min_len]; S_ref_o = S_ref_filt_single[:min_len]
        combined_S_est_temp = np.zeros(min_len); convergence_point = min(nlms_fl, min_len)
        if convergence_point > 0: combined_S_est_temp[:convergence_point] = wiener_o[:convergence_point]
        if min_len > convergence_point:
             alpha = 0.5; beta = 0.5 # 权重可调
             combined_S_est_temp[convergence_point:] = alpha * nlms_o[convergence_point:] + beta * wiener_o[convergence_point:]
        final_output_S_est_temp = smooth_and_correct(combined_S_est_temp, S_ref_o)
        final_output_S_est = np.zeros_like(S_ref_filt_single)
        len_to_copy = min(len(final_output_S_est_temp), len(final_output_S_est))
        final_output_S_est[:len_to_copy] = final_output_S_est_temp[:len_to_copy]
        reconstructed_S_filtered_list.append(final_output_S_est)
    print("信号重建完成。")
    try: return np.array(reconstructed_S_filtered_list)
    except ValueError: print("警告: 重建后的滤波信号长度不一致，将返回列表。"); return reconstructed_S_filtered_list

def add_respiration_component(target_signals_filtered, reference_bcg_original):
    """添加呼吸成分到目标信号"""
    is_list_input = isinstance(target_signals_filtered, list); num_samples = len(target_signals_filtered)
    reconstructed_with_resp_list = []
    if num_samples == 0: print("警告: 没有信号可添加呼吸成分。"); return []
    if num_samples != len(reference_bcg_original): print("警告: 目标信号和参考信号数量不匹配，无法添加呼吸成分。"); return target_signals_filtered
    print("\n向重建信号中添加呼吸成分...")
    for i in range(num_samples):
        target_filt = target_signals_filtered[i]; ref_orig = reference_bcg_original[i]
        if len(ref_orig) == 0: print(f"警告: 样本 {i} 的原始参考BCG为空，无法提取呼吸。"); reconstructed_with_resp_list.append(target_filt.copy()); continue
        if len(target_filt) == 0: print(f"警告: 样本 {i} 的目标滤波信号为空。"); reconstructed_with_resp_list.append(np.zeros_like(ref_orig)); continue
        respiration_component = extract_respiration(ref_orig)
        len_target = len(target_filt); len_resp = len(respiration_component); final_resp_comp = np.zeros(len_target)
        if len_resp > 0:
            if len_resp >= len_target: final_resp_comp = respiration_component[:len_target]
            else: final_resp_comp = np.pad(respiration_component, (0, len_target - len_resp), mode='edge')
        reconstructed_with_resp_list.append(target_filt + final_resp_comp)
    print("呼吸成分添加完成。")
    if not is_list_input:
        try: return np.array(reconstructed_with_resp_list)
        except ValueError: print("警告: 添加呼吸后信号长度不一致，将返回列表。")
    return reconstructed_with_resp_list

# ========== 性能评估函数 ==========
def log_transform(x): return np.sign(x) * np.log1p(np.abs(x)) # 使用 log1p 更精确

def calculate_scalar_performance_metrics(true_values, pred_values, tolerance=METRICS_ACCURACY_TOLERANCE, metric_name="标量"):
    """计算标量性能指标"""
    results = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'SMAPE': np.nan, 'Accuracy': np.nan, 'Correlation': np.nan}
    try:
        true_values = np.array(true_values).flatten(); pred_values = np.array(pred_values).flatten()
        if len(true_values) == 0 or len(pred_values) == 0: print(f"警告: 计算 {metric_name} 指标时输入数据为空。"); return results
        if len(true_values) != len(pred_values):
            print(f"警告: 计算 {metric_name} 指标时真值和预测值长度不匹配 ({len(true_values)} vs {len(pred_values)})。将尝试截断到最短长度。")
            min_len = min(len(true_values), len(pred_values)); true_values = true_values[:min_len]; pred_values = pred_values[:min_len]
            if min_len == 0: return results
        abs_error = np.abs(true_values - pred_values); results['MAE'] = np.mean(abs_error)
        results['RMSE'] = np.sqrt(np.mean((true_values - pred_values) ** 2))
        transformed_true = log_transform(true_values); transformed_pred = log_transform(pred_values)
        abs_transformed_true = np.abs(transformed_true); epsilon = 1e-9
        valid_mask_mape = abs_transformed_true > epsilon
        if np.sum(valid_mask_mape) > 0:
            mape_errors = np.abs(transformed_true[valid_mask_mape] - transformed_pred[valid_mask_mape]) / abs_transformed_true[valid_mask_mape]
            results['MAPE'] = np.mean(mape_errors) * 100
        else: results['MAPE'] = np.nan
        denominator_smape = (np.abs(true_values) + np.abs(pred_values)) / 2 + epsilon
        smape_errors = np.abs(true_values - pred_values) / denominator_smape; results['SMAPE'] = np.mean(smape_errors) * 100
        if np.sum(valid_mask_mape) > 0:
             accurate_points = np.abs(transformed_true[valid_mask_mape] - transformed_pred[valid_mask_mape]) <= tolerance * abs_transformed_true[valid_mask_mape]
             results['Accuracy'] = np.mean(accurate_points) * 100
        elif len(transformed_true) > 0: accurate_points = np.abs(transformed_pred) <= tolerance * epsilon; results['Accuracy'] = np.mean(accurate_points) * 100
        else: results['Accuracy'] = np.nan
        if len(true_values) > 1:
             var_true = np.var(true_values); var_pred = np.var(pred_values)
             if var_true > epsilon and var_pred > epsilon:
                 try:
                     corr_matrix = np.corrcoef(true_values, pred_values); correlation = corr_matrix[0, 1]
                     results['Correlation'] = correlation if not np.isnan(correlation) else (1.0 if np.allclose(true_values, pred_values) else 0.0)
                 except Exception as e: print(f"计算相关系数时出错: {e}"); results['Correlation'] = np.nan
             else: results['Correlation'] = 1.0 if np.allclose(true_values, pred_values) else 0.0
        else: results['Correlation'] = np.nan
    except Exception as e: print(f"计算 {metric_name} 指标时发生意外错误: {e}")
    return results

def calculate_overall_signal_metrics(all_true_signals, all_pred_signals, signal_type="整体信号"):
    """计算整体信号指标"""
    all_mae, all_rmse, all_corrs, all_smape, all_point_percentage_errors, all_accuracy_points = [], [], [], [], [], []
    num_signals = len(all_true_signals)
    if num_signals == 0 or num_signals != len(all_pred_signals): print(f"警告: 计算 {signal_type} 指标时，真值和预测值信号数量不匹配或为零。"); return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'SMAPE': np.nan, 'Correlation': np.nan, 'Accuracy': np.nan}
    print(f"\n计算 {signal_type} 的整体性能指标...")
    processed_count = 0
    for i in range(num_signals):
        true_s = np.array(all_true_signals[i]).flatten(); pred_s = np.array(all_pred_signals[i]).flatten()
        if len(true_s) == 0 or len(pred_s) == 0 or len(true_s) != len(pred_s): continue
        processed_count += 1; all_mae.append(np.mean(np.abs(true_s - pred_s))); all_rmse.append(np.sqrt(np.mean((true_s - pred_s) ** 2)))
        var_true = np.var(true_s); var_pred = np.var(pred_s); epsilon = 1e-9
        if len(true_s) > 1 and var_true > epsilon and var_pred > epsilon:
            try: corr = np.corrcoef(true_s, pred_s)[0, 1]; all_corrs.append(corr) if not np.isnan(corr) else None
            except Exception: pass
        denominator_smape = (np.abs(true_s) + np.abs(pred_s)) / 2 + epsilon; smape_values = np.abs(true_s - pred_s) / denominator_smape; all_smape.append(np.mean(smape_values) * 100)
        transformed_true = log_transform(true_s); transformed_pred = log_transform(pred_s); abs_transformed_true = np.abs(transformed_true)
        valid_mask = abs_transformed_true > epsilon; num_valid_points = np.sum(valid_mask)
        if num_valid_points > 0:
            point_errors = np.abs(transformed_true[valid_mask] - transformed_pred[valid_mask]) / abs_transformed_true[valid_mask]; all_point_percentage_errors.extend(point_errors)
            accurate_points = np.abs(transformed_true[valid_mask] - transformed_pred[valid_mask]) <= METRICS_ACCURACY_TOLERANCE * abs_transformed_true[valid_mask]; all_accuracy_points.extend(accurate_points)
        zero_mask = ~valid_mask; num_zero_points = np.sum(zero_mask)
        if num_zero_points > 0: accurate_at_zero = np.abs(transformed_pred[zero_mask]) <= METRICS_ACCURACY_TOLERANCE * epsilon; all_accuracy_points.extend(accurate_at_zero)
    print(f"基于 {processed_count} 个有效信号对计算指标。")
    results = {
        'MAE': np.nanmean(all_mae) if all_mae else np.nan, 'RMSE': np.nanmean(all_rmse) if all_rmse else np.nan,
        'MAPE': np.nanmean(all_point_percentage_errors) * 100 if all_point_percentage_errors else np.nan,
        'SMAPE': np.nanmean(all_smape) if all_smape else np.nan, 'Correlation': np.nanmean(all_corrs) if all_corrs else np.nan,
        'Accuracy': np.nanmean(all_accuracy_points) * 100 if all_accuracy_points else np.nan}
    return results

# ========== 可视化与特征提取辅助函数 ==========
def find_main_peak_valley_pairs(signal, min_peak_dist=MIN_PEAK_DISTANCE, trough_search_window=WINDOW_SIZE_TROUGH_SEARCH, min_p2v_diff=MIN_PEAK_VALLEY_DIFF, prominence_threshold=0.05):
    """优化波峰检测（考虑显著性）"""
    signal = np.array(signal).flatten()
    if len(signal) < min_peak_dist or len(signal) < 3: return []
    peaks, _ = find_peaks(signal, distance=min_peak_dist, prominence=prominence_threshold, height=min_p2v_diff*0.5)
    valleys, _ = find_peaks(-signal)
    peak_valley_pairs = [];
    if len(peaks) == 0 or len(valleys) == 0: return []
    for peak_idx in peaks:
        max_diff_found = -float('inf'); best_valley_candidate = -1
        possible_valleys_right = valleys[(valleys > peak_idx) & (valleys <= peak_idx + trough_search_window)]
        possible_valleys_left = valleys[(valleys < peak_idx) & (valleys >= peak_idx - trough_search_window)]
        all_possible_valleys = np.concatenate((possible_valleys_right, possible_valleys_left))
        if len(all_possible_valleys) == 0: continue
        for v_idx in all_possible_valleys:
            current_diff = signal[peak_idx] - signal[v_idx]
            if current_diff >= min_p2v_diff and current_diff > max_diff_found: max_diff_found = current_diff; best_valley_candidate = v_idx
        if best_valley_candidate != -1:
             is_too_close = any(abs(best_valley_candidate - existing_valley) < min_peak_dist / 2 for _, existing_valley in peak_valley_pairs)
             if not is_too_close: peak_valley_pairs.append((peak_idx, best_valley_candidate))
    peak_valley_pairs.sort(key=lambda x: x[0])
    return peak_valley_pairs

def pair_peak_valley_data_for_height_diff(true_signal, pred_signal, true_peak_valley_pairs, pred_peak_valley_pairs, time_window_samples=PEAK_PAIRING_TIME_WINDOW_SAMPLES):
    """峰谷对配对，用于计算峰谷高度差的比较"""
    paired_true_diff_values = []; paired_pred_diff_values = []
    if not true_peak_valley_pairs or not pred_peak_valley_pairs: return np.array(paired_true_diff_values), np.array(paired_pred_diff_values)
    num_pred_pairs = len(pred_peak_valley_pairs); used_pred_pair_indices = [False] * num_pred_pairs
    for i, (true_peak_idx, true_valley_idx) in enumerate(true_peak_valley_pairs):
        best_match_pred_pair_k = -1; min_combined_dist = float('inf')
        for k, (pred_peak_k_idx, pred_valley_k_idx) in enumerate(pred_peak_valley_pairs):
            if not used_pred_pair_indices[k]:
                peak_time_dist = abs(true_peak_idx - pred_peak_k_idx); valley_time_dist = abs(true_valley_idx - pred_valley_k_idx)
                if peak_time_dist <= time_window_samples and valley_time_dist <= time_window_samples:
                    combined_dist = peak_time_dist + valley_time_dist
                    if combined_dist < min_combined_dist: min_combined_dist = combined_dist; best_match_pred_pair_k = k
        if best_match_pred_pair_k != -1:
            matched_pred_peak_idx, matched_pred_valley_idx = pred_peak_valley_pairs[best_match_pred_pair_k]
            len_true = len(true_signal); len_pred = len(pred_signal)
            if (0 <= true_peak_idx < len_true and 0 <= true_valley_idx < len_true and 0 <= matched_pred_peak_idx < len_pred and 0 <= matched_pred_valley_idx < len_pred):
                true_diff = true_signal[true_peak_idx] - true_signal[true_valley_idx]; pred_diff = pred_signal[matched_pred_peak_idx] - pred_signal[matched_pred_valley_idx]
                paired_true_diff_values.append(true_diff); paired_pred_diff_values.append(pred_diff)
                used_pred_pair_indices[best_match_pred_pair_k] = True
    return np.array(paired_true_diff_values), np.array(paired_pred_diff_values)

def plot_comparison(original_bcg, original_beddot, reconstructed_bcg_with_resp, sample_idx=0, fs=FS):
    """绘制原始信号、衰减信号和重建信号的比较图"""
    num_samples = len(original_bcg)
    if not (0 <= sample_idx < num_samples): print(f"错误: 请求的样本索引 {sample_idx} 超出范围 (0-{num_samples-1})。"); return
    true_s = np.array(original_bcg[sample_idx]).flatten(); beddot_s = np.array(original_beddot[sample_idx]).flatten(); pred_s = np.array(reconstructed_bcg_with_resp[sample_idx]).flatten()
    if len(true_s) == 0 or len(beddot_s) == 0 or len(pred_s) == 0: print(f"无法绘制样本 {sample_idx}，一个或多个信号数据为空。"); return
    min_len = min(len(true_s), len(beddot_s), len(pred_s))
    if len(true_s) != min_len or len(beddot_s) != min_len or len(pred_s) != min_len: print(f"警告: 样本 {sample_idx} 的信号长度不一致，将截断到最短长度 {min_len} 进行绘图。"); true_s=true_s[:min_len]; beddot_s=beddot_s[:min_len]; pred_s=pred_s[:min_len]
    if min_len == 0: print(f"无法绘制样本 {sample_idx}，信号长度为0。"); return
    t = np.arange(0, min_len / fs, 1 / fs)
    plt.figure(figsize=(15, 5))
    plt.plot(t, true_s, label='原始信号', alpha=0.8, color='blue', linewidth=1.5)
    plt.plot(t, beddot_s, label='BedDot信号 (衰减)', alpha=0.7, color='green', linewidth=1.0)
    plt.plot(t, pred_s, label='重建信号', linewidth=1.5, linestyle='--', color='red')
    plt.title(f'信号比较 (样本 {sample_idx})', fontsize=14); plt.xlabel('时间 (秒)', fontsize=12); plt.ylabel('幅度', fontsize=12)
    plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(); plt.show()

def plot_peak_valley(signal, peak_valley_pairs, title, color='blue', fs=FS):
    """绘制单个信号及其检测到的峰谷对"""
    signal = np.array(signal).flatten()
    if len(signal) == 0: print(f"无法绘制 '{title}'，信号为空。"); return
    t = np.arange(0, len(signal) / fs, 1 / fs); plt.plot(t, signal, label='信号', color=color, linewidth=1.0)
    plotted_peak_label, plotted_valley_label = False, False
    for i, (peak_idx, valley_idx) in enumerate(peak_valley_pairs):
        if not (0 <= peak_idx < len(signal) and 0 <= valley_idx < len(signal)): continue
        peak_label = 'J峰' if not plotted_peak_label else ""; valley_label = '谷点' if not plotted_valley_label else ""
        plt.plot(t[peak_idx], signal[peak_idx], 'o', markersize=6, color='red', label=peak_label, alpha=0.8)
        plt.plot(t[valley_idx], signal[valley_idx], 'x', markersize=6, color='purple', label=valley_label, alpha=0.8)
        plotted_peak_label = True; plotted_valley_label = True
        plt.vlines(t[peak_idx], signal[valley_idx], signal[peak_idx], color='gray', linestyle=':', alpha=0.7)
        diff = signal[peak_idx] - signal[valley_idx]; text_y_position = signal[peak_idx] * 1.05
        text_x_position = min(t[peak_idx] + (t[1]-t[0]) * 5, t[-1])
        plt.text(text_x_position, text_y_position, f'{diff:.2f}', color='black', ha='left', va='bottom', fontsize=8)
    plt.title(title, fontsize=11); plt.xlabel('时间 (秒)', fontsize=9); plt.ylabel('幅度', fontsize=9)
    handles, labels = plt.gca().get_legend_handles_labels(); by_label = dict(zip(labels, handles))
    if by_label: plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6); plt.tick_params(axis='both', which='major', labelsize=8)

def print_metrics_dict(title, metrics_dict):
    """打印指标字典的内容"""
    print(f"\n--- {title} ---")
    if not metrics_dict: print("  指标字典为空。"); return
    has_valid_metric = False
    for key in ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'Accuracy', 'Correlation']:
        val = metrics_dict.get(key, np.nan)
        if val is not None and not np.isnan(val): print(f"  {key}: {val:.4f}"); has_valid_metric = True
        else: print(f"  {key}: N/A")
    if not has_valid_metric: print(f"  未能计算有效的 {title.lower()} 指标 (可能由于数据不足或无效)。")

# ========== 主程序 ==========
def main():
    """主程序 - 使用所有数据进行训练和评估"""
    # 1. 加载数据
    all_bcg, all_beddot = load_data(); num_samples = len(all_bcg)
    if num_samples == 0: print("错误: 未能加载或生成任何数据，程序退出。"); return
    print(f"共加载 {num_samples} 个信号样本。")
    # 2. 预处理
    print("\n对所有信号进行预处理..."); all_bcg_filtered = preprocess_signals(all_bcg, highpass_filter)
    all_beddot_filtered = preprocess_signals(all_beddot, highpass_filter)
    denoise_fn = lambda x: kalman_filter(fft_denoise(wavelet_denoise(x)))
    all_beddot_denoised = preprocess_signals(all_beddot_filtered, denoise_fn)
    print("所有信号预处理完成。")
    if not isinstance(all_bcg_filtered, (np.ndarray, list)) or len(all_bcg_filtered) != num_samples: print("错误: BCG滤波后数据无效或数量不匹配。"); return
    if not isinstance(all_beddot_denoised, (np.ndarray, list)) or len(all_beddot_denoised) != num_samples: print("错误: BedDot去噪后数据无效或数量不匹配。"); return
    if isinstance(all_bcg_filtered, np.ndarray): all_bcg_filtered = list(all_bcg_filtered)
    if isinstance(all_beddot_denoised, np.ndarray): all_beddot_denoised = list(all_beddot_denoised)
    # 3. 训练模型
    best_mu, best_fl = train_model(all_bcg_filtered, all_beddot_denoised); print(f"\n训练得到的最佳NLMS参数: mu={best_mu}, filter_length={best_fl}")
    # 4. 估计通道响应
    H_avg_channel = estimate_channel_response(all_bcg_filtered, all_beddot_denoised, N_resp=N_CHANNEL_RESPONSE)
    # 5. 重建信号 (滤波后)
    reconstructed_S_estimates_filtered = reconstruct_signals_combined(all_beddot_denoised, all_bcg_filtered, H_avg_channel, best_mu, best_fl)
    if len(reconstructed_S_estimates_filtered) != num_samples: print("错误: 重建后的滤波信号数量与原始样本不匹配。"); return
    if isinstance(reconstructed_S_estimates_filtered, np.ndarray): reconstructed_S_estimates_filtered = list(reconstructed_S_estimates_filtered)
    # 6. 添加呼吸成分
    reconstructed_S_final_with_resp = add_respiration_component(reconstructed_S_estimates_filtered, all_bcg)
    if len(reconstructed_S_final_with_resp) != num_samples: print("错误: 添加呼吸成分后的信号数量与原始样本不匹配。"); return
    if isinstance(reconstructed_S_final_with_resp, np.ndarray): reconstructed_S_final_with_resp = list(reconstructed_S_final_with_resp)
    # 7. 性能评估
    print("\n--- 性能评估 ---")
    # 7.1 整体信号指标
    print("\n--- 整体信号指标 (原始BCG vs. 最终重建BCG) ---")
    overall_final_metrics = calculate_overall_signal_metrics(all_bcg, reconstructed_S_final_with_resp, signal_type="原始BCG vs 最终重建BCG")
    print_metrics_dict("原始BCG vs. 最终重建BCG", overall_final_metrics)
    # 7.2 基于特征的指标 (滤波信号对比)
    all_true_height_diffs, all_pred_height_diffs = [], []; all_paired_true_peak_amplitudes, all_paired_pred_peak_amplitudes = [], []
    all_paired_true_ibis, all_paired_pred_ibis = [], []
    print("\n计算基于特征的指标 (在滤波信号上进行)..."); feature_processed_count = 0
    for i in range(num_samples):
        original_signal_filt = all_bcg_filtered[i]; reconstructed_signal_filt = reconstructed_S_estimates_filtered[i]
        if len(original_signal_filt) < MIN_PEAK_DISTANCE or len(reconstructed_signal_filt) < MIN_PEAK_DISTANCE: continue
        original_pv_pairs = find_main_peak_valley_pairs(original_signal_filt); reconstructed_pv_pairs = find_main_peak_valley_pairs(reconstructed_signal_filt)
        if not original_pv_pairs or not reconstructed_pv_pairs: continue
        feature_processed_count += 1
        # a) 峰谷高度差
        p_true_diffs, p_pred_diffs = pair_peak_valley_data_for_height_diff(original_signal_filt, reconstructed_signal_filt, original_pv_pairs, reconstructed_pv_pairs)
        all_true_height_diffs.extend(p_true_diffs); all_pred_height_diffs.extend(p_pred_diffs)
        # b) J峰幅度和IBI
        true_j_peak_indices = sorted([p[0] for p in original_pv_pairs]); pred_j_peak_indices = sorted([p[0] for p in reconstructed_pv_pairs])
        paired_true_j_peak_times_indices, paired_pred_j_peak_times_indices = [], []; used_pred_j_indices = [False] * len(pred_j_peak_indices)
        for true_j_idx in true_j_peak_indices:
            best_match_pred_j_idx, min_diff_j, best_k_j = -1, float('inf'), -1
            for k_j, pred_j_idx in enumerate(pred_j_peak_indices):
                if not used_pred_j_indices[k_j]:
                    diff_j = abs(true_j_idx - pred_j_idx)
                    if diff_j <= PEAK_PAIRING_TIME_WINDOW_SAMPLES and diff_j < min_diff_j: min_diff_j = diff_j; best_match_pred_j_idx = pred_j_idx; best_k_j = k_j
            if best_match_pred_j_idx != -1:
                if (0 <= true_j_idx < len(original_signal_filt)) and (0 <= best_match_pred_j_idx < len(reconstructed_signal_filt)):
                     paired_true_j_peak_times_indices.append(true_j_idx); paired_pred_j_peak_times_indices.append(best_match_pred_j_idx)
                     used_pred_j_indices[best_k_j] = True
        if len(paired_true_j_peak_times_indices) >= 1:
            amps_true = original_signal_filt[paired_true_j_peak_times_indices]; amps_pred = reconstructed_signal_filt[paired_pred_j_peak_times_indices]
            all_paired_true_peak_amplitudes.extend(amps_true); all_paired_pred_peak_amplitudes.extend(amps_pred)
            if len(paired_true_j_peak_times_indices) > 1:
                ibis_true_sec = np.diff(np.array(paired_true_j_peak_times_indices)) / FS; ibis_pred_sec = np.diff(np.array(paired_pred_j_peak_times_indices)) / FS
                all_paired_true_ibis.extend(ibis_true_sec); all_paired_pred_ibis.extend(ibis_pred_sec)
    print(f"基于 {feature_processed_count} 个有效信号对计算了特征指标。")
    print_metrics_dict("峰谷高度差 指标 (滤波信号对比)", calculate_scalar_performance_metrics(all_true_height_diffs, all_pred_height_diffs, metric_name="峰谷高度差"))
    print_metrics_dict("J峰幅度 指标 (滤波信号对比)", calculate_scalar_performance_metrics(all_paired_true_peak_amplitudes, all_paired_pred_peak_amplitudes, metric_name="J峰幅度"))
    print_metrics_dict("心跳间隔(IBI) 指标 (滤波信号对比, 来自J峰)", calculate_scalar_performance_metrics(all_paired_true_ibis, all_paired_pred_ibis, metric_name="IBI"))
    # 8. 可视化
    if num_samples > 0:
        print("\n绘制第一个样本的信号比较图...")
        plot_comparison(all_bcg, all_beddot, reconstructed_S_final_with_resp, sample_idx=0, fs=FS)
        print("\n为前几个样本绘制峰谷比较图 (滤波信号)...")
        num_plots = min(3, num_samples)
        for i in range(num_plots):
             original_s_f_plot = all_bcg_filtered[i]; reconstructed_s_f_plot = reconstructed_S_estimates_filtered[i]
             if len(original_s_f_plot) == 0 or len(reconstructed_s_f_plot) == 0: print(f"样本 {i} 滤波信号为空，无法绘制峰谷图。"); continue
             original_pv_p = find_main_peak_valley_pairs(original_s_f_plot); reconstructed_pv_p = find_main_peak_valley_pairs(reconstructed_s_f_plot)
             plt.figure(figsize=(12, 7))
             plt.subplot(2, 1, 1); plot_peak_valley(original_s_f_plot, original_pv_p, f'样本 {i} - 原始滤波BCG (J峰与谷点)', fs=FS, color='blue')
             plt.subplot(2, 1, 2); plot_peak_valley(reconstructed_s_f_plot, reconstructed_pv_p, f'样本 {i} - 重建滤波BCG (J峰与谷点)', color='orange', fs=FS)
             plt.tight_layout(pad=3.0); plt.show()
    else: print("\n没有样本可供可视化。")

# 程序入口
if __name__ == "__main__":
    main()
