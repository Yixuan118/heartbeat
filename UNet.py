import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
import random


# ===================== 固定随机种子 =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior for CUDA (might slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Recommended for deterministic behavior


SET_GLOBAL_SEED = 42
set_seed(SET_GLOBAL_SEED)
print(f"全局随机种子已设置为: {SET_GLOBAL_SEED}")


# ===================== 1. 数据集定义 (数据增强参数已调整，新增时间抖动和平移) =====================
class SignalDatasetManualAugment(Dataset):
    def __init__(self, atten_signals, raw_signals, segment_length=1024):
        self.atten_signals = atten_signals
        self.raw_signals = raw_signals
        self.segment_length = segment_length

    def __len__(self):
        return len(self.atten_signals)

    def __getitem__(self, idx):
        x = np.array(self.atten_signals[idx], dtype=np.float32)
        y = np.array(self.raw_signals[idx], dtype=np.float32)

        # 确保信号长度一致且符合segment_length
        if len(x) > self.segment_length:
            x, y = x[:self.segment_length], y[:self.segment_length]
        else:
            pad = self.segment_length - len(x)
            x = np.pad(x, (0, pad), 'edge')
            y = np.pad(y, (0, pad), 'edge')

        # --- 手动 NumPy 数据增强 (参数已调整，新增时间抖动和平移) ---

        # 1. 随机增益
        if np.random.rand() < 0.5:
            gain = np.random.uniform(0.8, 1.2)  # 增益范围略收缩
            x *= gain
            y *= gain

        # 2. 随机低通滤波
        if np.random.rand() < 0.3:  # 概率降低
            window_size = np.random.randint(2, 5)  # 窗口尺寸范围 (2,6) 缩小到 (2,5)
            kernel = np.ones(window_size, dtype=np.float32) / window_size
            x = np.convolve(x, kernel, mode='same')
            y = np.convolve(y, kernel, mode='same')

        # 3. 随机高斯加噪
        if np.random.rand() < 0.5:
            noise_std = np.random.uniform(0.005, 0.03) * np.std(x)  # 噪声强度范围 (0.005,0.05) 缩小到 (0.005,0.03)
            noise = np.random.normal(0, noise_std, x.shape)
            x += noise

        # 4. 优化：模拟更真实的基线漂移 (进一步降低概率和幅度，简化形态)
        if np.random.rand() < 0.4:  # 概率从 0.5 降低到 0.4
            drift_magnitude = np.random.uniform(0.01, 0.1) * np.std(x)  # 幅度范围从 (0.02,0.15) 缩小到 (0.01,0.1)
            time_points = np.arange(len(x)) / self.segment_length
            baseline_drift = np.zeros_like(x)
            num_sine_components = np.random.randint(1, 2)  # 正弦波叠加数量从 (1,3) 缩小到 (1,2)

            for _ in range(num_sine_components):
                freq = np.random.uniform(0.5, 2.0)  # 频率范围从 (0.5,3.0) 调整到 (0.5,2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.3, 0.7)  # 振幅从 (0.2,0.8) 缩小到 (0.3,0.7)
                baseline_drift += amplitude * np.sin(2 * np.pi * freq * time_points + phase)

            if np.std(baseline_drift) > 1e-8:
                baseline_drift = baseline_drift / np.std(baseline_drift) * drift_magnitude

            x += baseline_drift
            y += baseline_drift

        # 5. 模拟随机缺失/遮蔽 (概率进一步降低)
        if np.random.rand() < 0.1:  # 概率从 0.15 降低到 0.1
            mask_length_ratio = np.random.uniform(0.01, 0.05)  # 遮蔽长度比例从 (0.02,0.1) 缩小到 (0.01,0.05)
            mask_length = int(len(x) * mask_length_ratio)

            if mask_length > 0:
                start_idx = np.random.randint(0, len(x) - mask_length + 1)
                end_idx = start_idx + mask_length
                x[start_idx:end_idx] = 0.0

        # 6. 随机整体偏移 (保持不变)
        if np.random.rand() < 0.3:
            offset = np.random.uniform(-0.1, 0.1) * np.std(x)
            x += offset
            y += offset

        # 7. 随机信号截断/饱和 (概率进一步降低)
        if np.random.rand() < 0.03:  # 概率从 0.05 降低到 0.03
            current_max = np.max(x)
            current_min = np.min(x)
            clip_threshold_upper = np.random.uniform(current_max * 0.98, current_max * 1.01)  # 范围更窄
            clip_threshold_lower = np.random.uniform(current_min * 0.98, current_min * 1.01)  # 范围更窄

            if np.random.rand() < 0.5:
                x[x > clip_threshold_upper] = clip_threshold_upper
            else:
                x[x < clip_threshold_lower] = clip_threshold_lower

        # NEW: 8. 随机时间抖动 (Time Warping) - 核心修复在此处
        if np.random.rand() < 0.3:  # 适中概率
            num_points = 5  # 控制点数量
            warp_factor = np.random.uniform(0.005, 0.015)  # 抖动强度 (更小，避免过度扭曲)

            # 原始均匀分布的时间点 (长度 len(x)，范围 [0, 1])
            t_orig_full = np.linspace(0, 1, len(x))

            # 用于定义扭曲的控制点 (长度 num_points，范围 [0, 1])
            t_control_points_orig = np.linspace(0, 1, num_points)

            # 生成扭曲后的控制点，并确保单调性和边界
            t_control_points_warped = t_control_points_orig + np.random.uniform(-warp_factor, warp_factor, num_points)
            t_control_points_warped[0] = 0  # 固定起始点
            t_control_points_warped[-1] = 1  # 固定结束点
            t_control_points_warped = np.sort(t_control_points_warped)  # 确保单调性

            # 核心修正：
            # new_indices_for_sampling 描述了，在扭曲后的时间轴上，
            # 每一个原始信号的 t_orig_full 点，它应该从原始信号的哪个位置 (0-1 范围) 采样。
            # x: t_orig_full (要插值到的目标x坐标)
            # xp: t_control_points_warped (已知数据点的x坐标，即扭曲后的控制点)
            # fp: t_control_points_orig (已知数据点的y坐标，即扭曲前的原始控制点)
            new_indices_for_sampling = np.interp(t_orig_full, t_control_points_warped, t_control_points_orig)

            # 使用新计算出的采样索引对信号进行插值以实现时间扭曲
            # x: new_indices_for_sampling (新的采样位置，范围 [0, 1])
            # xp: t_orig_full (原始信号的x坐标，用于查找原始信号值)
            # fp: x (原始信号的y值)
            x = np.interp(new_indices_for_sampling, t_orig_full, x)
            y = np.interp(new_indices_for_sampling, t_orig_full, y)

        # NEW: 9. 随机小幅度时间平移 (Time Shifting)
        if np.random.rand() < 0.4:  # 适中概率
            max_shift = int(len(x) * np.random.uniform(0.005, 0.01))  # 最大平移量，例如0.5% - 1% 信号长度
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                x = np.roll(x, shift)
                y = np.roll(y, shift)

        # --- 归一化 (核心保持原始逐样本归一化) ---
        x_mean, x_std = np.mean(x), np.std(x) + 1e-8
        y_mean, y_std = np.mean(y), np.std(y) + 1e-8
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        # 转换为Tensor
        return torch.from_numpy(x_norm).float(), torch.from_numpy(y_norm).float()


# ===================== 2. U-Net1D模型 (增加模型容量参数, 引入Dropout) =====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.2):  # 引入dropout_rate
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 在ReLU后添加Dropout
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)  # 在ReLU后添加Dropout
        )

    def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):  # 传递dropout_rate
        super().__init__();
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)  # 使用带dropout的DoubleConv
        )

    def forward(self, x): return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.2):  # 传递dropout_rate
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True);
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)  # 使用带dropout的DoubleConv
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2);
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)  # 使用带dropout的DoubleConv

    def forward(self, x1, x2):
        x1 = self.up(x1);

        # 核心修复：调整 x1 的长度以精确匹配 x2
        diff = x2.size()[2] - x1.size()[2];
        if diff > 0:  # x1 比 x2 短，需要填充
            # 计算左右填充量
            padding_left = diff // 2
            padding_right = diff - padding_left
            x1 = nn.functional.pad(x1, [padding_left, padding_right]);
        elif diff < 0:  # x1 比 x2 长，需要截断
            # 计算左右截断量
            trunc_left = abs(diff) // 2
            trunc_right = abs(diff) - trunc_left
            x1 = x1[:, :, trunc_left:x1.size()[2] - trunc_right]

        x = torch.cat([x2, x1], dim=1);
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__();
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x): return self.conv(x)


class UNet1D(nn.Module):
    # 添加 base_channels 参数，默认值增加到 128
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_channels=128,
                 dropout_rate=0.2):  # 增加 dropout_rate 参数
        super().__init__();
        self.n_channels, self.n_classes, self.bilinear = n_channels, n_classes, bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, base_channels, dropout_rate=dropout_rate);
        self.down1 = Down(base_channels, base_channels * 2, dropout_rate=dropout_rate);
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate);
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout_rate=dropout_rate);
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout_rate=dropout_rate)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout_rate=dropout_rate);
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout_rate=dropout_rate);
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout_rate=dropout_rate);
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x1 = self.inc(x);
        x2 = self.down1(x1);
        x3 = self.down2(x2);
        x4 = self.down3(x3);
        x5 = self.down4(x4)
        x = self.up1(x5, x4);
        x = self.up2(x, x3);
        x = self.up3(x, x2);
        x = self.up4(x, x1)
        return self.outc(x).squeeze(1)


# ===================== NEW: 组合损失函数 =====================
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.7, mse_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.l1_weight * l1 + self.mse_weight * mse


# ===================== 3. 训练与推理 (损失函数更改为 CombinedLoss) =====================
def train_model_final(model, train_loader, val_loader, device, epochs=150, lr=1e-4, patience=20):
    model.to(device);
    optimizer = optim.Adam(model.parameters(), lr=lr);
    # 核心修改：使用组合损失函数
    loss_fn = CombinedLoss(l1_weight=0.7, mse_weight=0.3)  # L1 权重更高，保留波形细节，MSE 辅助整体匹配

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)  # 降低eta_min

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    best_epoch = 0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for x_norm, y_norm in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)"):
            x_norm, y_norm = x_norm.to(device), y_norm.to(device)
            optimizer.zero_grad()
            out_norm = model(x_norm)
            loss = loss_fn(out_norm, y_norm)
            loss.backward();
            optimizer.step();
            total_train_loss += loss.item()

        scheduler.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} (Val)"):
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val)
                val_loss = loss_fn(out_val, y_val)
                total_val_loss += val_loss.item()
        model.train()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"早停触发！验证集损失在 {patience} 个epoch内没有改善。最佳epoch为 {best_epoch}。")
                break

    if best_model_state is None:  # 如果没有触发早停，就保存最后一个epoch的模型状态
        best_model_state = model.state_dict()
        best_epoch = epochs  # 记录最后一个epoch作为最佳

    return best_model_state, train_losses, val_losses, best_epoch


def predict_final(model, atten_signals, device, target_mean, target_std, segment_length=1024):
    model.eval().to(device);
    results = []
    with torch.no_grad():
        for signal in tqdm(atten_signals, desc="推理中"):
            x = np.array(signal, dtype=np.float32);
            original_len = len(x)

            # 对比训练集，推理阶段无需数据增强，但仍需处理长度和归一化
            if len(x) > segment_length:
                x_processed = x[:segment_length]
            else:
                x_processed = np.pad(x, (0, segment_length - len(x)), 'edge')

            x_mean, x_std = np.mean(x_processed), np.std(x_processed) + 1e-8
            x_norm = (x_processed - x_mean) / x_std
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device).float()

            out_norm = model(x_tensor).cpu().numpy().flatten()

            # 预测结果反归一化，使用全局的target_mean和target_std
            pred_rescaled = out_norm * target_std + target_mean

            results.append(pred_rescaled[:original_len])  # 截取回原始长度
    return results


# ===================== 4. 数据加载与评估 =====================
def load_training_data():
    print("\n加载训练数据...");
    try:
        # 请根据实际文件路径修改
        all_bcg_signals = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-08-19T130203_2025-08-19T131924.npy',
            allow_pickle=True)
        all_beddot_signals = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-08-19T130203_2025-08-19T131924.npy',
            allow_pickle=True)
        return list(all_beddot_signals), list(all_bcg_signals)
    except FileNotFoundError:
        print("错误: 训练数据文件未找到！请检查路径。");
        return None, None


def load_testing_data():
    print("\n加载独立的测试数据...");
    try:
        # 请根据实际文件路径修改
        ground_truth_signal = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_before_2025-08-19T155611_2025-08-19T155851.npy',
            allow_pickle=True)
        beddot_to_predict = np.load(
            r'D:\UGA\CoreDemo-master\CoreDemo-master\influxexample\raw_signal_after_2025-08-19T155611_2025-08-19T155851.npy',
            allow_pickle=True)
        return list(ground_truth_signal), list(beddot_to_predict)
    except FileNotFoundError:
        print("错误: 测试数据文件未找到！请检查路径。");
        return None, None


def find_signal_peaks_troughs(signal, fs=100):
    if len(signal) < fs * 0.5:
        return np.array([]), np.array([])

    signal_range = np.max(signal) - np.min(signal)
    if signal_range < 1e-6:
        return np.array([]), np.array([])

    prominence_threshold = signal_range * 0.05
    distance_threshold = int(0.4 * fs)  # 40采样点 (假设心率在60-120bpm，周期0.5-1秒)

    peaks, _ = find_peaks(signal, prominence=prominence_threshold, distance=distance_threshold)
    troughs, _ = find_peaks(-signal, prominence=prominence_threshold, distance=distance_threshold)

    return peaks, troughs


def calculate_average_peak_trough_amplitude(signal, fs=100):
    peaks, troughs = find_signal_peaks_troughs(signal, fs)
    amplitude_diffs = []

    for peak_idx in peaks:
        subsequent_troughs = troughs[troughs > peak_idx]
        if subsequent_troughs.size > 0:
            trough_idx = subsequent_troughs[0]
            amplitude_diffs.append(signal[peak_idx] - signal[trough_idx])
            continue

        preceding_troughs = troughs[troughs < peak_idx]
        if preceding_troughs.size > 0:
            trough_idx = preceding_troughs[-1]
            amplitude_diffs.append(signal[peak_idx] - signal[trough_idx])

    if amplitude_diffs:
        return np.mean(amplitude_diffs)
    else:
        return 0.0


def evaluate_metrics(true_signals, pred_signals, fs=100):
    maes, mses, cors, dtw_distances, amplitude_recovery_ratios, cosine_similarities = [], [], [], [], [], []

    for i in range(len(true_signals)):
        t = np.array(true_signals[i]).flatten()
        p = np.array(pred_signals[i]).flatten()  # 修复：这里是 pred_signals[i]，而不是未定义的 'pred'
        min_len = min(len(t), len(p));
        t, p = t[:min_len], p[:min_len]

        if min_len < 2:
            continue

        maes.append(np.mean(np.abs(t - p)));
        mses.append(np.mean((t - p) ** 2))

        if np.std(t) > 1e-6 and np.std(p) > 1e-6:
            try:
                corr_val = np.corrcoef(t, p)[0, 1]
                if not np.isnan(corr_val):
                    cors.append(corr_val)
            except Exception:
                pass

        if min_len > 1:
            t_reshaped = t.reshape(-1, 1)
            p_reshaped = p.reshape(-1, 1)
            distance, path = fastdtw(t_reshaped, p_reshaped, dist=euclidean)
            dtw_distances.append(distance / min_len)

        norm_t = np.linalg.norm(t)
        norm_p = np.linalg.norm(p)
        if norm_t > 1e-8 and norm_p > 1e-8:
            cos_sim = np.dot(t, p) / (norm_t * norm_p)
            if not np.isnan(cos_sim):
                cosine_similarities.append(cos_sim)

        true_amplitude = calculate_average_peak_trough_amplitude(t, fs)
        pred_amplitude = calculate_average_peak_trough_amplitude(p, fs)

        if true_amplitude > 1e-8:
            amplitude_recovery_ratios.append(pred_amplitude / true_amplitude)
        elif true_amplitude < 1e-8 and pred_amplitude < 1e-8:
            amplitude_recovery_ratios.append(1.0)
        else:
            amplitude_recovery_ratios.append(0.0)

    print(f"MAE: {np.mean(maes):.4f}");
    print(f"MSE: {np.mean(mses):.4f}")
    if cors:
        print(f"相关系数: {np.mean(cors):.4f}")
    else:
        print("相关系数: 无法计算 (所有样本标准差过低或相关系数为NaN)")

    if dtw_distances:
        print(f"平均DTW距离 (归一化): {np.mean(dtw_distances):.4f} (越低越好)")
    else:
        print("平均DTW距离: 无法计算 (样本长度过短)")

    if cosine_similarities:
        print(f"平均余弦相似度: {np.mean(cosine_similarities):.4f} (越接近1越好)")
    else:
        print("平均余弦相似度: 无法计算 (信号模长过低或为NaN)")

    if amplitude_recovery_ratios:
        valid_ratios = [r for r in amplitude_recovery_ratios if not np.isnan(r)]
        if valid_ratios:
            print(f"平均周期幅度恢复比例 (理想值: 1.0): {np.mean(valid_ratios):.4f}")
        else:
            print("平均周期幅度恢复比例: 无法计算 (所有样本均无有效周期幅度)")
    else:
        print("平均周期幅度恢复比例: 无法计算 (没有有效样本)")


def plot_signals(raw, atten, recon, idx=0, fs=100, title_prefix=""):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'];
    matplotlib.rcParams['axes.unicode_minus'] = False
    min_len = min(len(raw[idx]), len(atten[idx]), len(recon[idx]));
    t = np.arange(min_len) / fs
    plt.figure(figsize=(15, 10));
    plt.subplot(3, 1, 1);
    plt.plot(t, raw[idx][:min_len], 'b', label='原始信号');
    plt.ylabel('幅度');
    plt.title(f'{title_prefix}第{idx}个样本-原始信号');
    plt.legend();
    plt.grid(True)
    plt.subplot(3, 1, 2);
    plt.plot(t, atten[idx][:min_len], 'g', label='衰减信号');
    plt.ylabel('幅度');
    plt.title(f'{title_prefix}第{idx}个样本-衰减信号');
    plt.legend();
    plt.grid(True)
    plt.subplot(3, 1, 3);
    plt.plot(t, recon[idx][:min_len], 'r', label='重建信号');
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度');
    plt.title(f'{title_prefix}第{idx}个样本-重建信号');
    plt.legend();
    plt.grid(True)
    plt.tight_layout();
    plt.show()


# ===================== 新增的同轴对比绘图函数 (加入波峰波谷可视化) =====================
def plot_signals_combined(raw, atten, recon, idx=0, fs=100, title_prefix=""):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'];
    matplotlib.rcParams['axes.unicode_minus'] = False

    min_len = min(len(raw[idx]), len(atten[idx]), len(recon[idx]));
    t = np.arange(min_len) / fs

    plt.figure(figsize=(15, 6))
    plt.title(f'{title_prefix}第{idx}个样本 - 三信号同轴对比与波峰波谷', fontsize=16)

    # 绘制三个信号
    plt.plot(t, raw[idx][:min_len], 'b-', linewidth=1.5, alpha=0.9, label='原始信号 (Ground Truth)')
    plt.plot(t, atten[idx][:min_len], 'g--', linewidth=1, alpha=0.6, label='衰减信号 (Input)')
    plt.plot(t, recon[idx][:min_len], 'r-', linewidth=1.5, alpha=0.9, label='重建信号 (Predicted)')

    # --- 标记原始信号的波峰和波谷 ---
    raw_peaks, raw_troughs = find_signal_peaks_troughs(raw[idx][:min_len], fs)
    if raw_peaks.size > 0:
        plt.plot(t[raw_peaks], raw[idx][:min_len][raw_peaks], 'bo', markersize=5, alpha=0.7, label='原始信号波峰')
    if raw_troughs.size > 0:
        plt.plot(t[raw_troughs], raw[idx][:min_len][raw_troughs], 'bv', markersize=5, alpha=0.7, label='原始信号波谷')

    # --- 标记重建信号的波峰和波谷 ---
    recon_peaks, recon_troughs = find_signal_peaks_troughs(recon[idx][:min_len], fs)
    if recon_peaks.size > 0:
        plt.plot(t[recon_peaks], recon[idx][:min_len][recon_peaks], 'ro', markerfacecolor='none', markeredgecolor='red',
                 markersize=7, alpha=0.8, label='重建信号波峰')
    if recon_troughs.size > 0:
        plt.plot(t[recon_troughs], recon[idx][:min_len][recon_troughs], 'rv', markerfacecolor='none',
                 markeredgecolor='red', markersize=7, alpha=0.8, label='重建信号波谷')

    plt.xlabel('时间 (秒)', fontsize=12)
    plt.ylabel('幅度', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ===================== 新增：绘制损失曲线函数 =====================
def plot_loss_curves(train_losses, val_losses, best_epoch):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'];
    matplotlib.rcParams['axes.unicode_minus'] = False

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_losses, 'b-', label='训练损失')
    plt.plot(epochs_range, val_losses, 'r-', label='验证损失')
    plt.title('训练与验证损失曲线', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    if best_epoch > 0:
        plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'最佳Epoch ({best_epoch})', linewidth=1.5)
        plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


# ===================== 5. 主流程 =====================
if __name__ == "__main__":
    # --- 确保安装必要的库 ---
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
    except ImportError:
        print("请安装 'fastdtw' 库: pip install fastdtw")
        exit(1)
    # ---------------------------

    # 1. 加载数据
    train_inputs_full, train_targets_full = load_training_data()
    test_targets, test_inputs = load_testing_data()

    if train_inputs_full is None or test_inputs is None:
        print("数据加载失败，程序终止。")
        exit(1)

    # --- 修正：将训练数据按时间顺序划分为训练集和验证集 ---
    total_train_samples = len(train_inputs_full)
    val_split_ratio = 0.2
    split_idx = int(total_train_samples * (1 - val_split_ratio))

    train_inputs = train_inputs_full[:split_idx]
    train_targets = train_targets_full[:split_idx]

    val_inputs = train_inputs_full[split_idx:]
    val_targets = train_targets_full[split_idx:]

    print(f"\n训练数据已按时间顺序划分为：训练集 {len(train_inputs)} 个样本, 验证集 {len(val_inputs)} 个样本.")
    # -----------------------------------------------------------

    # 2. 计算先验统计量 (从划分后的训练子集计算)
    # 这里的global_target_mean/std依然是用于预测阶段的去归一化，而不是训练阶段y的归一化
    all_train_targets_flat = np.concatenate([np.array(s).flatten() for s in train_targets])
    global_target_mean = np.mean(all_train_targets_flat)
    global_target_std = np.std(all_train_targets_flat)
    print(f"\n从训练集计算出的用于【去归一化】的统计量: Mean={global_target_mean:.2f}, Std={global_target_std:.2f}")

    # 3. 使用手动实现的、无依赖的增强数据集
    train_dataset = SignalDatasetManualAugment(train_inputs, train_targets, segment_length=1024)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                              pin_memory=True)  # num_workers=0 在 Windows 上避免多进程问题

    val_dataset = SignalDatasetManualAugment(val_inputs, val_targets, segment_length=1024)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0,
                            pin_memory=True)  # num_workers=0 在 Windows 上避免多进程问题

    # 4. 初始化模型 (使用新的 base_channels 和 dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = UNet1D(base_channels=128, dropout_rate=0.2)  # 核心修改：增加模型容量，通道数从96改为128，并设置dropout率

    # 5. 训练 (捕获损失列表和最佳epoch对应的模型状态)
    # train_model_final 中损失函数已更改为 CombinedLoss
    best_model_state_dict, train_losses, val_losses, best_epoch = train_model_final(
        model, train_loader, val_loader, device, epochs=200, lr=1e-4, patience=20  # 增加epochs到200
    )

    # --- 绘制损失曲线 ---
    print("\n===== 绘制损失曲线 =====")
    plot_loss_curves(train_losses, val_losses, best_epoch)

    # === 重要：加载最佳 epoch 对应的模型权重 ===
    # 实例化一个新的模型实例以加载权重，注意也要使用新的 base_channels 和 dropout_rate
    model_for_eval = UNet1D(base_channels=128, dropout_rate=0.2)
    model_for_eval.load_state_dict(best_model_state_dict)  # 加载在验证集上表现最佳时的模型权重
    model_for_eval.to(device)
    print(f"\n已加载验证损失最低时（Epoch {best_epoch}）的模型权重进行评估。")
    # =========================================

    # --- 训练集评估 ---
    print("\n\n" + "=" * 20 + " 训练集评估 " + "=" * 20)
    print("\n===== 在训练集上进行推理以供评估 =====")
    pred_train = predict_final(model_for_eval, train_inputs, device, global_target_mean, global_target_std)

    print("\n===== 自动绘制前3个训练样本分离对比图 =====")
    n_plot_train = min(3, len(train_targets), len(train_inputs), len(pred_train))
    for idx in range(n_plot_train):
        plot_signals(train_targets, train_inputs, pred_train, idx=idx, fs=100, title_prefix="【训练集】")

    print("\n===== 自动绘制前3个训练样本同轴对比图 (含波峰波谷) =====")
    for idx in range(n_plot_train):
        plot_signals_combined(train_targets, train_inputs, pred_train, idx=idx, fs=100, title_prefix="【训练集】")

    print("\n===== 重建效果评估 (训练集) =====")
    evaluate_metrics(train_targets, pred_train, fs=100)

    # --- 测试集评估 ---
    print("\n\n" + "=" * 20 + " 测试集评估 " + "=" * 20)
    print("\n===== 在测试集上进行推理 =====")
    pred_test = predict_final(model_for_eval, test_inputs, device, global_target_mean, global_target_std)

    print("测试集重建完成，结果已保存。")
    # np.save("reconstructed_test.npy", np.array(pred_test, dtype=object)) # 取消保存，避免不必要的磁盘写入

    print("\n===== 自动绘制前3个测试样本分离对比图 =====")
    n_plot_test = min(3, len(test_targets), len(test_inputs), len(pred_test))
    for idx in range(n_plot_test):
        plot_signals(test_targets, test_inputs, pred_test, idx=idx, fs=100, title_prefix="【测试集】")

    print("\n===== 自动绘制前3个测试样本同轴对比图 (含波峰波谷) =====")
    for idx in range(n_plot_test):
        plot_signals_combined(test_targets, test_inputs, pred_test, idx=idx, fs=100, title_prefix="【测试集】")

    print("\n===== 重建效果评估 (测试集) =====")
    evaluate_metrics(test_targets, pred_test, fs=100)
