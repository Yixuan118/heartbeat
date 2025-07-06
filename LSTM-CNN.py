# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math

warnings.filterwarnings("ignore")
# 设置Matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================================================================
# 1. 全局参数配置
# ===================================================================
SEGMENT_LENGTH = 1024
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BILINEAR_UPSAMPLE = True
FS = 100
MIN_PEAK_DISTANCE = 50
MIN_PEAK_VALLEY_DIFF = 0.15
PEAK_PAIRING_TIME_WINDOW_SAMPLES = 15

from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_dim: int = SEGMENT_LENGTH
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.0005
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lstm_layers: int = 2


# ===================================================================
# 2. 1D U-Net 模型定义 (无需修改)
# ===================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__();
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x): return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1);
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2]);
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__();
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x): return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=BILINEAR_UPSAMPLE):
        super(UNet1D, self).__init__()
        self.n_channels, self.n_classes, self.bilinear = n_channels, n_classes, bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1, self.down2, self.down3, self.down4 = Down(64, 128), Down(128, 256), Down(256, 512), Down(512,
                                                                                                             1024 // factor)
        self.up1, self.up2, self.up3, self.up4 = Up(1024, 512 // factor, bilinear), Up(512, 256 // factor,
                                                                                       bilinear), Up(256, 128 // factor,
                                                                                                     bilinear), Up(128,
                                                                                                                   64,
                                                                                                                   bilinear)
        self.outc = OutConv(64, n_classes)

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


# ===================================================================
# 3. Seq2Seq LSTM-CNN模型定义 (无需修改)
# ===================================================================
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = (out_channels // 3) * 3
        if self.out_channels == 0: self.out_channels = out_channels
        channels_per_scale = self.out_channels // 3
        self.use_multiscale = channels_per_scale * 3 == self.out_channels and channels_per_scale > 0
        if self.use_multiscale:
            self.conv3 = nn.Conv1d(in_channels, channels_per_scale, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(in_channels, channels_per_scale, kernel_size=5, padding=2)
            self.conv7 = nn.Conv1d(in_channels, channels_per_scale, kernel_size=7, padding=3)
        else:
            self.conv3 = nn.Conv1d(in_channels, self.out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.use_multiscale:
            out = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        else:
            out = self.conv3(x)
        return self.relu(out)


class LSTMCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        cnn_out_channels = 384
        self.cnn_encoder = nn.Sequential(
            MultiScaleConv(1, 96), nn.MaxPool1d(2), MultiScaleConv(96, 192), nn.MaxPool1d(2),
            MultiScaleConv(192, cnn_out_channels), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=config.d_model, num_layers=config.lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=config.dropout if config.lstm_layers > 1 else 0)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(config.d_model * 2, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.cnn_encoder(x);
        x = x.transpose(1, 2);
        lstm_out, _ = self.lstm(x);
        x = lstm_out.transpose(1, 2)
        return self.decoder(x).squeeze(1)


# ===================================================================
# 4. 信号处理与数据集类 (无需修改)
# ===================================================================
def highpass_filter(data, cutoff=0.8, fs=FS, order=4):
    if len(data) <= order * 3: return data
    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, data)


def extract_respiration(data, cutoff=0.5, fs=FS, order=4):
    if len(data) <= order * 3: return np.zeros_like(data)
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)


def add_realistic_noise(signal, fs=FS):
    """
    向信号中概率性地添加多种类型的噪声以进行数据增强。
    :param signal: 输入的1D numpy信号数组。
    :param fs: 信号的采样率。
    :return: 添加了噪声的信号数组。
    """
    augmented_signal = signal.copy()
    p2p = np.ptp(augmented_signal)
    if p2p < 1e-6: p2p = 1.0  # 避免信号太平坦导致除零错误

    # 1. 概率性添加高斯噪声
    if np.random.rand() < 0.7:  # 70%的概率添加高斯噪声
        noise_level = np.random.uniform(0.01, 0.08) * p2p
        gaussian_noise = np.random.normal(0, noise_level, augmented_signal.shape)
        augmented_signal += gaussian_noise

    # 2. 概率性添加运动伪影 (低频正弦波)
    if np.random.rand() < 0.4:  # 40%的概率添加运动伪影
        num_waves = np.random.randint(1, 3)
        for _ in range(num_waves):
            artifact_amp = np.random.uniform(0.1, 0.5) * p2p
            artifact_freq = np.random.uniform(0.05, 0.4)  # 极低频
            artifact_phase = np.random.uniform(0, 2 * np.pi)
            time_axis = np.arange(len(augmented_signal)) / fs
            motion_artifact = artifact_amp * np.sin(2 * np.pi * artifact_freq * time_axis + artifact_phase)
            augmented_signal += motion_artifact

    # 3. 概率性添加随机尖峰
    if np.random.rand() < 0.3:  # 30%的概率添加随机尖峰
        num_spikes = np.random.randint(1, 6)
        spike_indices = np.random.randint(0, len(augmented_signal), num_spikes)
        for idx in spike_indices:
            spike_amp = np.random.uniform(0.5, 2.0) * p2p * np.random.choice([-1, 1])
            augmented_signal[idx] += spike_amp

    return augmented_signal


class SignalDataset_NoNorm(Dataset):
    def __init__(self, input_signals, target_signals, segment_length=SEGMENT_LENGTH, is_train=True):
        self.input_signals = input_signals
        self.target_signals = target_signals
        self.segment_length = segment_length
        self.is_train = is_train

    def __len__(self):
        return len(self.input_signals)

    def __getitem__(self, idx):
        input_seg_orig = np.array(self.input_signals[idx]).flatten().astype(np.float32)
        target_seg = np.array(self.target_signals[idx]).flatten().astype(np.float32)

        # 随机裁剪或填充到固定长度
        if len(input_seg_orig) > self.segment_length:
            start_idx = np.random.randint(0, len(input_seg_orig) - self.segment_length + 1) if self.is_train else 0
            input_seg = input_seg_orig[start_idx: start_idx + self.segment_length]
            target_seg = target_seg[start_idx: start_idx + self.segment_length]
        else:  # len(input_seg) <= self.segment_length
            pad_len = self.segment_length - len(input_seg_orig)
            input_seg = np.pad(input_seg_orig, (0, pad_len), 'edge')
            target_seg = np.pad(target_seg, (0, pad_len), 'edge')

        # 如果是训练集，则对输入信号应用噪声增强
        if self.is_train:
            input_seg = add_realistic_noise(input_seg, fs=FS)

        # 幅度随机化数据增强 (对含噪输入和干净目标应用相同因子)
        if self.is_train:
            scale_factor = np.random.uniform(0.2, 5.0)
            input_seg *= scale_factor
            target_seg *= scale_factor

        return torch.from_numpy(input_seg), torch.from_numpy(target_seg)


# ===================================================================
# 5. 数据加载函数 (无需修改)
# ===================================================================
def load_training_data():
    print("\n加载预生成的Chirp训练数据...");
    try:
        all_beddot_signals = np.load(r'D:\UGA\heartbeat_system\data\chirp_input_samples.npy', allow_pickle=True)
        all_bcg_signals = np.load(r'D:\UGA\heartbeat_system\data\chirp_output_samples.npy', allow_pickle=True)
        print(f"成功加载Chirp训练数据: 输入 {len(all_beddot_signals)}, 目标 {len(all_bcg_signals)}");
        return list(all_beddot_signals), list(all_bcg_signals)
    except FileNotFoundError:
        print("错误: 预生成的训练数据文件未找到！请检查路径。");
        return None, None


def load_testing_data():
    print("\n加载独立的测试数据...");
    try:
        ground_truth_signal = np.load(r'D:\UGA\heartbeat_system\data\ABP_extracted_first6000.npy', allow_pickle=True)
        beddot_to_predict = np.load(r'D:\UGA\heartbeat_system\data\beddot_signals.npy', allow_pickle=True)
        print(f"成功加载测试数据: BCG {len(ground_truth_signal)}, BedDot {len(beddot_to_predict)}");
        return list(ground_truth_signal), list(beddot_to_predict)
    except FileNotFoundError:
        print("错误: 测试数据文件未找到！请检查路径。");
        return None, None


# ===================================================================
# 6. 模型训练与推理 (无需修改)
# ===================================================================
def train_model(dataset, model_type='UNet1D'):
    if model_type == 'UNet1D':
        model = UNet1D().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        print(f"\n在 {DEVICE} 上进行【U-Net】训练 (LR={LEARNING_RATE}) (损失函数: MSELoss)...")
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss) or torch.isinf(loss): print(f"!!! 警告: Epoch {epoch + 1} 损失发散!"); return None
                loss.backward();
                optimizer.step();
                epoch_loss += loss.item()
            print(f"--- Epoch {epoch + 1}/{EPOCHS}, 平均MSE损失: {epoch_loss / len(loader):.6f} ---")
    elif model_type == 'LSTMCNN':
        config = ModelConfig()
        model = LSTMCNN(config).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        print(f"\n在 {DEVICE} 上进行【LSTMCNN Seq2Seq】训练 (LR={config.learning_rate}) (损失函数: MSELoss)...")
        model.train()
        for epoch in range(config.num_epochs):
            total_loss = 0
            for features, labels in loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward();
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0);
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f'Epoch {epoch + 1}/{config.num_epochs}, Training MSE Loss: {avg_loss:.6f}')
            scheduler.step(avg_loss)
    return model


def reconstruct_long_signal(model, signal, window_size=SEGMENT_LENGTH, step_size=None):
    model.eval()
    step_size = step_size or window_size // 2
    signal = signal.flatten().astype(np.float32)
    signal_len = len(signal)
    if signal_len == 0: return np.array([])
    if signal_len <= window_size:
        tensor_in = torch.from_numpy(signal).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        with torch.no_grad(): output = model(tensor_in)
        return output.cpu().numpy().flatten()
    reconstruction, window_sum = np.zeros(signal_len), np.zeros(signal_len)
    window = np.bartlett(window_size)
    for i in range(0, signal_len, step_size):
        start, end = i, min(i + window_size, signal_len)
        if end == signal_len: start = signal_len - window_size
        segment = signal[start:start + window_size]
        tensor_in = torch.from_numpy(segment).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor_in)
        reconstruction[start:start + window_size] += output.cpu().numpy().flatten() * window
        window_sum[start:start + window_size] += window
        if end == signal_len: break
    window_sum[window_sum == 0] = 1
    return reconstruction / window_sum


def reconstruct_signals(model, signals):
    results = []
    print(f"\n使用 {type(model).__name__} 重建信号 (无归一化，端到端)...")
    for i, s in enumerate(signals):
        print(f"  处理 {i + 1}/{len(signals)}...");
        results.append(reconstruct_long_signal(model, s))
    return results


# ===================================================================
# 7. 评估与绘图函数 (无需修改)
# ===================================================================
def calculate_overall_signal_metrics(all_true, all_pred):
    metrics_list = [calculate_scalar_performance_metrics(t, p) for t, p in zip(all_true, all_pred)]
    if not metrics_list: return {}
    return {key: np.nanmean([m[key] for m in metrics_list]) for key in metrics_list[0]}


def calculate_scalar_performance_metrics(true_signal, pred_signal):
    metrics = {'MAE': np.nan, 'RMSE': np.nan, 'Correlation': np.nan, 'SMAPE': np.nan}
    true, pred = np.array(true_signal).flatten(), np.array(pred_signal).flatten()
    min_len = min(len(true), len(pred))
    if min_len == 0: return metrics
    true, pred = true[:min_len], pred[:min_len]
    metrics['MAE'] = np.mean(np.abs(true - pred));
    metrics['RMSE'] = np.sqrt(np.mean((true - pred) ** 2))
    epsilon = 1e-9
    metrics['SMAPE'] = np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2 + epsilon)) * 100
    if min_len > 1 and np.var(true) > epsilon and np.var(pred) > epsilon:
        metrics['Correlation'] = np.corrcoef(true, pred)[0, 1]
    return metrics


def find_main_peak_valley_pairs(signal, distance=MIN_PEAK_DISTANCE, prominence_ratio=0.05):
    signal = np.array(signal).flatten()
    if len(signal) < distance: return []
    prominence = (np.max(signal) - np.min(signal)) * prominence_ratio
    if prominence < 1e-9: return []
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-signal, distance=distance)
    pairs = []
    for p_idx in peaks:
        possible_valleys = valleys[np.abs(valleys - p_idx) < PEAK_PAIRING_TIME_WINDOW_SAMPLES]
        if len(possible_valleys) > 0:
            v_idx = possible_valleys[np.argmin(np.abs(possible_valleys - p_idx))]
            pairs.append((p_idx, v_idx))
    return sorted(list(set(pairs)), key=lambda x: x[0])


def evaluate_feature_metrics(true_filtered, pred_filtered):
    p2v_maes, ibi_maes = [], []
    for true_f, pred_f in zip(true_filtered, pred_filtered):
        true_pairs, pred_pairs = find_main_peak_valley_pairs(true_f), find_main_peak_valley_pairs(pred_f)
        if not true_pairs or not pred_pairs: continue
        true_p2v = [true_f[p] - true_f[v] for p, v in true_pairs]
        pred_p2v = [pred_f[p] - pred_f[v] for p, v in pred_pairs]
        min_len_p2v = min(len(true_p2v), len(pred_p2v))
        if min_len_p2v > 0: p2v_maes.append(
            np.mean(np.abs(np.array(true_p2v[:min_len_p2v]) - np.array(pred_p2v[:min_len_p2v]))))
        true_ibi, pred_ibi = np.diff([p for p, v in true_pairs]) / FS, np.diff([p for p, v in pred_pairs]) / FS
        min_len_ibi = min(len(true_ibi), len(pred_ibi))
        if min_len_ibi > 0: ibi_maes.append(
            np.mean(np.abs(np.array(true_ibi[:min_len_ibi]) - np.array(pred_ibi[:min_len_ibi]))))
    return {'P2V_Height_MAE': np.nanmean(p2v_maes), 'IBI_MAE_sec': np.nanmean(ibi_maes)}


def print_metrics_dict(title, metrics):
    print(f"\n--- {title} ---")
    for key, value in metrics.items(): print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: N/A")


def plot_comparison(orig, beddot, recon, idx=0, fs=FS, suffix=""):
    t, b, p = orig[idx], beddot[idx], recon[idx]
    min_len = min(len(t), len(b), len(p))
    time_axis = np.arange(min_len) / fs
    plt.figure(figsize=(18, 6));
    plt.plot(time_axis, t[:min_len], label='真实ABP', alpha=0.8, color='blue');
    plt.plot(time_axis, b[:min_len], label='衰减信号', alpha=0.5, color='green');
    plt.plot(time_axis, p[:min_len], label='重建ABP', linestyle='--', color='red', alpha=0.9);
    plt.title(f'信号比较 (样本 {idx}){suffix}');
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度');
    plt.legend();
    plt.grid(True);
    plt.tight_layout();
    plt.show()


def plot_peak_valley(signal, pairs, title, color='blue', fs=FS):
    time_axis = np.arange(len(signal)) / fs;
    plt.plot(time_axis, signal, label='信号', color=color)
    peak_indices, valley_indices = [p for p, v in pairs], [v for p, v in pairs]
    if peak_indices: plt.plot(time_axis[peak_indices], signal[peak_indices], 'o', color='red', label='J峰')
    if valley_indices: plt.plot(time_axis[valley_indices], signal[valley_indices], 'x', color='purple', label='谷点')
    plt.title(title);
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度');
    plt.legend();
    plt.grid(True)


# ===================================================================
# 8. 主程序 (核心修改部分)
# ===================================================================
def main():
    # --- 1. 加载数据 ---
    synthetic_beddot, synthetic_bcg_full = load_training_data()
    test_g, test_p = load_testing_data()
    if synthetic_beddot is None or test_g is None: print("程序中止: 数据加载失败。"); return

    # --- 2. 准备训练数据 ---
    synthetic_bcg_filtered = [highpass_filter(s) for s in synthetic_bcg_full]
    # 【核心修改】is_train=True 将自动启用在Dataset类中定义的噪声和幅度增强
    train_dataset = SignalDataset_NoNorm(synthetic_beddot, synthetic_bcg_filtered, segment_length=SEGMENT_LENGTH,
                                         is_train=True)
    print("\n训练数据集已启用【噪声增强】和【幅度增强】。")

    # --- 3. 训练模型 ---UNet1D/LSTMCNN
    model_type_to_train = 'LSTMCNN'
    model = train_model(train_dataset, model_type=model_type_to_train)
    if model is None: print("模型训练失败，程序中止。"); return

    # --- 4. 在测试集上进行端到端评估 ---
    raw_reconstructed_test = reconstruct_signals(model, test_p)
    final_reconstructed_test = []
    for i in range(len(raw_reconstructed_test)):
        resp = extract_respiration(test_g[i])
        hbeat = raw_reconstructed_test[i]
        min_len = min(len(resp), len(hbeat))
        final_reconstructed_test.append(hbeat[:min_len] + resp[:min_len])

    # --- 5. 性能评估与可视化 ---
    print("\n--- 测试集性能评估 ---")
    overall_metrics = calculate_overall_signal_metrics(test_g, final_reconstructed_test)
    print_metrics_dict(f"整体信号 ({model_type_to_train})", overall_metrics)

    test_g_filtered = [highpass_filter(s) for s in test_g]
    feature_metrics = evaluate_feature_metrics(test_g_filtered, raw_reconstructed_test)
    print_metrics_dict(f"心脏特征 ({model_type_to_train})", feature_metrics)

    # --- 测试集可视化 ---
    if test_g:
        idx = 0
        plot_comparison(test_g, test_p, final_reconstructed_test, idx=idx, fs=FS,
                        suffix=f" (测试集, {model_type_to_train}, 端到端, MSE损失)")

        # 调整图形尺寸以容纳三个子图
        plt.figure(figsize=(18, 15))

        # 子图1: 真实的滤波后BCG信号及其峰谷点
        plt.subplot(3, 1, 1)
        plot_peak_valley(test_g_filtered[idx], find_main_peak_valley_pairs(test_g_filtered[idx]),
                         f'真实滤波ABP (测试集样本 {idx})', 'blue', FS)

        # 子图2: 重建的滤波后BCG信号及其峰谷点
        plt.subplot(3, 1, 2)
        plot_peak_valley(raw_reconstructed_test[idx], find_main_peak_valley_pairs(raw_reconstructed_test[idx]),
                         f'重建滤波ABP (测试集, {model_type_to_train})', 'orange', FS)

        # 子图3: 详细的输入信号(衰减信号)波形
        plt.subplot(3, 1, 3)
        min_len = len(test_p[idx])
        time_axis = np.arange(min_len) / FS
        plt.plot(time_axis, test_p[idx], label='衰减信号', color='green', alpha=0.8)
        plt.title(f'衰减信号详细波形 (测试集样本 {idx})')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True)

        plt.tight_layout(pad=3.0)
        plt.show()

    # --- 训练集可视化 ---
    if synthetic_beddot:
        print("\n--- 训练集信号可视化 ---")
        train_idx = 0  # 选择训练集中的第一个样本进行可视化

        train_input_signal = synthetic_beddot[train_idx]
        train_target_signal = synthetic_bcg_full[train_idx]

        reconstructed_train_signal_raw = reconstruct_long_signal(model, train_input_signal)

        train_resp = extract_respiration(train_target_signal)
        min_len_train = min(len(train_resp), len(reconstructed_train_signal_raw))
        final_reconstructed_train = reconstructed_train_signal_raw[:min_len_train] + train_resp[:min_len_train]

        # 核心修改：将单个信号包裹在列表中，以符合 plot_comparison 的预期
        plot_comparison([train_target_signal], [train_input_signal], [final_reconstructed_train], idx=0, fs=FS,
                        suffix=f" (训练集, {model_type_to_train}, 端到端, 原始输入无噪声)")

        plt.figure(figsize=(18, 15))

        # 子图1: 真实的滤波后BCG信号及其峰谷点 (训练集)
        plt.subplot(3, 1, 1)
        # 对训练集的目标信号进行高通滤波以提取BCG部分
        train_target_filtered = highpass_filter(train_target_signal)
        plot_peak_valley(train_target_filtered, find_main_peak_valley_pairs(train_target_filtered),
                         f'真实滤波ABP (训练集样本 {train_idx})', 'blue', FS)

        # 子图2: 重建的滤波后BCG信号及其峰谷点 (训练集)
        plt.subplot(3, 1, 2)
        plot_peak_valley(reconstructed_train_signal_raw, find_main_peak_valley_pairs(reconstructed_train_signal_raw),
                         f'重建滤波ABP (训练集, {model_type_to_train})', 'orange', FS)

        # 子图3: 详细的输入信号(衰减信号)波形 (训练集)
        plt.subplot(3, 1, 3)
        min_len_input_train = len(train_input_signal)
        time_axis_input_train = np.arange(min_len_input_train) / FS
        plt.plot(time_axis_input_train, train_input_signal, label='衰减信号', color='green', alpha=0.8)
        plt.title(f'衰减信号详细波形 (训练集样本 {train_idx})')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True)

        plt.tight_layout(pad=3.0)
        plt.show()


if __name__ == "__main__":
    main()
