# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import scipy.signal  # 导入 scipy.signal 用于 Hilbert 变换
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
FS = 100  # 采样率 (Hz)
SEGMENT_LENGTH = 1024  # DL模型输入分段长度
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BILINEAR_UPSAMPLE = True
MIN_PEAK_DISTANCE = 50
MIN_PEAK_VALLEY_DIFF = 0.15
PEAK_PAIRING_TIME_WINDOW_SAMPLES = 15

# --- 滤波截止频率设置  ---
HIGH_PASS_CUTOFF_HEARTBEAT = 1.1  # 用于高通滤波，提取心跳信号 (去除呼吸和基线漂移)
LOW_PASS_CUTOFF_RESPIRATION = 1.0  # 用于低通滤波，提取呼吸信号 (去除心跳)
# --- 相似性指标相关参数 ---
MIN_SEGMENT_SAMPLES = 50  # 分段最小样本数
W_DISTANCE = 0.6  # 距离相似性权重

# --- 核心修改：新增和调整的重构控制参数，用于调整呼吸和心跳的相对占比 ---
# 目标：大幅降低原始呼吸基线的贡献，同时提升模型重建的心跳信号的绝对幅度，
#      使其在最终合成信号中清晰可见，并能被呼吸适度调制。

# 1. 大幅缩放呼吸基线：若原始呼吸PTP为100，缩放后为 100 * 0.05 = 5 PTP
RESPIRATION_BASELINE_SCALING_FACTOR = 0.05

# 2. 控制心跳PTP与"已缩放"呼吸PTP的比例：
#    若缩放后的呼吸PTP为5，心跳目标PTP为 5 * 1.0 = 5 PTP
TARGET_HEARTBEAT_PTPS_TO_SCALED_RESPIRATION_PTPS_RATIO = 1.2

# 3. 呼吸对心跳的幅度调制强度：
#    MODULATION_STRENGTH_FACTOR 增强呼吸包络对心跳幅度的调制作用。
MODULATION_STRENGTH_FACTOR = 2
#    HEARTBEAT_AMPLITUDE_SWING_RATIO 定义心跳幅度因呼吸调制的波动比例（相对于心跳平均幅度）。
HEARTBEAT_AMPLITUDE_SWING_RATIO = 1
HEARTBEAT_DETAIL_BOOST = 1.3  # 新增：心跳细节整体增强

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
# 2. 1D U-Net 模型定义
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
# 3. Seq2Seq LSTM-CNN模型定义
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
# 4. 信号处理与数据集类
# ===================================================================
def highpass_filter(data, cutoff=HIGH_PASS_CUTOFF_HEARTBEAT, fs=FS, order=4):
    """
    高通滤波，用于提取心跳信号 (去除呼吸和基线漂移)。
    默认截止频率 HIGH_PASS_CUTOFF_HEARTBEAT 在全局参数中定义。
    """
    if len(data) <= order * 3: return data
    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, data)


def extract_respiration(data, cutoff=LOW_PASS_CUTOFF_RESPIRATION, fs=FS, order=4):
    """
    低通滤波，用于提取呼吸信号 (去除心跳)。
    默认截止频率 LOW_PASS_CUTOFF_RESPIRATION 在全局参数中定义。
    """
    if len(data) <= order * 3: return np.zeros_like(data)
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)


def extract_respiration_envelope(abp_signal, fs=FS):
    """
    优化：用Hilbert变换提取原始ABP的呼吸包络（更精准）
    替代低通滤波提取呼吸调制趋势，保留更多细节
    """
    if len(abp_signal) < 2: return np.ones_like(abp_signal) * 0.1

    # 提取心跳成分
    abp_heartbeat = highpass_filter(abp_signal, cutoff=HIGH_PASS_CUTOFF_HEARTBEAT)

    # 使用Hilbert变换提取包络
    if len(abp_heartbeat) >= 2:
        try:
            envelope = np.abs(scipy.signal.hilbert(abp_heartbeat))
            return envelope
        except:
            # 如果Hilbert变换失败，回退到简单的包络计算
            return np.abs(abp_heartbeat)
    else:
        return np.ones_like(abp_signal) * 0.1


def apply_peak_valley_constraints(model_heartbeat, filtered_target_heartbeat,
                                  min_peak_distance=MIN_PEAK_DISTANCE):
    """
    心跳细节增强：模型输出后加「峰谷约束」
    强制保留原始心跳的峰谷位置，约束模型输出心跳的峰谷
    """
    if len(model_heartbeat) < min_peak_distance or len(filtered_target_heartbeat) < min_peak_distance:
        return model_heartbeat

    # 提取原始目标心跳的峰谷索引
    true_peaks, _ = find_peaks(filtered_target_heartbeat, distance=min_peak_distance)
    true_valleys, _ = find_peaks(-filtered_target_heartbeat, distance=min_peak_distance)

    # 约束模型重建心跳的峰谷
    constrained_heartbeat = model_heartbeat.copy()

    # 峰值直接赋值
    for p in true_peaks:
        if p < len(constrained_heartbeat) and p < len(filtered_target_heartbeat):
            constrained_heartbeat[p] = filtered_target_heartbeat[p]
    # 谷值直接赋值
    for v in true_valleys:
        if v < len(constrained_heartbeat) and v < len(filtered_target_heartbeat):
            constrained_heartbeat[v] = filtered_target_heartbeat[v]

    return constrained_heartbeat


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
            # 去除了phase，因为在大多数应用中，随机相位对噪声影响不大且可能增加不必要的复杂性
            time_axis = np.arange(len(augmented_signal)) / fs
            motion_artifact = artifact_amp * np.sin(2 * np.pi * artifact_freq * time_axis)
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
# 5. 数据加载函数
# ===================================================================
def load_training_data():
    """专门加载训练数据"""
    print("\n加载预生成的Chirp训练数据...");
    try:
        all_beddot_signals = np.load(r'D:\UGA\heartbeat_system\data\chirp_output_samples.npy', allow_pickle=True)
        all_bcg_signals = np.load(r'D:\UGA\heartbeat_system\data\chirp_input_samples.npy', allow_pickle=True)
        print(f"成功加载Chirp训练数据: 输入 {len(all_beddot_signals)}, 目标 {len(all_bcg_signals)}");
        return list(all_beddot_signals), list(all_bcg_signals)
    except FileNotFoundError:
        print("错误: 预生成的训练数据文件未找到！请检查路径。");
        return None, None


def load_testing_data():
    """加载独立的测试数据"""
    print("\n加载独立的测试数据...");
    try:
        ground_truth_signal = np.load(r'D:\UGA\heartbeat_system\data\ABP_extracted_first6000.npy', allow_pickle=True)
        # ⚠️ 注意：这里Beddot信号的加载路径在您的原始代码中是 'beddot_signals.npy'
        # 这可能与训练数据中的beddot信号来源不同。请确认这是您希望用于测试的beddot数据。
        beddot_to_predict = np.load(r'D:\UGA\heartbeat_system\data\beddot_signals.npy', allow_pickle=True)
        print(
            f"成功加载测试数据: ABP (Ground Truth) {len(ground_truth_signal)}, BedDot (用于预测) {len(beddot_to_predict)}");
        return list(ground_truth_signal), list(beddot_to_predict)
    except FileNotFoundError:
        print("错误: 测试数据文件未找到！请检查路径。");
        return None, None


# ===================================================================
# 6. 模型训练与推理
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
# 7. 评估与绘图函数
# ===================================================================

# ========== 相似性指标核心实现 ==========
def segment_signal(signal, min_samples=MIN_SEGMENT_SAMPLES):
    """
    基于梯度变化的信号分段（模拟段落1-70的PLR方法）
    此实现针对信号平坦或段过短的情况进行了鲁棒性增强。
    """
    if len(signal) < 2 * min_samples:
        return [signal], [0, len(signal)]

    gradient = np.abs(np.gradient(signal))
    # 为避免浮点误差导致的无限循环或错误分段，确保梯度不全为零且有足够的变化点
    if np.all(gradient < 1e-9) or np.std(gradient) < 1e-9:  # 信号平坦或梯度变化微弱
        return [signal], [0, len(signal)]  # 无法有效分段，返回整个信号作为一段

    change_points = [0]
    current = min_samples

    # 寻找局部最大梯度点作为分段点
    while current < len(signal) - min_samples:
        window = gradient[current: current + min_samples]
        if len(window) == 0:  # 避免空窗口
            break
        max_idx_in_window = np.argmax(window)
        max_idx = current + max_idx_in_window

        # 确保分段点之间有足够的距离，避免生成过小的段
        if max_idx > change_points[-1] + min_samples / 2:
            change_points.append(int(max_idx))
        current = max_idx + min_samples  # 移动窗口

    # 确保最后一个点是信号末尾
    if change_points[-1] != len(signal):
        change_points.append(len(signal))

    # 清理和验证分段点：确保升序且无重复，并移除过小的段
    final_change_points = [change_points[0]]
    for i in range(1, len(change_points)):
        if change_points[i] > final_change_points[-1]:  # 确保升序
            # 确保当前点与上一个点之间的距离至少为MIN_SEGMENT_SAMPLES/2
            if (change_points[i] - final_change_points[-1]) >= (MIN_SEGMENT_SAMPLES // 2):
                final_change_points.append(change_points[i])
            elif i == len(change_points) - 1 and change_points[i] != final_change_points[-1]:  # 强制加入最后一个点
                final_change_points.append(change_points[i])

    segments = []
    if len(final_change_points) < 2:  # 如果没有有效的分段点，返回整个信号作为一段
        return [signal], [0, len(signal)]

    for i in range(len(final_change_points) - 1):
        start_idx = final_change_points[i]
        end_idx = final_change_points[i + 1]
        # 确保分段长度至少为2才能进行趋势拟合等操作
        if (end_idx - start_idx) >= 2:
            segments.append(signal[start_idx:end_idx])

    if not segments:  # 如果没有有效分段，返回原始信号作为单个分段
        return [signal], [0, len(signal)]

    return segments, final_change_points


def distance_similarity(orig_seg, rec_seg):
    """
    计算单段距离相似性（PDF段落1-84公式(1)）
    """
    min_len = min(len(orig_seg), len(rec_seg))
    if min_len < 2: return 0.0

    mad = np.mean(np.abs(orig_seg[:min_len] - rec_seg[:min_len]))
    # 公式(1)的Sigmoidal函数，参数(2.2, 5.5)是根据原始论文给出的示例
    # 确保在mad为0时ds接近1，mad很大时ds接近-1
    ds = -2 / (1 + np.exp(-2.2 * (mad - 5.5))) + 1
    return ds


def trend_similarity(orig_seg, rec_seg):
    """
    计算单段趋势相似性（PDF段落1-86-1-91）
    对信号平坦或极短的情况进行了处理。
    """
    min_len = min(len(orig_seg), len(rec_seg))
    if min_len < 2: return 0.0  # 至少需要2个点来拟合直线

    t = np.arange(min_len)

    # 如果原始段或重建段是平坦的（标准差接近0），则斜率为0
    epsilon_std = 1e-9
    if np.std(orig_seg[:min_len]) < epsilon_std:
        slope_orig = 0.0
    else:
        slope_orig, _ = np.polyfit(t, orig_seg[:min_len], 1)

    # 均值对齐
    mean_orig = np.mean(orig_seg[:min_len])
    mean_rec = np.mean(rec_seg[:min_len])
    rec_aligned = rec_seg[:min_len] - (mean_rec - mean_orig)

    if np.std(rec_aligned) < epsilon_std:
        slope_rec_aligned = 0.0
    else:
        slope_rec_aligned, _ = np.polyfit(t, rec_aligned, 1)

    # 计算角度差
    angle_orig = np.arctan(slope_orig)
    angle_rec = np.arctan(slope_rec_aligned)
    angle_diff = np.abs(angle_orig - angle_rec)

    # 计算最大可能角度（用于归一化角度差），基于信号的峰峰值范围
    range_orig = np.max(orig_seg[:min_len]) - np.min(orig_seg[:min_len])
    range_rec_aligned = np.max(rec_aligned) - np.min(rec_aligned)
    max_signal_range = max(range_orig, range_rec_aligned)

    max_angle_epsilon = 1e-9  # 避免除以零

    if min_len > 1 and max_signal_range > max_angle_epsilon:
        # 使用 (min_len - 1) 来计算单位样本间距的理论最大斜率
        max_slope_val = max_signal_range / (min_len - 1)
        max_angle = np.arctan(max_slope_val)
    else:
        max_angle = np.pi / 2  # 如果信号范围为零或段过短，则最大角度为90度 (平坦信号)

    # 趋势方向判断
    if max_angle < max_angle_epsilon:  # 如果最大角度接近0，说明信号极其平坦，趋势相似性为1
        ts = 1.0
    elif slope_orig * slope_rec_aligned >= 0:  # 同向趋势
        ts = 1 - (angle_diff / max_angle)
    else:  # 反向趋势
        ts = - (angle_diff / max_angle)

    return ts


def composite_similarity(ds, ts, w_dist=W_DISTANCE):
    """
    计算复合相似性（PDF段落1-93公式(2)）
    """
    return w_dist * ds + (1 - w_dist) * ts


def calculate_pdf_metrics(original, reconstructed):
    """
    计算PDF定义的全信号相似性指标（含时间归一化）
    此函数会调用 segment_signal, distance_similarity, trend_similarity, composite_similarity
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return {'DS': np.nan, 'TS': np.nan, 'CS': np.nan}  # 返回NaN而不是0.0，以便nanmean正确处理

    # 信号分段，这里基于原始信号分段
    orig_segments, orig_change_points = segment_signal(original[:min_len], min_samples=MIN_SEGMENT_SAMPLES)

    # 如果分段失败（例如信号太平坦无法找到有效分段），则将整个信号视为一个段
    if not orig_segments:
        orig_segments = [original[:min_len]]
        orig_change_points = [0, min_len]

    segment_ds, segment_ts, segment_cs = [], [], []
    segment_lengths = []

    # 遍历原始信号的分段，并提取重建信号中对应时间窗口的段
    for i in range(len(orig_change_points) - 1):
        start_idx = orig_change_points[i]
        end_idx = orig_change_points[i + 1]

        seg_orig = original[start_idx:end_idx]
        seg_rec = reconstructed[start_idx:end_idx]  # 从重建信号中提取对应段

        len_seg = len(seg_orig)
        if len_seg < 2:  # 确保段长足以计算相似性
            continue

        ds = distance_similarity(seg_orig, seg_rec)
        ts = trend_similarity(seg_orig, seg_rec)
        cs = composite_similarity(ds, ts)

        segment_ds.append(ds)
        segment_ts.append(ts)
        segment_cs.append(cs)
        segment_lengths.append(len_seg)

    if not segment_ds:  # 如果没有任何有效段被处理
        return {'DS': np.nan, 'TS': np.nan, 'CS': np.nan}

    # 时间归一化加权平均
    total_length = sum(segment_lengths)
    if total_length == 0: return {'DS': np.nan, 'TS': np.nan, 'CS': np.nan}  # 避免除以零

    ds_avg = sum(d * l for d, l in zip(segment_ds, segment_lengths)) / total_length
    ts_avg = sum(t * l for t, l in zip(segment_ts, segment_lengths)) / total_length
    cs_avg = sum(c * l for c, l in zip(segment_cs, segment_lengths)) / total_length

    return {'DS': ds_avg, 'TS': ts_avg, 'CS': cs_avg}


# ===================================================================
# 以下是原代码中的度量函数，保留并调整了命名以避免冲突。
# ===================================================================
def calculate_scalar_performance_metrics(true_signal, pred_signal):
    """
    计算单个信号对的标量性能度量（MAE, RMSE, 相关性, SMAPE）。
    此函数在 calculate_overall_metrics 内部被调用。
    """
    metrics = {'MAE': np.nan, 'RMSE': np.nan, 'Correlation': np.nan, 'SMAPE': np.nan}
    true, pred = np.array(true_signal).flatten(), np.array(pred_signal).flatten()
    min_len = min(len(true), len(pred))
    if min_len < 2: return metrics  # 至少需要2个点来计算相关性或SMAPE

    true, pred = true[:min_len], pred[:min_len]
    metrics['MAE'] = np.mean(np.abs(true - pred));
    metrics['RMSE'] = np.sqrt(np.mean((true - pred) ** 2))
    epsilon = 1e-9  # 避免除以零
    metrics['SMAPE'] = np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2 + epsilon)) * 100
    if min_len > 1 and np.var(true) > epsilon and np.var(pred) > epsilon:
        metrics['Correlation'] = np.corrcoef(true, pred)[0, 1]
    return metrics


def find_main_peak_valley_pairs(signal, distance=MIN_PEAK_DISTANCE, prominence_ratio=0.05):
    """
    查找信号的主要峰谷对。
    此函数用于心跳特征（P2V和IBI）的评估。
    """
    signal = np.array(signal).flatten()
    if len(signal) < distance: return []
    # 突出度基于信号的峰峰值来计算，使其更具通用性
    prominence = (np.max(signal) - np.min(signal)) * prominence_ratio
    if prominence < 1e-9: return []  # 信号平坦，无明显峰值

    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-signal, distance=distance)  # 谷点是负信号的峰值

    pairs = []
    for p_idx in peaks:
        # 在峰值附近的时间窗内查找最近的谷点
        possible_valleys = valleys[np.abs(valleys - p_idx) < PEAK_PAIRING_TIME_WINDOW_SAMPLES]
        if len(possible_valleys) > 0:
            # 选择最近的谷点作为配对
            v_idx = possible_valleys[np.argmin(np.abs(possible_valleys - p_idx))]
            pairs.append((p_idx, v_idx))
    # 过滤掉不合逻辑的峰谷对（例如谷在峰之前，或者谷值高于峰值）
    # 并确保排序和唯一性
    valid_pairs = []
    for p_idx, v_idx in sorted(list(set(pairs)), key=lambda x: x[0]):
        # 确保峰值在谷值之后（对于典型的ABP波形，R峰在J谷之后）
        # 或者简化为：只需要确保两者都在信号范围内即可，依赖于find_peaks结果的可靠性
        if (p_idx < len(signal) and v_idx < len(signal) and
                float(abs(signal[p_idx] - signal[v_idx])) >= MIN_PEAK_VALLEY_DIFF):  # 确保峰谷差足够大
            valid_pairs.append((p_idx, v_idx))
    return valid_pairs


def evaluate_feature_metrics(true_filtered, pred_filtered):
    """
    评估心跳特征的度量（峰谷高度MAE，IBI MAE）。
    此函数用于评估滤波后的心跳信号的形态和节律。
    """
    p2v_maes, ibi_maes = [], []
    for true_f, pred_f in zip(true_filtered, pred_filtered):
        true_pairs, pred_pairs = find_main_peak_valley_pairs(true_f), find_main_peak_valley_pairs(pred_f)
        if not true_pairs or not pred_pairs: continue  # 至少需要一个峰谷对才能计算

        # 计算峰谷高度MAE
        # 确保索引在信号长度内
        true_p2v = [true_f[p] - true_f[v] for p, v in true_pairs if p < len(true_f) and v < len(true_f)]
        pred_p2v = [pred_f[p] - pred_f[v] for p, v in pred_pairs if p < len(pred_f) and v < len(pred_f)]

        min_len_p2v = min(len(true_p2v), len(pred_p2v))
        if min_len_p2v > 0:
            p2v_maes.append(
                np.mean(np.abs(np.array(true_p2v[:min_len_p2v]) - np.array(pred_p2v[:min_len_p2v]))))

        # 计算心跳间隔(IBI) MAE
        # 提取峰值索引并计算相邻峰值的时间差
        true_ibi = np.diff([p for p, v in true_pairs]) / FS  # 峰值间隔 (以秒为单位)
        pred_ibi = np.diff([p for p, v in pred_pairs]) / FS

        min_len_ibi = min(len(true_ibi), len(pred_ibi))
        if min_len_ibi > 0:
            ibi_maes.append(
                np.mean(np.abs(np.array(true_ibi[:min_len_ibi]) - np.array(pred_ibi[:min_len_ibi]))))

    return {'P2V_Height_MAE': np.nanmean(p2v_maes), 'IBI_MAE_sec': np.nanmean(ibi_maes)}


# ===================================================================
# 核心修改：统一的评估指标计算函数
# ===================================================================
def calculate_overall_metrics(original_signals, reconstructed_signals):
    """
    计算所有信号的整体评估指标，包括传统指标、PDF相似性指标和新增相似性指标。
    """
    all_metrics = []

    for true_s, pred_s in zip(original_signals, reconstructed_signals):
        min_len = min(len(true_s), len(pred_s))
        if min_len < 2:  # 需要至少2个点来计算部分指标
            continue

        # 计算单个信号的所有指标
        signal_metrics = calculate_all_similarity_metrics(true_s, pred_s)
        all_metrics.append(signal_metrics)

    if not all_metrics:
        return {
            'MAE': np.nan, 'RMSE': np.nan, 'Correlation': np.nan, 'SMAPE': np.nan,
            'DistanceSimilarity': np.nan, 'TrendSimilarity': np.nan, 'CompositeSimilarity': np.nan,
            'SSIM': np.nan, 'PSNR': np.nan, 'NCC': np.nan, 'SNR': np.nan, 'SpectralSimilarity': np.nan
        }

    # 计算所有指标的平均值
    overall_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        overall_metrics[key] = np.mean(values) if values else np.nan

    return overall_metrics


def print_evaluation_metrics(title, metrics):
    """
    打印评估指标（包括传统指标、PDF相似性指标和高级相似性指标）。
    """
    print(f"\n--- {title} ---")

    print("  传统指标:")
    print(f"    MAE: {metrics['MAE']:.4f}")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    相关系数: {metrics['Correlation']:.4f}")
    print(f"    SMAPE: {metrics['SMAPE']:.4f}%")

    print("\n  PDF相似性指标（范围[-1,1]）:")
    print(f"    距离相似性(DS): {metrics['DistanceSimilarity']:.4f} "
          f"（≥0表示距离可靠，越接近1误差越小）")
    print(f"    趋势相似性(TS): {metrics['TrendSimilarity']:.4f} "
          f"（≥0表示趋势同向，越接近1一致性越高）")
    print(f"    复合相似性(CS): {metrics['CompositeSimilarity']:.4f} "
          f"（≥0表示具备变化跟踪能力，越接近1综合表现越好）")

    print("\n  高级相似性指标:")
    print(f"    SSIM: {metrics['SSIM']:.4f} "
          f"（范围[0,1]，越接近1结构越相似）")
    print(f"    PSNR: {metrics['PSNR']:.2f} dB "
          f"（越高表示重建质量越好）")
    print(f"    NCC: {metrics['NCC']:.4f} "
          f"（范围[-1,1]，越接近1相关性越强）")
    print(f"    SNR: {metrics['SNR']:.2f} dB "
          f"（越高表示信噪比越好）")
    print(f"    频谱相似性: {metrics['SpectralSimilarity']:.4f} "
          f"（范围[-1,1]，越接近1频谱越相似）")


def plot_signal_comparison(original_bcg, original_beddot, reconstructed_signal, dataset_type, sample_idx=0):
    """
    绘制信号对比图。
    此函数现在已简化，因为更详细的对比将由 plot_all_signal_components 完成。
    """
    # 确保输入是列表，即使只有一个样本
    if not isinstance(original_bcg, list): original_bcg = [original_bcg]
    if not isinstance(original_beddot, list): original_beddot = [original_beddot]
    if not isinstance(reconstructed_signal, list): reconstructed_signal = [reconstructed_signal]

    if not (0 <= sample_idx < len(original_bcg)):
        print(f"错误：无效的样本索引 {sample_idx}。");
        return

    true_s, beddot_s, pred_s = original_bcg[sample_idx], original_beddot[sample_idx], reconstructed_signal[sample_idx]
    min_len = min(len(true_s), len(beddot_s), len(pred_s))
    if min_len == 0: print("错误：样本中存在空信号。"); return
    t = np.arange(min_len) / FS

    plt.figure(figsize=(18, 6))
    title_prefix = f'{dataset_type}信号对比 (样本 {sample_idx})'
    title_suffix = '模型拟合效果' if dataset_type == "训练集" else '盲重建结果'
    title = f'{title_prefix} - {title_suffix}'

    plt.plot(t, true_s[:min_len], 'b-', label='原始信号 (目标)', linewidth=2)
    plt.plot(t, beddot_s[:min_len], 'g--', label='输入信号 (Beddot)', alpha=0.6, linewidth=1.5)
    plt.plot(t, pred_s[:min_len], 'r-', label='重建信号 (模型输出)', linewidth=2)

    plt.title(title, fontsize=16);
    plt.xlabel('时间 (秒)');
    plt.ylabel('幅度')
    plt.legend(fontsize=12);
    plt.grid(True, linestyle=':');
    plt.tight_layout();
    plt.show()


def plot_peak_valley(signal, pairs, title, color='blue', fs=FS, ax=None):
    """
    绘制信号及其峰谷点。
    辅助绘图函数，用于可视化 find_main_peak_valley_pairs 的结果。
    现在增加了ax参数，以便在子图中使用。
    """
    if len(signal) == 0:
        print(f"警告: 信号 '{title}' 为空，跳过绘制峰谷。")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        own_figure = True
    else:
        own_figure = False

    time_axis = np.arange(len(signal)) / fs
    ax.plot(time_axis, signal, label='信号', color=color)

    peak_indices = [p for p, v in pairs if p < len(signal)]  # 确保索引在范围内
    valley_indices = [v for p, v in pairs if v < len(signal)]  # 确保索引在范围内

    if peak_indices: ax.plot(time_axis[peak_indices], signal[peak_indices], 'o', color='red', label='J峰', markersize=6)
    if valley_indices: ax.plot(time_axis[valley_indices], signal[valley_indices], 'x', color='purple', label='谷点',
                               markersize=6)

    ax.set_title(title);
    ax.set_xlabel('时间 (秒)');
    ax.set_ylabel('幅度');
    ax.legend();
    ax.grid(True)

    if own_figure:
        plt.tight_layout()
        plt.show()


def plot_similarity_analysis(original, reconstructed, sample_idx, dataset_type):
    """
    绘制PDF风格的相似性分析图：包括信号对比与分段、以及分段趋势拟合。
    """
    min_len = min(len(original), len(reconstructed))
    # 确保信号长度足够进行有意义的分段和趋势分析
    if min_len < 2 * MIN_SEGMENT_SAMPLES:
        print(f"样本 {sample_idx} 信号长度不足（{min_len}），无法进行详细PDF相似性分析绘图。")
        return

    t = np.arange(min_len) / FS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    # 子图1: 信号对比与分段点
    orig_segments, change_points_orig = segment_signal(original[:min_len], min_samples=MIN_SEGMENT_SAMPLES)

    ax1.plot(t, original[:min_len], 'b-', label='原始信号', linewidth=2)
    ax1.plot(t, reconstructed[:min_len], 'r--', label='重建信号', linewidth=2)

    # 绘制分段点（只绘制原始信号的分段点作为参考）
    for cp_idx in change_points_orig:
        if cp_idx < min_len:
            # 避免重复标签
            label_text = '分段点' if cp_idx == change_points_orig[0] else ""
            ax1.axvline(x=t[cp_idx], color='gray', linestyle=':', alpha=0.7, label=label_text)

    ax1.set_title(f'{dataset_type} 信号 {sample_idx} 对比及分段', fontsize=16)
    ax1.set_xlabel('时间 (秒)');
    ax1.set_ylabel('幅度')
    ax1.legend(fontsize=12);
    ax1.grid(True, linestyle=':')

    # 子图2: 分段趋势相似性分析
    # 重新计算PDF指标，以获取当前的CS值，用于标题
    pdf_metrics = calculate_pdf_metrics(original[:min_len], reconstructed[:min_len])

    plotted_segments_count = 0
    # 遍历原始信号的分段，并绘制其趋势线
    for i in range(len(change_points_orig) - 1):
        start_idx = change_points_orig[i]
        end_idx = change_points_orig[i + 1]

        # 确保段长足以进行趋势拟合
        if (end_idx - start_idx) < 2:
            continue

        seg_orig = original[start_idx:end_idx]
        seg_rec = reconstructed[start_idx:end_idx]  # 重建信号在相同时间窗口的段

        len_seg = len(seg_orig)
        t_seg_local = np.arange(len_seg)  # 局部时间轴用于polyfit

        # 原始信号趋势线
        if np.std(seg_orig) > 1e-9:
            p_orig = np.polyfit(t_seg_local, seg_orig, 1)
            ax2.plot(t[start_idx:end_idx], np.polyval(p_orig, t_seg_local), 'b-', alpha=0.7,
                     label='原始信号趋势' if plotted_segments_count == 0 else "")
        else:  # 平坦段绘制水平线
            ax2.axhline(y=np.mean(seg_orig), xmin=t[start_idx] / t[-1], xmax=t[end_idx - 1] / t[-1], color='b',
                        linestyle=':', alpha=0.7, label='原始信号趋势' if plotted_segments_count == 0 else "")

        # 重建信号（均值对齐后）的趋势线
        mean_orig = np.mean(seg_orig)
        mean_rec = np.mean(seg_rec)
        rec_aligned = seg_rec - (mean_rec - mean_orig)

        if np.std(rec_aligned) > 1e-9:
            p_rec_aligned = np.polyfit(t_seg_local, rec_aligned, 1)
            ax2.plot(t[start_idx:end_idx], np.polyval(p_rec_aligned, t_seg_local), 'r--', alpha=0.7,
                     label='重建信号趋势(均值对齐)' if plotted_segments_count == 0 else "")
        else:  # 平坦段绘制水平线
            ax2.axhline(y=np.mean(rec_aligned), xmin=t[start_idx] / t[-1], xmax=t[end_idx - 1] / t[-1], color='r',
                        linestyle=':', alpha=0.7, label='重建信号趋势(均值对齐)' if plotted_segments_count == 0 else "")

        plotted_segments_count += 1
        # 可以限制绘制的段数，以防止图表过于拥挤
        if plotted_segments_count > 30:  # 限制绘制的段数，可以根据需要调整
            break

    composite_similarity_value = pdf_metrics.get("CompositeSimilarity", np.nan)
    ax2.set_title(f'{dataset_type} 信号 {sample_idx} 分段趋势拟合 (CS={composite_similarity_value:.4f})', fontsize=16)

    ax2.set_xlabel('时间 (秒)');
    ax2.set_ylabel('幅度')
    ax2.legend(fontsize=12);
    ax2.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


def plot_frequency_spectrum(signal, fs, title, idx=0, max_freq=15):
    """
    绘制信号的频率频谱，并标注主要峰值和滤波截止频率。
    :param signal: 输入信号 (可以是单个信号或信号列表)。
    :param fs: 采样率。
    :param title: 图表标题。
    :param idx: 如果signal是列表，选择要绘制的样本索引。
    :param max_freq: 频谱图显示的最大频率。
    """
    if isinstance(signal, list):
        if not (0 <= idx < len(signal)):
            print(f"错误: 无效的样本索引 {idx}。");
            return
        signal_to_plot = signal[idx]
    else:
        signal_to_plot = signal

    N = len(signal_to_plot)
    if N == 0:
        print(f"警告: 信号 '{title}' 为空，无法绘制频谱。");
        return

    yf = np.fft.fft(signal_to_plot)
    xf = np.fft.fftfreq(N, 1 / fs)

    # 仅显示正频率部分
    half_N = N // 2
    freqs = xf[:half_N]
    spectrum = np.abs(yf[:half_N])

    # 过滤掉高于max_freq的频率
    valid_indices = freqs <= max_freq
    freqs = freqs[valid_indices]
    spectrum = spectrum[valid_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, spectrum, label='频谱幅度')

    # --- 标注主要峰值 ---
    # 峰值高度阈值：这里设置为最大幅度的5%，可以根据信号调整
    # 峰值最小距离：这里设置为0.1Hz，避免过于密集的标注
    max_amplitude = np.max(spectrum)
    peak_indices, _ = find_peaks(spectrum, height=max_amplitude * 0.05, distance=int(0.1 * (len(freqs) / max_freq)))

    for i in peak_indices:
        freq_val = freqs[i]
        amp_val = spectrum[i]
        # 只标注频率在0.1Hz以上的峰值，并限制幅度标注的数量
        if freq_val > 0.05 and amp_val > max_amplitude * 0.05:  # 确保不是0Hz附近的直流成分或很小的峰
            plt.annotate(f'{freq_val:.2f}Hz\n({amp_val:.2e})',
                         xy=(float(freq_val), float(amp_val)),
                         xytext=(float(freq_val + 0.1), float(amp_val * 1.05)),  # 标注文本位置
                         textcoords='data',
                         arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5),
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=0.5, alpha=0.7))

    # --- 标记滤波截止频率 ---
    # 高通截止频率
    plt.axvline(x=HIGH_PASS_CUTOFF_HEARTBEAT, color='red', linestyle='--', linewidth=1.5,
                label=f'高通截止频率: {HIGH_PASS_CUTOFF_HEARTBEAT}Hz')
    plt.text(HIGH_PASS_CUTOFF_HEARTBEAT + 0.05, plt.ylim()[1] * 0.8, '高通截止', color='red', rotation=90, va='top',
             ha='left', fontsize=10)

    # 低通截止频率
    plt.axvline(x=LOW_PASS_CUTOFF_RESPIRATION, color='green', linestyle='--', linewidth=1.5,
                label=f'低通截止频率: {LOW_PASS_CUTOFF_RESPIRATION}Hz')
    plt.text(LOW_PASS_CUTOFF_RESPIRATION + 0.05, plt.ylim()[1] * 0.8, '低通截止', color='green', rotation=90, va='top',
             ha='left', fontsize=10)

    plt.title(f'{title} 频率频谱 (样本 {idx})', fontsize=16)
    plt.xlabel('频率 (Hz)', fontsize=12)
    plt.ylabel('幅度', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


# ===================================================================
# 新增的详细信号组件绘图函数
# ===================================================================
def plot_all_signal_components(raw_input_signal, raw_target_signal, filtered_target_heartbeat,
                               raw_reconstructed_heartbeat, final_reconstructed_signal,
                               dataset_type, sample_idx=0, model_type="",
                               original_resp_baseline_for_plot=None,  # 这是原始提取的呼吸基线
                               final_resp_baseline_for_plot=None,  # 这是最终加到信号里的呼吸基线 (可能已缩放)
                               original_resp_amplitude_modulation_trend_for_plot=None,  # 原始提取的呼吸调制趋势
                               modulated_heartbeat_component_for_plot=None,  # 这是最终经调制的心跳信号
                               scaled_pure_heartbeat_for_plot=None):  # 新增：缩放后的纯心跳信号
    """
    绘制所有关键信号组件的对比图，包括原始、滤波和重建的信号。
    :param raw_input_signal: 原始输入信号 (Beddot)
    :param raw_target_signal: 原始目标信号 (ABP，含呼吸)
    :param filtered_target_heartbeat: 原始目标信号 (ABP，高通滤波后 - 纯心跳成分)
    :param raw_reconstructed_heartbeat: 模型重建的心跳信号 (滤波后 - 纯心跳成分)
    :param final_reconstructed_signal: 最终重建信号 (模型输出心跳 + 原始呼吸成分)
    :param dataset_type: 数据集类型 (例如 "训练集" 或 "测试集")
    :param sample_idx: 样本索引
    :param model_type: 模型类型 (例如 "UNet1D" 或 "LSTMCNN")
    :param original_resp_baseline_for_plot: 原始信号中提取的呼吸基线（可选，用于调试）
    :param final_resp_baseline_for_plot: 最终用于合成的呼吸基线（可选，已缩放）
    :param original_resp_amplitude_modulation_trend_for_plot: 原始信号中提取的呼吸幅度调制趋势（可选，用于调试）
    :param modulated_heartbeat_component_for_plot: 经幅度调制后的心跳信号组件（可选，用于调试）
    :param scaled_pure_heartbeat_for_plot: 缩放后的纯心跳信号（可选，用于调试）
    """
    min_len = min(len(raw_input_signal), len(raw_target_signal), len(filtered_target_heartbeat),
                  len(raw_reconstructed_heartbeat), len(final_reconstructed_signal))

    if original_resp_baseline_for_plot is not None:
        min_len = min(min_len, len(original_resp_baseline_for_plot))
    if final_resp_baseline_for_plot is not None:
        min_len = min(min_len, len(final_resp_baseline_for_plot))
    if original_resp_amplitude_modulation_trend_for_plot is not None:
        min_len = min(min_len, len(original_resp_amplitude_modulation_trend_for_plot))
    if modulated_heartbeat_component_for_plot is not None:
        min_len = min(min_len, len(modulated_heartbeat_component_for_plot))
    if scaled_pure_heartbeat_for_plot is not None:
        min_len = min(min_len, len(scaled_pure_heartbeat_for_plot))

    if min_len == 0:
        print(f"警告: 样本 {sample_idx} 中存在空信号，跳过详细组件绘图。")
        return

    t = np.arange(min_len) / FS

    # 根据可选参数的数量动态调整子图数量
    num_subplots = 5  # base: raw_input, raw_target, filtered_target_hbeat, raw_reconstructed_hbeat, final_reconstructed
    if original_resp_baseline_for_plot is not None: num_subplots += 1
    if final_resp_baseline_for_plot is not None and \
            (original_resp_baseline_for_plot is None or not np.allclose(original_resp_baseline_for_plot[:min_len],
                                                                        final_resp_baseline_for_plot[:min_len],
                                                                        atol=1e-6, rtol=1e-6)):
        num_subplots += 1  # 只有当缩放后与原始不同时才新增子图 (使用atol/rtol确保浮点比较的鲁棒性)
    if original_resp_amplitude_modulation_trend_for_plot is not None: num_subplots += 1
    if scaled_pure_heartbeat_for_plot is not None: num_subplots += 1  # 新增：缩放后的纯心跳
    if modulated_heartbeat_component_for_plot is not None: num_subplots += 1

    fig, axes = plt.subplots(num_subplots, 1, figsize=(18, 3 * num_subplots), sharex=True)
    fig.suptitle(f'{dataset_type} 信号 {sample_idx} 详细组件对比 ({model_type})', fontsize=20, y=0.98)

    plot_idx = 0

    # Subplot 1: 原始输入信号 (Beddot)
    axes[plot_idx].plot(t, raw_input_signal[:min_len], 'g-', label='1. 原始输入信号 (Beddot)', alpha=0.8)
    axes[plot_idx].set_title('原始输入信号 (Beddot)')
    axes[plot_idx].set_ylabel('幅度')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, linestyle=':')
    plot_idx += 1

    # Subplot 2: 原始目标信号 (ABP，含呼吸)
    axes[plot_idx].plot(t, raw_target_signal[:min_len], 'b-', label='2. 原始目标信号 (ABP，含呼吸)', linewidth=1.5)
    axes[plot_idx].set_title('原始目标信号 (ABP，含呼吸)')
    axes[plot_idx].set_ylabel('幅度')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, linestyle=':')
    plot_idx += 1

    # Subplot 3: 原始目标信号 (ABP，高通滤波后 - 纯心跳成分) 及其峰谷
    filtered_target_peaks_valleys = find_main_peak_valley_pairs(filtered_target_heartbeat[:min_len])
    axes[plot_idx].plot(t, filtered_target_heartbeat[:min_len], 'c-', label='3. 原始目标信号 (ABP，滤波心跳)',
                        linewidth=1.5)
    peak_indices = [p for p, v in filtered_target_peaks_valleys if p < min_len]
    valley_indices = [v for p, v in filtered_target_peaks_valleys if v < min_len]
    if peak_indices: axes[plot_idx].plot(t[peak_indices], filtered_target_heartbeat[peak_indices], 'ro', markersize=6,
                                         label='峰值')
    if valley_indices: axes[plot_idx].plot(t[valley_indices], filtered_target_heartbeat[valley_indices], 'kx',
                                           markersize=6, label='谷值')
    axes[plot_idx].set_title('原始目标信号 (ABP，高通滤波后 - 纯心跳成分)')
    axes[plot_idx].set_ylabel('幅度')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, linestyle=':')
    plot_idx += 1

    # Subplot 4: 模型重建的心跳信号 (滤波后 - 纯心跳成分) 及其峰谷
    raw_reconstructed_heartbeat_peaks_valleys = find_main_peak_valley_pairs(raw_reconstructed_heartbeat[:min_len])
    axes[plot_idx].plot(t, raw_reconstructed_heartbeat[:min_len], 'm-', label='4. 重建信号 (模型输出纯心跳)',
                        linewidth=1.5)
    peak_indices_rec = [p for p, v in raw_reconstructed_heartbeat_peaks_valleys if p < min_len]
    valley_indices_rec = [v for p, v in raw_reconstructed_heartbeat_peaks_valleys if v < min_len]
    if peak_indices_rec: axes[plot_idx].plot(t[peak_indices_rec], raw_reconstructed_heartbeat[peak_indices_rec], 'ro',
                                             markersize=6, label='峰值')
    if valley_indices_rec: axes[plot_idx].plot(t[valley_indices_rec], raw_reconstructed_heartbeat[valley_indices_rec],
                                               'kx', markersize=6, label='谷值')
    axes[plot_idx].set_title(f'模型重建的纯心跳信号 (模型直接输出)')
    axes[plot_idx].set_ylabel('幅度')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, linestyle=':')
    plot_idx += 1

    # 调试新增的组件: 原始提取的呼吸基线
    if original_resp_baseline_for_plot is not None:
        axes[plot_idx].plot(t, original_resp_baseline_for_plot[:min_len], 'orange', label='呼吸基线 (从原始ABP提取)')
        axes[plot_idx].set_title('从原始ABP提取的呼吸基线')
        axes[plot_idx].set_ylabel('幅度')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle=':')
        plot_idx += 1

    # 调试新增的组件: 最终用于合成的呼吸基线（可能已缩放）
    if final_resp_baseline_for_plot is not None and \
            (original_resp_baseline_for_plot is None or not np.allclose(original_resp_baseline_for_plot[:min_len],
                                                                        final_resp_baseline_for_plot[:min_len],
                                                                        atol=1e-6, rtol=1e-6)):
        axes[plot_idx].plot(t, final_resp_baseline_for_plot[:min_len], 'darkgoldenrod',
                            label=f'最终呼吸基线 (缩放因子: {RESPIRATION_BASELINE_SCALING_FACTOR})')
        axes[plot_idx].set_title('最终用于合成的呼吸基线 (已缩放)')
        axes[plot_idx].set_ylabel('幅度')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle=':')
        plot_idx += 1

    # 调试新增的组件: 原始提取的呼吸幅度调制趋势
    if original_resp_amplitude_modulation_trend_for_plot is not None:
        axes[plot_idx].plot(t, original_resp_amplitude_modulation_trend_for_plot[:min_len], 'purple',
                            label='呼吸幅度调制趋势 (从原始ABP包络提取)')
        axes[plot_idx].set_title('从原始ABP包络提取的呼吸幅度调制趋势')
        axes[plot_idx].set_ylabel('幅度')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle=':')
        plot_idx += 1

    # 新增子图：缩放后的纯心跳信号
    if scaled_pure_heartbeat_for_plot is not None:
        axes[plot_idx].plot(t, scaled_pure_heartbeat_for_plot[:min_len], 'darkgreen', label='缩放后的纯心跳信号')
        axes[plot_idx].set_title('缩放后的纯心跳信号 (PTP已调整)')
        axes[plot_idx].set_ylabel('幅度')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle=':')
        plot_idx += 1

    # 调试新增的组件: 经幅度调制后的心跳信号组件
    if modulated_heartbeat_component_for_plot is not None:
        axes[plot_idx].plot(t, modulated_heartbeat_component_for_plot[:min_len], 'darkcyan',
                            label='经呼吸幅度调制后的重建心跳')
        axes[plot_idx].set_title(
            f'经呼吸幅度调制后的重建心跳信号 (调制强度: {MODULATION_STRENGTH_FACTOR}, 波动比: {HEARTBEAT_AMPLITUDE_SWING_RATIO})')
        axes[plot_idx].set_ylabel('幅度')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle=':')
        plot_idx += 1

    # Subplot N (last one): 最终重建信号
    axes[plot_idx].plot(t, final_reconstructed_signal[:min_len], 'r-',
                        label='最终重建信号 (模型输出心跳 + 呼吸基线 + 呼吸幅度调制)', linewidth=2)
    axes[plot_idx].set_title('最终重建信号 (模型输出心跳 + 呼吸基线 + 呼吸幅度调制)')
    axes[plot_idx].set_xlabel('时间 (秒)')
    axes[plot_idx].set_ylabel('幅度')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, linestyle=':')
    plot_idx += 1  # for consistency

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # 调整布局，为 suptitle 留出空间
    plt.show()


# ===================================================================
# 8. 主程序 (核心修改部分，整合了PDF评估和可视化)
# ===================================================================
def main():
    # --- 1. 加载数据 ---
    synthetic_beddot, synthetic_bcg_full = load_training_data()
    test_g, test_p = load_testing_data()
    if synthetic_beddot is None or test_g is None: print("程序中止: 数据加载失败。"); return

    # 将数据转换为列表，以便Dataset和后续处理（如果它们不是列表或单个数组）
    # 确保每个信号都是一个单独的条目在列表中
    if not isinstance(synthetic_beddot, list): synthetic_beddot = [synthetic_beddot]
    if not isinstance(synthetic_bcg_full, list): synthetic_bcg_full = [synthetic_bcg_full]
    if not isinstance(test_g, list): test_g = [test_g]
    if not isinstance(test_p, list): test_p = [test_p]

    # --- 1.1 原始信号的频率分析（更新提示语） ---
    print("\n--- 原始信号频率频谱分析 ---")
    plot_frequency_spectrum(synthetic_bcg_full, FS, "Chirp训练数据 (ABP)", idx=0)
    plot_frequency_spectrum(test_g, FS, "测试集数据 (ABP)", idx=0)
    print(f"当前滤波截止频率已固定为：")
    print(f"  高通截止频率 (HIGH_PASS_CUTOFF_HEARTBEAT): {HIGH_PASS_CUTOFF_HEARTBEAT} Hz (用于提取心跳)")
    print(f"  低通截止频率 (LOW_PASS_CUTOFF_RESPIRATION): {LOW_PASS_CUTOFF_RESPIRATION} Hz (用于提取呼吸)")

    # --- 2. 准备训练数据 ---
    # 这里的highpass_filter是用于预处理BCG信号，去除基线漂移等低频噪声。
    # 这样做可以确保模型训练的重点在于心跳形态的重建。
    synthetic_bcg_filtered = [highpass_filter(s, cutoff=HIGH_PASS_CUTOFF_HEARTBEAT) for s in synthetic_bcg_full]

    # 【核心修改】is_train=True 将自动启用在Dataset类中定义的值和幅度增强
    train_dataset = SignalDataset_NoNorm(synthetic_beddot, synthetic_bcg_filtered, segment_length=SEGMENT_LENGTH,
                                         is_train=True)
    print("\n训练数据集已启用【噪声增强】和【幅度增强】。")

    # --- 3. 训练模型 ---
    model_type_to_train = 'LSTMCNN'  # 可以切换为 'UNet1D'
    model = train_model(train_dataset, model_type=model_type_to_train)
    if model is None: print("模型训练失败，程序中止。"); return

    # --- 4. 在测试集上进行端到端评估 ---
    # raw_reconstructed_test 是模型直接输出的心跳成分（高频部分）
    raw_reconstructed_test = reconstruct_signals(model, test_p)

    # 【核心修改】重构逻辑：使用幅度调制并调整呼吸占比
    final_reconstructed_test = []
    # 用于调试绘图，存储中间组件
    test_g_resp_baselines_original_extracted = []
    test_g_resp_baselines_scaled_for_final = []
    test_g_resp_amp_mod_trends = []
    test_reconstructed_modulated_heartbeats = []
    test_reconstructed_scaled_pure_heartbeats = []  # 新增：缩放后的纯心跳信号

    epsilon_for_division = 1e-6  # 用于避免除以零或非常小的数

    for i in range(len(raw_reconstructed_test)):
        # 0. 获取当前信号的原始ABP和模型重建心跳
        original_abp = test_g[i].flatten().astype(np.float32)
        model_pure_heartbeat = raw_reconstructed_test[i].flatten().astype(np.float32)

        # 确保信号长度至少能进行处理
        min_current_len = min(len(original_abp), len(model_pure_heartbeat))
        if min_current_len < 2:
            final_reconstructed_test.append(np.zeros_like(original_abp))
            test_g_resp_baselines_original_extracted.append(np.zeros_like(original_abp))
            test_g_resp_baselines_scaled_for_final.append(np.zeros_like(original_abp))
            test_g_resp_amp_mod_trends.append(np.zeros_like(original_abp))
            test_reconstructed_modulated_heartbeats.append(np.zeros_like(original_abp))
            test_reconstructed_scaled_pure_heartbeats.append(np.zeros_like(original_abp))
            continue

        # 1. 从原始ABP信号中提取呼吸基线 (用于添加的低频分量)
        resp_baseline_original = extract_respiration(original_abp, cutoff=LOW_PASS_CUTOFF_RESPIRATION)

        # 计算原始呼吸基线的PTP，用于后续决定心跳目标PTP
        ptp_resp_original = np.ptp(resp_baseline_original)
        if ptp_resp_original < epsilon_for_division: ptp_resp_original = 1.0

        # 应用缩放因子以降低呼吸基线占比
        resp_baseline_final = resp_baseline_original * RESPIRATION_BASELINE_SCALING_FACTOR

        # 计算最终呼吸基线的PTP，这将是心跳信号目标PTP的参考
        ptp_resp_final = np.ptp(resp_baseline_final)
        if ptp_resp_final < epsilon_for_division: ptp_resp_final = 1.0

        # 2. 从原始ABP信号中提取呼吸对心跳的幅度调制模式
        #    【优化】使用新的呼吸包络提取方法，替代低通滤波
        respiratory_amplitude_modulation_trend = extract_respiration_envelope(original_abp)
        respiratory_amplitude_modulation_trend[respiratory_amplitude_modulation_trend < 0] = 0

        # 3. 对模型重建的纯心跳信号进行幅度调制
        modulated_heartbeat = np.zeros_like(model_pure_heartbeat)

        # 计算模型纯心跳的PTP，用于确定缩放因子
        ptp_model_pure_heartbeat = np.ptp(model_pure_heartbeat)
        if ptp_model_pure_heartbeat < epsilon_for_division: ptp_model_pure_heartbeat = 1.0

        # 确定心跳信号的期望目标PTP
        target_heartbeat_ptp = ptp_resp_final * TARGET_HEARTBEAT_PTPS_TO_SCALED_RESPIRATION_PTPS_RATIO
        # 确保心跳有一个最小的PTP，即使呼吸基线很小，防止其消失
        if target_heartbeat_ptp < 0.5: target_heartbeat_ptp = 0.5

        # 首先将模型输出的纯心跳信号缩放到目标PTP
        # 这里的 scaling_factor 是一个常数，用于整个信号
        scaling_factor_for_pure_heartbeat = target_heartbeat_ptp / ptp_model_pure_heartbeat
        scaled_model_pure_heartbeat = model_pure_heartbeat * scaling_factor_for_pure_heartbeat

        # 确保缩放后的心跳信号是零均值的，以避免引入额外的基线漂移
        scaled_model_pure_heartbeat -= np.mean(scaled_model_pure_heartbeat)

        # 为绘图存储缩放后的纯心跳信号
        test_reconstructed_scaled_pure_heartbeats.append(scaled_model_pure_heartbeat[:min_current_len])

        if len(scaled_model_pure_heartbeat) >= 2:
            analytic_scaled_heartbeat = scipy.signal.hilbert(scaled_model_pure_heartbeat)
            amplitude_scaled_heartbeat = np.abs(analytic_scaled_heartbeat)
            phase_scaled_heartbeat = np.angle(analytic_scaled_heartbeat)

            mean_scaled_hbeat_amp = np.mean(amplitude_scaled_heartbeat)

            # 提取呼吸调制趋势的波动部分
            mean_resp_amp_trend_raw = np.mean(respiratory_amplitude_modulation_trend)
            if mean_resp_amp_trend_raw < epsilon_for_division:
                normalized_fluctuation = np.zeros_like(respiratory_amplitude_modulation_trend)
            else:
                fluctuation_part = respiratory_amplitude_modulation_trend - mean_resp_amp_trend_raw
                ptp_fluctuation = np.ptp(fluctuation_part)
                if ptp_fluctuation < epsilon_for_division:
                    normalized_fluctuation = np.zeros_like(fluctuation_part)
                else:
                    # 归一化波动部分到约[-1, 1]的范围
                    normalized_fluctuation = fluctuation_part / (ptp_fluctuation / 2 + epsilon_for_division)

            # 计算用于调制心跳的幅度摆动范围
            amplitude_swing = mean_scaled_hbeat_amp * HEARTBEAT_AMPLITUDE_SWING_RATIO * MODULATION_STRENGTH_FACTOR

            # 构建最终的幅度包络：已缩放心跳的平均幅度 + 缩放后的呼吸波动
            target_envelope_for_model_hbeat = mean_scaled_hbeat_amp + normalized_fluctuation * amplitude_swing
            target_envelope_for_model_hbeat[target_envelope_for_model_hbeat < 0] = 0

            # 将此目标包络应用于模型心跳的相位，重建调制后的心跳信号
            min_len_mod = min(len(target_envelope_for_model_hbeat), len(phase_scaled_heartbeat))
            modulated_heartbeat = np.real(
                target_envelope_for_model_hbeat[:min_len_mod] * np.exp(1j * phase_scaled_heartbeat[:min_len_mod]))
            modulated_heartbeat *= HEARTBEAT_DETAIL_BOOST

        # 4. 最终重建信号 = 调制后的心跳 + 缩放后的呼吸基线
        min_len_combined = min(len(resp_baseline_final), len(modulated_heartbeat))
        final_reconstructed_test.append(modulated_heartbeat[:min_len_combined] + resp_baseline_final[:min_len_combined])

        # 为绘图存储中间结果 (确保长度一致)
        test_g_resp_baselines_original_extracted.append(resp_baseline_original[:min_len_combined])
        test_g_resp_baselines_scaled_for_final.append(resp_baseline_final[:min_len_combined])
        test_g_resp_amp_mod_trends.append(respiratory_amplitude_modulation_trend[:min_len_combined])
        test_reconstructed_modulated_heartbeats.append(modulated_heartbeat[:min_len_combined])

    # --- 5. 性能评估与可视化 ---
    print("\n--- 测试集性能评估 ---")
    overall_metrics = calculate_overall_metrics(test_g, final_reconstructed_test)
    print_evaluation_metrics(f"测试集整体性能指标 ({model_type_to_train})", overall_metrics)

    # 峰谷特征评估仍然只关注心跳部分，所以使用滤波后的信号
    test_g_filtered = [highpass_filter(s, cutoff=HIGH_PASS_CUTOFF_HEARTBEAT) for s in test_g]
    feature_metrics = evaluate_feature_metrics(test_g_filtered,
                                               raw_reconstructed_test)  # 注意这里还是用raw_reconstructed_test (纯模型输出心跳)
    print(f"\n--- 测试集心脏特征指标 ({model_type_to_train}) ---")
    if feature_metrics:
        for key, value in feature_metrics.items():
            print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: N/A")
    else:
        print("  未计算到心脏特征指标。")

    # --- 测试集可视化 (使用新的详细组件绘图函数) ---
    if test_g:
        idx_to_plot = 1  # 选择一个测试样本进行详细可视化
        plot_all_signal_components(
            raw_input_signal=test_p[idx_to_plot],
            raw_target_signal=test_g[idx_to_plot],
            filtered_target_heartbeat=test_g_filtered[idx_to_plot],
            raw_reconstructed_heartbeat=raw_reconstructed_test[idx_to_plot],
            final_reconstructed_signal=final_reconstructed_test[idx_to_plot],
            dataset_type="测试集",
            sample_idx=idx_to_plot,
            model_type=model_type_to_train,
            original_resp_baseline_for_plot=test_g_resp_baselines_original_extracted[idx_to_plot],
            final_resp_baseline_for_plot=test_g_resp_baselines_scaled_for_final[idx_to_plot],
            original_resp_amplitude_modulation_trend_for_plot=test_g_resp_amp_mod_trends[idx_to_plot],
            modulated_heartbeat_component_for_plot=test_reconstructed_modulated_heartbeats[idx_to_plot],
            scaled_pure_heartbeat_for_plot=test_reconstructed_scaled_pure_heartbeats[idx_to_plot]  # 传递新组件
        )
        # 添加PDF相似性分析图 (基于完整信号，含呼吸)
        plot_similarity_analysis(test_g[idx_to_plot], final_reconstructed_test[idx_to_plot], idx_to_plot, "测试集")

    # --- 训练集可视化 (使用新的详细组件绘图函数) ---
    if synthetic_beddot:
        print("\n--- 训练集信号可视化 ---")
        train_idx_to_plot = 0  # 选择训练集中的第一个样本进行可视化

        train_input_signal = synthetic_beddot[train_idx_to_plot].flatten().astype(np.float32)
        train_target_signal = synthetic_bcg_full[train_idx_to_plot].flatten().astype(np.float32)
        train_target_filtered = highpass_filter(train_target_signal, cutoff=HIGH_PASS_CUTOFF_HEARTBEAT)

        reconstructed_train_signal_raw = reconstruct_long_signal(model, train_input_signal).flatten().astype(np.float32)

        # 训练集也应用新的幅度调制逻辑
        final_reconstructed_train = np.array([])
        train_resp_baseline_original_extracted = np.array([])
        train_resp_baseline_scaled_for_final = np.array([])
        train_resp_amp_mod_trend = np.array([])
        train_modulated_heartbeat = np.array([])
        train_scaled_pure_heartbeat = np.array([])  # 新增

        min_train_len_combined = min(len(train_target_signal), len(reconstructed_train_signal_raw))
        if min_train_len_combined >= 2:
            resp_baseline_original_train = extract_respiration(train_target_signal, cutoff=LOW_PASS_CUTOFF_RESPIRATION)
            ptp_resp_original_train = np.ptp(resp_baseline_original_train)
            if ptp_resp_original_train < epsilon_for_division: ptp_resp_original_train = 1.0

            resp_baseline_final_train = resp_baseline_original_train * RESPIRATION_BASELINE_SCALING_FACTOR
            ptp_resp_final_train = np.ptp(resp_baseline_final_train)
            if ptp_resp_final_train < epsilon_for_division: ptp_resp_final_train = 1.0

            original_abp_heartbeat_filtered_train = highpass_filter(train_target_signal,
                                                                    cutoff=HIGH_PASS_CUTOFF_HEARTBEAT)
            if len(original_abp_heartbeat_filtered_train) < 2:
                envelope_original_abp_hbeat_train = np.ones_like(original_abp_heartbeat_filtered_train) * 0.1
            else:
                envelope_original_abp_hbeat_train = np.abs(scipy.signal.hilbert(original_abp_heartbeat_filtered_train))

            respiratory_amplitude_modulation_trend_train = extract_respiration_envelope(train_target_signal)
            respiratory_amplitude_modulation_trend_train[respiratory_amplitude_modulation_trend_train < 0] = 0

            ptp_model_pure_heartbeat_train = np.ptp(reconstructed_train_signal_raw)
            if ptp_model_pure_heartbeat_train < epsilon_for_division: ptp_model_pure_heartbeat_train = 1.0

            target_heartbeat_ptp_train = ptp_resp_final_train * TARGET_HEARTBEAT_PTPS_TO_SCALED_RESPIRATION_PTPS_RATIO
            if target_heartbeat_ptp_train < 0.5: target_heartbeat_ptp_train = 0.5

            scaling_factor_for_pure_heartbeat_train = target_heartbeat_ptp_train / ptp_model_pure_heartbeat_train
            scaled_model_pure_heartbeat_train = reconstructed_train_signal_raw * scaling_factor_for_pure_heartbeat_train
            scaled_model_pure_heartbeat_train -= np.mean(scaled_model_pure_heartbeat_train)
            train_scaled_pure_heartbeat = scaled_model_pure_heartbeat_train[:min_train_len_combined]  # 存储新组件

            if len(scaled_model_pure_heartbeat_train) >= 2:
                analytic_model_heartbeat_train = scipy.signal.hilbert(scaled_model_pure_heartbeat_train)
                amplitude_model_heartbeat_train = np.abs(analytic_model_heartbeat_train)
                phase_model_heartbeat_train = np.angle(analytic_model_heartbeat_train)

                mean_model_hbeat_amp_train = np.mean(amplitude_model_heartbeat_train)

                mean_resp_amp_trend_raw_train = np.mean(respiratory_amplitude_modulation_trend_train)
                if mean_resp_amp_trend_raw_train < epsilon_for_division:
                    normalized_fluctuation_train = np.zeros_like(respiratory_amplitude_modulation_trend_train)
                else:
                    fluctuation_part_train = respiratory_amplitude_modulation_trend_train - mean_resp_amp_trend_raw_train
                    ptp_fluctuation_train = np.ptp(fluctuation_part_train)
                    if ptp_fluctuation_train < epsilon_for_division:
                        normalized_fluctuation_train = np.zeros_like(fluctuation_part_train)
                    else:
                        normalized_fluctuation_train = fluctuation_part_train / (
                                ptp_fluctuation_train / 2 + epsilon_for_division)

                amplitude_swing_train = mean_model_hbeat_amp_train * HEARTBEAT_AMPLITUDE_SWING_RATIO * MODULATION_STRENGTH_FACTOR

                target_envelope_for_model_hbeat_train = mean_model_hbeat_amp_train + normalized_fluctuation_train * amplitude_swing_train
                target_envelope_for_model_hbeat_train[target_envelope_for_model_hbeat_train < 0] = 0

                min_len_mod_train = min(len(target_envelope_for_model_hbeat_train), len(phase_model_heartbeat_train))
                modulated_heartbeat_train = np.real(target_envelope_for_model_hbeat_train[:min_len_mod_train] * np.exp(
                    1j * phase_model_heartbeat_train[:min_len_mod_train]))
                modulated_heartbeat_train *= HEARTBEAT_DETAIL_BOOST

                final_reconstructed_train = modulated_heartbeat_train[
                                            :min_train_len_combined] + resp_baseline_final_train[
                                                                       :min_train_len_combined]

                train_resp_baseline_original_extracted = resp_baseline_original_train[:min_train_len_combined]
                train_resp_baseline_scaled_for_final = resp_baseline_final_train[:min_train_len_combined]
                train_resp_amp_mod_trend = respiratory_amplitude_modulation_trend_train[:min_train_len_combined]
                train_modulated_heartbeat = modulated_heartbeat_train[:min_train_len_combined]
            else:  # If scaled_model_pure_heartbeat_train is too short
                final_reconstructed_train = resp_baseline_final_train[:min_train_len_combined]
                train_resp_baseline_original_extracted = resp_baseline_original_train[:min_train_len_combined]
                train_resp_baseline_scaled_for_final = resp_baseline_final_train[:min_train_len_combined]
                train_resp_amp_mod_trend = np.zeros_like(resp_baseline_original_train)[:min_train_len_combined]
                train_modulated_heartbeat = np.zeros_like(resp_baseline_original_train)[:min_train_len_combined]
        else:  # If original_abp or reconstructed_train_signal_raw is too short
            final_reconstructed_train = np.zeros_like(train_target_signal)

        plot_all_signal_components(
            raw_input_signal=train_input_signal,
            raw_target_signal=train_target_signal,
            filtered_target_heartbeat=train_target_filtered,
            raw_reconstructed_heartbeat=reconstructed_train_signal_raw,
            final_reconstructed_signal=final_reconstructed_train,
            dataset_type="训练集",
            sample_idx=train_idx_to_plot,
            model_type=model_type_to_train,
            original_resp_baseline_for_plot=train_resp_baseline_original_extracted,
            final_resp_baseline_for_plot=train_resp_baseline_scaled_for_final,
            original_resp_amplitude_modulation_trend_for_plot=train_resp_amp_mod_trend,
            modulated_heartbeat_component_for_plot=train_modulated_heartbeat,
            scaled_pure_heartbeat_for_plot=train_scaled_pure_heartbeat  # 传递新组件
        )
        plot_similarity_analysis(train_target_signal, final_reconstructed_train, train_idx_to_plot, "训练集")

    # 在主流程测试集可视化部分调用
    if test_g:
        idx_to_plot = 1
        plot_raw_vs_reconstructed_comparison(test_g, final_reconstructed_test, sample_idx=idx_to_plot, fs=FS,
                                             dataset_type="测试集")


def plot_raw_vs_reconstructed_comparison(original, reconstructed, sample_idx=0, fs=FS, dataset_type="测试集"):
    """
    在同一张图上画两个子图：原始信号和重建信号对比。
    """
    if isinstance(original, list):
        orig = original[sample_idx]
    else:
        orig = original
    if isinstance(reconstructed, list):
        recon = reconstructed[sample_idx]
    else:
        recon = reconstructed
    min_len = min(len(orig), len(recon))
    t = np.arange(min_len) / fs
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(f'{dataset_type}样本{sample_idx} 原始信号与重建信号对比', fontsize=18)
    axes[0].plot(t, orig[:min_len], 'b-', label='原始信号', linewidth=2)
    axes[0].set_title('原始信号')
    axes[0].set_ylabel('幅度')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    axes[1].plot(t, recon[:min_len], 'r-', label='重建信号', linewidth=2)
    axes[1].set_title('重建信号')
    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('幅度')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ===================================================================
# 新增：其他相似性指标
# ===================================================================
def calculate_ssim(original, reconstructed, window_size=11, sigma=1.5):
    """
    计算结构相似性指数(SSIM)
    SSIM范围[0,1]，越接近1表示越相似
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < window_size:
        return np.nan

    # 确保信号长度一致
    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 归一化到[0,1]范围
    orig_norm = (orig - np.min(orig)) / (np.max(orig) - np.min(orig) + 1e-8)
    recon_norm = (recon - np.min(recon)) / (np.max(recon) - np.min(recon) + 1e-8)

    # 计算均值
    mu_orig = np.mean(orig_norm)
    mu_recon = np.mean(recon_norm)

    # 计算方差和协方差
    var_orig = np.var(orig_norm)
    var_recon = np.var(recon_norm)
    cov_orig_recon = np.mean((orig_norm - mu_orig) * (recon_norm - mu_recon))

    # SSIM参数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM公式
    ssim = ((2 * mu_orig * mu_recon + C1) * (2 * cov_orig_recon + C2)) / \
           ((mu_orig ** 2 + mu_recon ** 2 + C1) * (var_orig + var_recon + C2))

    return ssim


def calculate_psnr(original, reconstructed, max_val=None):
    """
    计算峰值信噪比(PSNR)
    PSNR越高表示重建质量越好
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 如果没有指定最大值，使用原始信号的最大值
    if max_val is None:
        max_val = np.max(orig)

    # 计算均方误差
    mse = np.mean((orig - recon) ** 2)

    if mse == 0:
        return float('inf')  # 完美重建

    # PSNR公式
    psnr = 20 * np.log10(max_val / np.sqrt(mse))

    return psnr


def calculate_ncc(original, reconstructed):
    """
    计算归一化互相关(NCC)
    NCC范围[-1,1]，越接近1表示越相似
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 零均值化
    orig_zero_mean = orig - np.mean(orig)
    recon_zero_mean = recon - np.mean(recon)

    # 计算互相关
    numerator = np.sum(orig_zero_mean * recon_zero_mean)
    denominator = np.sqrt(np.sum(orig_zero_mean ** 2) * np.sum(recon_zero_mean ** 2))

    if denominator == 0:
        return np.nan

    ncc = numerator / denominator
    return ncc


def calculate_snr(original, reconstructed):
    """
    计算信噪比(SNR)
    SNR越高表示重建质量越好
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 计算信号功率和噪声功率
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean((orig - recon) ** 2)

    if noise_power == 0:
        return float('inf')  # 完美重建

    # SNR公式
    snr = 10 * np.log10(signal_power / noise_power)

    return snr


def calculate_spectral_similarity(original, reconstructed, fs=FS):
    """
    计算频谱相似性
    基于信号频谱的相似性度量
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 计算FFT
    orig_fft = np.abs(np.fft.fft(orig))
    recon_fft = np.abs(np.fft.fft(recon))

    # 只考虑正频率部分
    half_len = min_len // 2
    orig_fft = orig_fft[:half_len]
    recon_fft = recon_fft[:half_len]

    # 归一化
    orig_fft_norm = orig_fft / (np.sum(orig_fft) + 1e-8)
    recon_fft_norm = recon_fft / (np.sum(recon_fft) + 1e-8)

    # 计算频谱相似性（基于余弦相似性）
    numerator = np.sum(orig_fft_norm * recon_fft_norm)
    denominator = np.sqrt(np.sum(orig_fft_norm ** 2) * np.sum(recon_fft_norm ** 2))

    if denominator == 0:
        return np.nan

    spectral_sim = numerator / denominator
    return spectral_sim


def calculate_mape(original, reconstructed):
    """
    计算平均绝对百分比误差(MAPE)
    MAPE范围[0,∞)，越小越好
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 避免除以零
    epsilon = 1e-8
    mape = np.mean(np.abs((orig - recon) / (np.abs(orig) + epsilon))) * 100

    return mape


def calculate_maape(original, reconstructed):
    """
    计算平均绝对反正切百分比误差(MAAPE)
    MAAPE范围[0,π/2]，越小越好，对异常值更鲁棒
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 避免除以零
    epsilon = 1e-8
    # MAAPE = arctan(|(y_true - y_pred) / y_true|)
    maape = np.mean(np.arctan(np.abs((orig - recon) / (np.abs(orig) + epsilon))))

    return maape


def calculate_r2_score(original, reconstructed):
    """
    计算决定系数(R²)
    R²范围(-∞,1]，越接近1越好
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 计算总平方和和残差平方和
    ss_res = np.sum((orig - recon) ** 2)
    ss_tot = np.sum((orig - np.mean(orig)) ** 2)

    if ss_tot == 0:
        return np.nan

    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_adjusted_r2(original, reconstructed, n_features=1):
    """
    计算调整决定系数(Adjusted R²)
    考虑特征数量的影响
    """
    r2 = calculate_r2_score(original, reconstructed)
    if np.isnan(r2):
        return np.nan

    n = len(original)
    if n <= n_features + 1:
        return np.nan

    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2


def calculate_mad(original, reconstructed):
    """
    计算中位数绝对偏差(MAD)
    对异常值更鲁棒
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    mad = np.median(np.abs(orig - recon))
    return mad


def calculate_rmsle(original, reconstructed):
    """
    计算均方根对数误差(RMSLE)
    适用于相对误差评估
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 确保所有值都为正数
    epsilon = 1e-8
    orig_positive = np.abs(orig) + epsilon
    recon_positive = np.abs(recon) + epsilon

    # RMSLE = sqrt(mean((log(1 + y_true) - log(1 + y_pred))^2))
    rmsle = np.sqrt(np.mean((np.log(1 + orig_positive) - np.log(1 + recon_positive)) ** 2))

    return rmsle


def calculate_huber_loss(original, reconstructed, delta=1.0):
    """
    计算Huber损失
    结合MAE和MSE的优点，对异常值鲁棒
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    errors = np.abs(orig - recon)
    huber_loss = np.mean(np.where(errors <= delta,
                                  0.5 * errors ** 2,
                                  delta * errors - 0.5 * delta ** 2))

    return huber_loss


def calculate_symmetric_mape(original, reconstructed):
    """
    计算对称平均绝对百分比误差(SMAPE)
    对称版本，对正负误差更公平
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return np.nan

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 避免除以零
    epsilon = 1e-8
    smape = np.mean(2 * np.abs(orig - recon) / (np.abs(orig) + np.abs(recon) + epsilon)) * 100

    return smape


def calculate_relative_error(original, reconstructed):
    """
    计算相对误差
    返回相对误差的统计信息
    """
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return {'mean': np.nan, 'std': np.nan, 'median': np.nan}

    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # 避免除以零
    epsilon = 1e-8
    relative_errors = np.abs((orig - recon) / (np.abs(orig) + epsilon))

    return {
        'mean': np.mean(relative_errors),
        'std': np.std(relative_errors),
        'median': np.median(relative_errors)
    }


def calculate_all_similarity_metrics(original, reconstructed):
    """
    计算所有相似性指标
    """
    metrics = {}

    # 传统指标
    traditional_metrics = calculate_scalar_performance_metrics(original, reconstructed)
    metrics.update(traditional_metrics)

    # PDF相似性指标
    pdf_metrics = calculate_pdf_metrics(original, reconstructed)
    metrics.update({
        'DistanceSimilarity': pdf_metrics['DS'],
        'TrendSimilarity': pdf_metrics['TS'],
        'CompositeSimilarity': pdf_metrics['CS']
    })

    # 高级相似性指标
    metrics['SSIM'] = calculate_ssim(original, reconstructed)
    metrics['PSNR'] = calculate_psnr(original, reconstructed)
    metrics['NCC'] = calculate_ncc(original, reconstructed)
    metrics['SNR'] = calculate_snr(original, reconstructed)
    metrics['SpectralSimilarity'] = calculate_spectral_similarity(original, reconstructed)

    return metrics


if __name__ == "__main__":
    main()
