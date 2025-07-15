import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt


# ===================== 1. 数据集定义 =====================
class SignalDataset(Dataset):
    def __init__(self, atten_signals, raw_signals, segment_length=1024):
        self.atten_signals = atten_signals
        self.raw_signals = raw_signals
        self.segment_length = segment_length

    def __len__(self):
        return len(self.atten_signals)

    def __getitem__(self, idx):
        x = np.array(self.atten_signals[idx]).astype(np.float32)
        y = np.array(self.raw_signals[idx]).astype(np.float32)
        # 裁剪或填充到固定长度
        if len(x) > self.segment_length:
            x = x[:self.segment_length]
            y = y[:self.segment_length]
        else:
            pad = self.segment_length - len(x)
            x = np.pad(x, (0, pad), 'edge')
            y = np.pad(y, (0, pad), 'edge')
        # 只对输入归一化，输出保持原始幅度
        x_mean, x_std = np.mean(x), np.std(x) + 1e-8
        x_norm = (x - x_mean) / x_std
        return torch.from_numpy(x_norm), torch.from_numpy(y), x_mean, x_std


# ===================== 2. U-Net1D模型 =====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = nn.functional.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x).squeeze(1)


# ===================== 3. 训练与推理 =====================
def correlation_loss(y_pred, y_true):
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return 1 - corr  # 越小越好


def spectral_loss(y_pred, y_true):
    pred_fft = torch.fft.rfft(y_pred, dim=-1)
    true_fft = torch.fft.rfft(y_true, dim=-1)
    return torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(true_fft)))


# 训练时适配新的输出格式
# for batch in train_loader: x, y, x_mean, x_std = batch
# 推理时同样适配

def train_model(model, train_loader, device, epochs=50, lr=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            x, y, _, _ = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = mse_loss(out, y) + 0.2 * correlation_loss(out, y) + 0.3 * spectral_loss(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
    return model


def predict(model, atten_signals, device, segment_length=1024):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(len(atten_signals)):
            x = np.array(atten_signals[i]).astype(np.float32)
            if len(x) > segment_length:
                x = x[:segment_length]
    else:
    x = np.pad(x, (0, segment_length - len(x)), 'edge')


x_mean, x_std = np.mean(x), np.std(x) + 1e-8
x_norm = (x - x_mean) / x_std
x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
out = model(x_tensor)
out_np = out.cpu().numpy().flatten()
results.append(out_np)
return results


# ===================== 4. 数据加载（你的方式） =====================
def load_training_data():
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
    print("\n加载独立的测试数据...");
    try:
        ground_truth_signal = np.load(r'D:\UGA\heartbeat_system\data\ABP_extracted_first6000.npy', allow_pickle=True)
        beddot_to_predict = np.load(r'D:\UGA\heartbeat_system\data\beddot_signals.npy', allow_pickle=True)
        print(
            f"成功加载测试数据: ABP (Ground Truth) {len(ground_truth_signal)}, BedDot (用于预测) {len(beddot_to_predict)}");
        return list(ground_truth_signal), list(beddot_to_predict)
    except FileNotFoundError:
        print("错误: 测试数据文件未找到！请检查路径。");
        return None, None


def evaluate_metrics(true_signals, pred_signals):
    maes, mses, cors = [], [], []
    for t, p in zip(true_signals, pred_signals):
        t, p = np.array(t).flatten(), np.array(p).flatten()
        min_len = min(len(t), len(p))
        t, p = t[:min_len], p[:min_len]
        maes.append(np.mean(np.abs(t - p)))
        mses.append(np.mean((t - p) ** 2))
        if np.std(t) > 1e-6 and np.std(p) > 1e-6:
            cors.append(np.corrcoef(t, p)[0, 1])
    print(f"MAE: {np.mean(maes):.4f}")
    print(f"MSE: {np.mean(mses):.4f}")
    print(f"相关系数: {np.mean(cors):.4f}")


def plot_signals(raw, atten, recon, idx=0, fs=100):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    min_len = min(len(raw[idx]), len(atten[idx]), len(recon[idx]))
    t = np.arange(min_len) / fs
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, raw[idx][:min_len], color='b', label='原始信号', linewidth=2)
    plt.ylabel('幅度')
    plt.title(f'第{idx}个样本-原始信号')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, atten[idx][:min_len], color='g', label='衰减信号', linewidth=2)
    plt.ylabel('幅度')
    plt.title(f'第{idx}个样本-衰减信号')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, recon[idx][:min_len], color='r', label='重建信号', linewidth=2)
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.title(f'第{idx}个样本-重建信号')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def calculate_scalar_performance_metrics(true_signal, pred_signal):
    metrics = {'MAE': np.nan, 'RMSE': np.nan, 'Correlation': np.nan, 'SMAPE': np.nan}
    true, pred = np.array(true_signal).flatten(), np.array(pred_signal).flatten()
    min_len = min(len(true), len(pred))
    if min_len < 2: return metrics
    true, pred = true[:min_len], pred[:min_len]
    metrics['MAE'] = np.mean(np.abs(true - pred))
    metrics['RMSE'] = np.sqrt(np.mean((true - pred) ** 2))
    epsilon = 1e-9
    metrics['SMAPE'] = np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2 + epsilon)) * 100
    if min_len > 1 and np.var(true) > epsilon and np.var(pred) > epsilon:
        metrics['Correlation'] = np.corrcoef(true, pred)[0, 1]
    return metrics


def distance_similarity(orig_seg, rec_seg):
    min_len = min(len(orig_seg), len(rec_seg))
    if min_len < 2: return 0.0
    mad = np.mean(np.abs(orig_seg[:min_len] - rec_seg[:min_len]))
    ds = -2 / (1 + np.exp(-2.2 * (mad - 5.5))) + 1
    return ds


def trend_similarity(orig_seg, rec_seg):
    min_len = min(len(orig_seg), len(rec_seg))
    if min_len < 2: return 0.0
    t = np.arange(min_len)
    epsilon_std = 1e-9
    if np.std(orig_seg[:min_len]) < epsilon_std:
        slope_orig = 0.0
    else:
        slope_orig, _ = np.polyfit(t, orig_seg[:min_len], 1)
    mean_orig = np.mean(orig_seg[:min_len])
    mean_rec = np.mean(rec_seg[:min_len])
    rec_aligned = rec_seg[:min_len] - (mean_rec - mean_orig)
    if np.std(rec_aligned) < epsilon_std:
        slope_rec_aligned = 0.0
    else:
        slope_rec_aligned, _ = np.polyfit(t, rec_aligned, 1)
    angle_orig = np.arctan(slope_orig)
    angle_rec = np.arctan(slope_rec_aligned)
    angle_diff = np.abs(angle_orig - angle_rec)
    range_orig = np.max(orig_seg[:min_len]) - np.min(orig_seg[:min_len])
    range_rec_aligned = np.max(rec_aligned) - np.min(rec_aligned)
    max_signal_range = max(range_orig, range_rec_aligned)
    max_angle_epsilon = 1e-9
    if min_len > 1 and max_signal_range > max_angle_epsilon:
        max_slope_val = max_signal_range / (min_len - 1)
        max_angle = np.arctan(max_slope_val)
    else:
        max_angle = np.pi / 2
    if max_angle < max_angle_epsilon:
        ts = 1.0
    elif slope_orig * slope_rec_aligned >= 0:
        ts = 1 - (angle_diff / max_angle)
    else:
        ts = - (angle_diff / max_angle)
    return ts


def composite_similarity(ds, ts, w_dist=0.6):
    return w_dist * ds + (1 - w_dist) * ts


def calculate_pdf_metrics(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    if min_len < 2:
        return {'DS': np.nan, 'TS': np.nan, 'CS': np.nan}
    orig_segments, orig_change_points = [original[:min_len]], [0, min_len]
    segment_ds, segment_ts, segment_cs = [], [], []
    segment_lengths = []
    for i in range(len(orig_change_points) - 1):
        start_idx = orig_change_points[i]
        end_idx = orig_change_points[i + 1]
        seg_orig = original[start_idx:end_idx]
        seg_rec = reconstructed[start_idx:end_idx]
        len_seg = len(seg_orig)
        if len_seg < 2:
            continue
        ds = distance_similarity(seg_orig, seg_rec)
        ts = trend_similarity(seg_orig, seg_rec)
        cs = composite_similarity(ds, ts)
        segment_ds.append(ds)
        segment_ts.append(ts)
        segment_cs.append(cs)
        segment_lengths.append(len_seg)
    if not segment_ds:
        return {'DS': np.nan, 'TS': np.nan, 'CS': np.nan}
    total_length = sum(segment_lengths)
    if total_length == 0: return {'DS': np.nan, 'TS': np.nan, 'CS': np.nan}
    ds_avg = sum(d * l for d, l in zip(segment_ds, segment_lengths)) / total_length
    ts_avg = sum(t * l for t, l in zip(segment_ts, segment_lengths)) / total_length
    cs_avg = sum(c * l for c, l in zip(segment_cs, segment_lengths)) / total_length
    return {'DS': ds_avg, 'TS': ts_avg, 'CS': cs_avg}


def calculate_all_similarity_metrics(original, reconstructed):
    metrics = {}
    traditional_metrics = calculate_scalar_performance_metrics(original, reconstructed)
    metrics.update(traditional_metrics)
    pdf_metrics = calculate_pdf_metrics(original, reconstructed)
    metrics['DistanceSimilarity'] = pdf_metrics['DS']
    metrics['TrendSimilarity'] = pdf_metrics['TS']
    metrics['CompositeSimilarity'] = pdf_metrics['CS']
    return metrics


def calculate_overall_metrics(original_signals, reconstructed_signals):
    all_metrics = []
    for true_s, pred_s in zip(original_signals, reconstructed_signals):
        min_len = min(len(true_s), len(pred_s))
        if min_len < 2:
            continue
        signal_metrics = calculate_all_similarity_metrics(true_s, pred_s)
        all_metrics.append(signal_metrics)
    if not all_metrics:
        return {
            'MAE': np.nan, 'RMSE': np.nan, 'Correlation': np.nan, 'SMAPE': np.nan
        }
    overall_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        overall_metrics[key] = np.mean(values) if values else np.nan
    return overall_metrics


def print_evaluation_metrics(title, metrics):
    print(f"\n--- {title} ---")
    print("  传统指标:")
    print(f"    MAE: {metrics['MAE']:.4f}")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    相关系数: {metrics['Correlation']:.4f}")
    print(f"    SMAPE: {metrics['SMAPE']:.4f}%")
    print("\n  PDF相似性指标（范围[-1,1]）:")
    print(f"    距离相似性(DS): {metrics['DistanceSimilarity']:.4f}")
    print(f"    趋势相似性(TS): {metrics['TrendSimilarity']:.4f}")
    print(f"    复合相似性(CS): {metrics['CompositeSimilarity']:.4f}")


# ===================== 5. 主流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    train_x, train_y = load_training_data()
    test_y, test_x = load_testing_data()  # 注意：推理时只用test_x，test_y仅用于评估

    if train_x is None or test_x is None:
        print("数据加载失败，程序终止。")
        exit(1)

    # 2. 构建数据集
    train_dataset = SignalDataset(train_x, train_y, segment_length=1024)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet1D(n_channels=1, n_classes=1, bilinear=True)

    # 4. 训练
    model = train_model(model, train_loader, device, epochs=50, lr=1e-4)

    # 5. 推理（只用衰减信号，反归一化用目标信号均值方差）
    pred_test = predict(model, test_x, device, segment_length=1024)

    # 6. 保存结果
    np.save("reconstructed_test.npy", np.array(pred_test))
    print("测试集重建完成，结果已保存。")

    # 7. 自动评估与可视化
    print("\n===== 重建效果评估 =====")
    evaluate_metrics(test_y, pred_test)
    print("\n===== 自动绘制前3个样本对比图 =====")
    for idx in range(min(3, len(test_y))):
        plot_signals(test_y, test_x, pred_test, idx=idx, fs=100)

    # 在主流程最后加入评估指标打印
    overall_metrics = calculate_overall_metrics(test_y, pred_test)
    print_evaluation_metrics("测试集重建信号评估指标", overall_metrics)