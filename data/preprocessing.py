"""
波形预处理函数
==============

提供雷电波形的常用预处理操作：
    - Butterworth 低通滤波
    - 峰值裁剪
    - Z-score 归一化
    - MinMax 归一化
    - 完整预处理流水线
"""

import numpy as np


def butterworth_filter(piece, fc=120000, fs=5000000, order=2):
    """
    Butterworth 低通滤波 (sos 形式，与 3.py 一致)。

    使用 second-order sections (sos) 形式，比 ba 形式更数值稳定。

    Args:
        piece: (T,) float32 波形数据
        fc:    截止频率 (Hz)，默认 120 kHz
        fs:    采样率 (Hz)，默认 5 MHz
        order: 滤波器阶数，默认 2

    Returns:
        (T,) float32 滤波后的波形
    """
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, fc / (fs / 2), btype='low', output='sos')
        filtered = sosfiltfilt(sos, piece)
        return filtered.astype(np.float32)
    except ImportError:
        print("[Lig2Vec] scipy 未安装，跳过 Butterworth 滤波")
        return piece


def cut_around_peak(piece, before=2000, after=6000, target_length=8000):
    """
    以最大值为中心裁剪波形（与 3.py 推理管线一致）。

    使用 np.argmax 定位峰值，与推理时打过标签的 .lig 数据的处理方式一致。

    默认: 峰值前 2000 点 + 峰值后 6000 点 = 8000 点 (1.6 ms)

    Args:
        piece:         (T,) float32 波形数据
        before:        峰值前保留的采样点数
        after:         峰值后保留的采样点数
        target_length: 目标长度（不足则零填充，超出则截断）

    Returns:
        (target_length,) float32 裁剪后的波形
    """
    index_max = int(np.argmax(piece))

    # 与 3.py 的 cut_piece_around_max 逻辑完全一致:
    # 边界处理: begin<0 → begin=0, end=target_length
    #         end>len → end=len(piece), begin=end-target_length
    begin = index_max - before
    end = index_max + after

    if begin < 0:
        begin = 0
        end = target_length
    elif end > len(piece):
        end = len(piece)
        begin = end - target_length

    cut_piece = piece[begin:end]

    if len(cut_piece) < target_length:
        cut_piece = np.pad(cut_piece, (0, target_length - len(cut_piece)), 'constant')
    elif len(cut_piece) > target_length:
        cut_piece = cut_piece[:target_length]

    return cut_piece


def normalize_zscore(piece):
    """
    Z-score 归一化: (data - mean) / std

    Args:
        piece: (T,) float32 波形数据

    Returns:
        (T,) float32 归一化后的波形
    """
    std = piece.std()
    if std > 1e-8:
        return ((piece - piece.mean()) / std).astype(np.float32)
    return (piece - piece.mean()).astype(np.float32)


def normalize_minmax(piece):
    """
    MinMax 归一化: (data - mean) / (max - min)

    Args:
        piece: (T,) float32 波形数据

    Returns:
        (T,) float32 归一化后的波形
    """
    pmin, pmax = piece.min(), piece.max()
    if pmax - pmin < 1e-8:
        return (piece - piece.mean()).astype(np.float32)
    return ((piece - piece.mean()) / (pmax - pmin)).astype(np.float32)


def preprocess_waveform(piece, use_filter=True, cut_peak=True,
                        target_length=8000, normalize_mode='minmax'):
    """
    完整预处理流水线（与 3.py 推理管线一致）。

    依次执行: 滤波 → 峰值裁剪 → 归一化

    归一化方式与 3.py 一致:
        (x - mean) / (max - min)  即 normalize_mode='minmax'

    Args:
        piece:          (T,) float32 原始波形数据
        use_filter:     是否使用 Butterworth 滤波
        cut_peak:       是否以峰值为中心裁剪
        target_length:  裁剪后的目标长度
        normalize_mode: 归一化模式: 'minmax'(默认, 与3.py一致), 'zscore', 'none'

    Returns:
        (target_length,) float32 预处理后的波形
    """
    if use_filter:
        piece = butterworth_filter(piece)

    if cut_peak:
        piece = cut_around_peak(piece, target_length=target_length)

    if normalize_mode == 'zscore':
        piece = normalize_zscore(piece)
    elif normalize_mode == 'minmax':
        piece = normalize_minmax(piece)
    elif normalize_mode == 'none':
        pass
    else:
        piece = normalize_minmax(piece)

    return piece


# ============================================================
# Batch preprocessing — orders of magnitude faster
# ============================================================
def butterworth_filter_batch(pieces, fc=120000, fs=5000000, order=2):
    """
    Batch Butterworth 低通滤波 (sos 形式)。

    对 (N, T) 数组一次性滤波，避免 per-sample 调用开销。
    sosfiltfilt 原生支持沿最后一维滤波。

    Args:
        pieces: (N, T) float32 波形数据
        fc:     截止频率 (Hz)，默认 120 kHz
        fs:     采样率 (Hz)，默认 5 MHz
        order:  滤波器阶数，默认 2

    Returns:
        (N, T) float32 滤波后的波形
    """
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, fc / (fs / 2), btype='low', output='sos')
        # sosfiltfilt processes along last axis by default
        filtered = sosfiltfilt(sos, pieces, axis=-1)
        return filtered.astype(np.float32)
    except ImportError:
        return pieces


def cut_around_peak_batch(pieces, before=2000, after=6000, target_length=8000):
    """
    批量峰值裁剪（与 cut_around_peak 逻辑完全一致）。

    Args:
        pieces:        (N, T) float32 波形数据
        before:        峰值前保留的采样点数
        after:         峰值后保留的采样点数
        target_length: 目标长度

    Returns:
        (N, target_length) float32
    """
    N, T = pieces.shape
    peak_indices = np.argmax(pieces, axis=1).astype(np.int64)  # (N,)

    begins = peak_indices - before
    ends = peak_indices + after

    # 边界处理：begin < 0
    clamp_begin = (begins < 0)
    begins[clamp_begin] = 0
    ends[clamp_begin] = target_length

    # 边界处理：end > T
    clamp_end = (ends > T)
    ends[clamp_end] = T
    begins[clamp_end] = ends[clamp_end] - target_length

    # 逐条裁剪（用 numpy slice，很快）
    result = np.empty((N, target_length), dtype=np.float32)
    for i in range(N):
        seg = pieces[i, begins[i]:ends[i]]
        L = len(seg)
        if L < target_length:
            result[i, :L] = seg
            # rest already zero-ish from np.empty, but pad explicitly for safety
            result[i, L:] = 0.0
        else:
            result[i] = seg[:target_length]

    return result


def normalize_minmax_batch(pieces):
    """
    批量 MinMax 归一化: (x - mean) / (max - min)，逐条独立。

    Args:
        pieces: (N, T) float32

    Returns:
        (N, T) float32
    """
    pmin = pieces.min(axis=1, keepdims=True)
    pmax = pieces.max(axis=1, keepdims=True)
    pmean = pieces.mean(axis=1, keepdims=True)
    denom = pmax - pmin
    denom[denom < 1e-8] = 1.0
    return ((pieces - pmean) / denom).astype(np.float32)


def preprocess_batch(pieces, use_filter=True, cut_peak=True,
                     target_length=8000, normalize_mode='minmax'):
    """
    批量预处理流水线 — 用于训练/推理的快速路径。

    与 preprocess_waveform 逻辑完全一致，但对 (N, T) 数组批量操作，
    Butterworth 滤波速度提升 100-1000x。

    Args:
        pieces:         (N, T) float32 原始波形数据
        use_filter:     是否使用 Butterworth 滤波
        cut_peak:       是否以峰值为中心裁剪
        target_length:  裁剪后的目标长度
        normalize_mode: 归一化模式: 'minmax'(默认), 'zscore', 'none'

    Returns:
        (N, target_length) float32 预处理后的波形
    """
    if pieces.ndim == 1:
        pieces = pieces.reshape(1, -1)

    if use_filter:
        pieces = butterworth_filter_batch(pieces)

    if cut_peak:
        pieces = cut_around_peak_batch(pieces, target_length=target_length)

    if normalize_mode == 'minmax':
        pieces = normalize_minmax_batch(pieces)
    elif normalize_mode == 'zscore':
        # zscore per sample
        std = pieces.std(axis=1, keepdims=True)
        mean = pieces.mean(axis=1, keepdims=True)
        std[std < 1e-8] = 1.0
        pieces = ((pieces - mean) / std).astype(np.float32)
    elif normalize_mode == 'none':
        pass
    else:
        pieces = normalize_minmax_batch(pieces)

    return pieces
