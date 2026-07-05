"""
.lig 二进制文件解析器
=====================

本模块实现 .lig 格式雷电波形文件的底层解析。
所有 .lig 文件的读取操作都应通过本模块完成。

.lig 文件二进制格式:
    文件头: 112 字节
    每个 piece: 32208 字节
        - piece 头: 108 字节
        - 时间戳: 28 字节 (6 个 int32 + 4 字节对齐)
        - 秒小数: 8 字节 (1 个 float64)
        - 保留字段: 64 字节
        - 波形数据: 32000 字节 (16000 个 uint16 采样点)
    采样率: 5 MHz, 每段波形: 16000 个采样点 (3.2 ms)

格式验证:
    - validate_lig_file():  验证 .lig 文件格式完整性
    - LigFormatError:       自定义格式错误异常
"""

import os
import struct
import logging
import numpy as np

logger = logging.getLogger(__name__)

# .lig 文件格式常量
_LIG_FILE_HEADER_BYTES = 112
_LIG_PIECE_BYTES = 32208
_LIG_PIECE_HEADER_BYTES = 108
_LIG_TIMESTAMP_BYTES = 28
_LIG_SECFRAC_BYTES = 8
_LIG_RESERVED_BYTES = 64
_LIG_WAVEFORM_BYTES = 32000
_LIG_WAVEFORM_SAMPLES = 16000

# 波形数据在单个 piece 内的偏移量（pv==1001 的默认值）
_LIG_WAVEFORM_OFFSET = (
    _LIG_PIECE_HEADER_BYTES +
    _LIG_TIMESTAMP_BYTES +
    _LIG_SECFRAC_BYTES +
    _LIG_RESERVED_BYTES
)

# 不同 piece 版本的头部字节数（从 piece 起始到波形数据起始）
_LIG_PIECE_HDR_SIZE_PV1001 = 208   # pv==1001: 108 + 28 + 8 + 64
_LIG_PIECE_HDR_SIZE_PV_OTHER = 464  # pv!=1001: 包含额外字段

# piece 版本字段在 piece 内的偏移（类内常量，piece header 前 4 字节）
_LIG_PV_OFFSET_IN_PIECE = 0
_PV_READ_SIZE = 4

# 时间戳字段的合理范围
_TIMESTAMP_YEAR_RANGE = (2000, 2100)          # 四位年份范围
_TIMESTAMP_YEAR_RANGE_TWO_DIGIT = (0, 99)     # 两位年份范围 (00-99 → 2000-2099)
_TIMESTAMP_MONTH_RANGE = (1, 12)
_TIMESTAMP_DAY_RANGE = (1, 31)
_TIMESTAMP_HOUR_RANGE = (0, 23)
_TIMESTAMP_MINUTE_RANGE = (0, 59)
_TIMESTAMP_SECOND_RANGE = (0, 60)  # 60 允许闰秒


class LigFormatError(Exception):
    """ .lig 文件格式错误异常 """
    pass


def validate_lig_file(filepath):
    """
    验证 .lig 文件格式完整性

    检查项:
        1. 文件存在且可读
        2. 文件大小 >= 文件头大小
        3. 文件包含至少一个完整 piece
        4. 时间戳字段在合理范围内（采样验证第一个 piece）
        5. 波形数据不全为零（采样验证第一个 piece）

    注意: .lig 文件不要求固定的 piece 数量，文件尾部不完整字节会被自动忽略。

    Args:
        filepath: .lig 文件路径

    Returns:
        dict: 验证结果，包含 valid, n_pieces, warnings, errors

    Raises:
        LigFormatError: 文件无法读取或格式严重错误
    """
    result = {
        'valid': False,
        'n_pieces': 0,
        'file_size': 0,
        'warnings': [],
        'errors': [],
    }

    # 1. 文件存在性检查
    if not os.path.exists(filepath):
        raise LigFormatError(f"文件不存在: {filepath}")

    if not os.path.isfile(filepath):
        raise LigFormatError(f"路径不是文件: {filepath}")

    # 2. 文件大小检查
    try:
        file_size = os.path.getsize(filepath)
    except OSError as e:
        raise LigFormatError(f"无法获取文件大小: {filepath}, 错误: {e}")

    result['file_size'] = file_size

    if file_size < _LIG_FILE_HEADER_BYTES:
        result['errors'].append(
            f"文件过小 ({file_size} 字节)，小于文件头大小 ({_LIG_FILE_HEADER_BYTES} 字节)"
        )
        return result

    # 3. 计算完整 piece 数量（尾部不完整字节静默忽略）
    data_size = file_size - _LIG_FILE_HEADER_BYTES
    n_complete_pieces = data_size // _LIG_PIECE_BYTES

    if n_complete_pieces == 0:
        result['errors'].append("文件不包含任何完整的 piece")
        return result

    result['n_pieces'] = n_complete_pieces

    # 4. 采样验证第一个 piece 的时间戳和波形数据
    try:
        with open(filepath, 'rb') as f:
            # 读取第一个 piece 的时间戳区域
            ts_offset = _LIG_FILE_HEADER_BYTES + _LIG_PIECE_HEADER_BYTES
            f.seek(ts_offset)
            ts_bytes = f.read(_LIG_TIMESTAMP_BYTES + _LIG_SECFRAC_BYTES)

            if len(ts_bytes) < _LIG_TIMESTAMP_BYTES + _LIG_SECFRAC_BYTES:
                result['warnings'].append("第一个 piece 的时间戳数据不完整")
            else:
                # 解析时间戳
                ymdhms = struct.unpack_from('6i4x', ts_bytes, 0)
                sec_frac = struct.unpack_from('d', ts_bytes, _LIG_TIMESTAMP_BYTES)[0]

                # 验证时间戳范围
                year, month, day, hour, minute, second = ymdhms
                ts_warnings = []

                # 年份验证: 支持两位年份（如19代表2019）和四位年份
                if year < 100:
                    # 两位年份: 0-99 → 2000-2099
                    if not (_TIMESTAMP_YEAR_RANGE_TWO_DIGIT[0] <= year <= _TIMESTAMP_YEAR_RANGE_TWO_DIGIT[1]):
                        ts_warnings.append(f"两位年份 {year} 超出合理范围 {_TIMESTAMP_YEAR_RANGE_TWO_DIGIT}")
                    else:
                        # 转换为四位年份用于显示
                        year_display = 2000 + year
                else:
                    # 四位年份
                    if not (_TIMESTAMP_YEAR_RANGE[0] <= year <= _TIMESTAMP_YEAR_RANGE[1]):
                        ts_warnings.append(f"年份 {year} 超出合理范围 {_TIMESTAMP_YEAR_RANGE}")
                    year_display = year
                if not (_TIMESTAMP_MONTH_RANGE[0] <= month <= _TIMESTAMP_MONTH_RANGE[1]):
                    ts_warnings.append(f"月份 {month} 超出合理范围 {_TIMESTAMP_MONTH_RANGE}")
                if not (_TIMESTAMP_DAY_RANGE[0] <= day <= _TIMESTAMP_DAY_RANGE[1]):
                    ts_warnings.append(f"日期 {day} 超出合理范围 {_TIMESTAMP_DAY_RANGE}")
                if not (_TIMESTAMP_HOUR_RANGE[0] <= hour <= _TIMESTAMP_HOUR_RANGE[1]):
                    ts_warnings.append(f"小时 {hour} 超出合理范围 {_TIMESTAMP_HOUR_RANGE}")
                if not (_TIMESTAMP_MINUTE_RANGE[0] <= minute <= _TIMESTAMP_MINUTE_RANGE[1]):
                    ts_warnings.append(f"分钟 {minute} 超出合理范围 {_TIMESTAMP_MINUTE_RANGE}")
                if not (_TIMESTAMP_SECOND_RANGE[0] <= second <= _TIMESTAMP_SECOND_RANGE[1]):
                    ts_warnings.append(f"秒 {second} 超出合理范围 {_TIMESTAMP_SECOND_RANGE}")
                if sec_frac < 0 or sec_frac >= 1.0:
                    ts_warnings.append(f"秒小数 {sec_frac:.7f} 超出合理范围 [0, 1)")

                if ts_warnings:
                    result['warnings'].extend(
                        [f"时间戳异常: {w}" for w in ts_warnings]
                    )

            # 读取第一个 piece 的波形数据
            waveform_offset = (
                _LIG_FILE_HEADER_BYTES +
                _LIG_PIECE_HEADER_BYTES +
                _LIG_TIMESTAMP_BYTES +
                _LIG_SECFRAC_BYTES +
                _LIG_RESERVED_BYTES
            )
            f.seek(waveform_offset)
            waveform_bytes = f.read(_LIG_WAVEFORM_BYTES)

            if len(waveform_bytes) < _LIG_WAVEFORM_BYTES:
                result['warnings'].append("第一个 piece 的波形数据不完整")
            else:
                waveform = np.frombuffer(waveform_bytes, dtype=np.uint16).astype(np.float32)
                # 检查波形是否全为零
                if np.all(waveform == 0):
                    result['warnings'].append("第一个 piece 的波形数据全为零，可能为空数据")

    except (OSError, struct.error) as e:
        result['warnings'].append(f"采样验证时读取失败: {e}")

    # 5. 汇总结果
    result['valid'] = len(result['errors']) == 0

    return result


def count_lig_pieces(filepath):
    """
    统计 .lig 文件中的完整 piece 数量

    Args:
        filepath: .lig 文件路径

    Returns:
        int: 完整 piece 数量

    Raises:
        LigFormatError: 文件不存在或过小
    """
    if not os.path.exists(filepath):
        raise LigFormatError(f"文件不存在: {filepath}")

    try:
        file_size = os.path.getsize(filepath)
    except OSError as e:
        raise LigFormatError(f"无法获取文件大小: {filepath}, 错误: {e}")

    if file_size < _LIG_FILE_HEADER_BYTES:
        raise LigFormatError(
            f"文件过小 ({file_size} 字节)，小于 .lig 文件头大小 ({_LIG_FILE_HEADER_BYTES} 字节): {filepath}"
        )

    n_pieces = (file_size - _LIG_FILE_HEADER_BYTES) // _LIG_PIECE_BYTES
    return max(0, n_pieces)


def read_lig_binary(filepath, validate=True):
    """
    读取 .lig 二进制格式文件，返回所有 piece 和时间戳

    Args:
        filepath:  .lig 文件路径
        validate:  是否在读取前进行格式验证 (默认 True)

    Returns:
        pieces:     list of numpy arrays, 每个 (16000,) float32
        timestamps: list of str, 每段波形的时间戳字符串

    Raises:
        LigFormatError: 文件格式验证失败
    """
    # 格式验证
    if validate:
        result = validate_lig_file(filepath)
        if not result['valid']:
            raise LigFormatError(
                f".lig 文件格式验证失败: {filepath}, 错误: {result['errors']}"
            )
        if result['warnings']:
            for w in result['warnings']:
                logger.warning(f".lig 验证警告 ({filepath}): {w}")

    pieces = []
    timestamps = []

    with open(filepath, 'rb') as f:
        raw = f.read()

    num_pieces = (len(raw) - _LIG_FILE_HEADER_BYTES) // _LIG_PIECE_BYTES

    offset = _LIG_FILE_HEADER_BYTES
    for piece_idx in range(num_pieces):
        try:
            # 跳过 piece 头
            offset += _LIG_PIECE_HEADER_BYTES

            # 解析时间戳: 6 个 int32 (年月日时分秒) + 4 字节对齐
            ymdhms = struct.unpack_from('6i4x', raw, offset)
            offset += _LIG_TIMESTAMP_BYTES

            # 解析秒小数部分
            sec_frac = struct.unpack_from('d', raw, offset)[0]
            offset += _LIG_SECFRAC_BYTES

            # 构造时间戳字符串
            # 支持两位年份（如19代表2019）和四位年份
            year = ymdhms[0]
            if year < 100:
                year = 2000 + year
            ts_str = '%04d%02d%02d%02d%02d%010.7f' % (
                year, ymdhms[1], ymdhms[2],
                ymdhms[3], ymdhms[4], ymdhms[5] + sec_frac
            )

            # 跳过保留字段
            offset += _LIG_RESERVED_BYTES

            # 读取波形数据: 16000 个 uint16
            # 使用 np.frombuffer 替代 struct.unpack，速度提升 10-50x
            waveform = np.frombuffer(raw, dtype=np.uint16, count=_LIG_WAVEFORM_SAMPLES, offset=offset).astype(np.float32)
            offset += _LIG_WAVEFORM_BYTES

            pieces.append(waveform)
            timestamps.append(ts_str)

        except struct.error as e:
            logger.warning(
                f"解析 piece {piece_idx} 时出错: {e}，跳过剩余数据"
            )
            break

    return pieces, timestamps


def read_lig_piece(filepath, piece_index, validate=True):
    """
    读取 .lig 文件中单个 piece（seek 方式，高效）

    通过 seek 直接定位到目标 piece 的波形数据位置，
    无需读取整个文件，适合大规模数据的随机访问。
    自动检测 piece 版本并使用正确的波形数据偏移量。

    Args:
        filepath:    .lig 文件路径
        piece_index: piece 索引（从 0 开始）
        validate:    是否验证 piece_index 有效性 (默认 True)

    Returns:
        numpy array: (16000,) float32 波形数据

    Raises:
        LigFormatError: 文件格式错误
        IndexError: piece_index 超出范围
    """
    import struct as _struct

    n_pieces = count_lig_pieces(filepath)

    if piece_index < 0 or piece_index >= n_pieces:
        raise IndexError(
            f"piece_index={piece_index} 超出范围，文件共 {n_pieces} 个 piece"
        )

    piece_start = _LIG_FILE_HEADER_BYTES + piece_index * _LIG_PIECE_BYTES

    # 读取 piece 版本，确定正确的波形偏移量
    try:
        with open(filepath, 'rb') as f:
            f.seek(piece_start + _LIG_PV_OFFSET_IN_PIECE)
            pv_bytes = f.read(_PV_READ_SIZE)
            if len(pv_bytes) >= _PV_READ_SIZE:
                pv = _struct.unpack('i', pv_bytes)[0]
                hdr_size = _LIG_PIECE_HDR_SIZE_PV1001 if pv == 1001 else _LIG_PIECE_HDR_SIZE_PV_OTHER
            else:
                hdr_size = _LIG_WAVEFORM_OFFSET

            waveform_offset = piece_start + hdr_size
            f.seek(waveform_offset)
            raw_bytes = f.read(_LIG_WAVEFORM_BYTES)
    except OSError as e:
        raise LigFormatError(f"读取文件失败: {filepath}, 错误: {e}")

    if len(raw_bytes) < _LIG_WAVEFORM_BYTES:
        # 文件不完整，零填充
        waveform = np.zeros(_LIG_WAVEFORM_SAMPLES, dtype=np.float32)
        if len(raw_bytes) >= 2:
            n_valid = len(raw_bytes) // 2
            partial = np.frombuffer(raw_bytes[:n_valid * 2], dtype=np.uint16).astype(np.float32)
            waveform[:n_valid] = partial
    else:
        # 使用 np.frombuffer 替代 struct.unpack，速度提升 10-50x
        waveform = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32)

    return waveform


class LigFileIndex:
    """
    多 .lig 文件的 piece 索引

    构建全局索引，支持按全局索引定位到具体文件和文件内偏移。
    缓存文件句柄以加速频繁读取。

    使用方式:
        index = LigFileIndex(['file1.lig', 'file2.lig'])
        piece = index.read_piece(500)  # 读取全局第 500 个 piece
        print(index.total_pieces)      # 总 piece 数
    """

    def __init__(self, filepaths, validate=True):
        """
        Args:
            filepaths:  .lig 文件路径列表
            validate:   是否在索引构建时验证文件格式 (默认 True)

        Raises:
            LigFormatError: 所有文件格式验证失败
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        self.filepaths = []
        self.num_pieces_per_file = []
        self._validation_warnings = []

        for fp in filepaths:
            try:
                if validate:
                    result = validate_lig_file(fp)
                    if not result['valid']:
                        logger.warning(
                            f"跳过无效 .lig 文件: {fp}, 错误: {result['errors']}"
                        )
                        continue
                    if result['warnings']:
                        self._validation_warnings.extend(
                            [f"{fp}: {w}" for w in result['warnings']]
                        )
                        for w in result['warnings']:
                            logger.warning(f".lig 验证警告 ({fp}): {w}")

                n = count_lig_pieces(fp)
                if n > 0:
                    self.filepaths.append(fp)
                    self.num_pieces_per_file.append(n)
                else:
                    logger.warning(f"跳过空 .lig 文件: {fp}")

            except LigFormatError as e:
                logger.warning(f"跳过无效 .lig 文件: {fp}, 错误: {e}")
                continue

        if len(self.filepaths) == 0:
            raise LigFormatError(
                f"没有找到包含有效 piece 的 .lig 文件 (共检查 {len(filepaths)} 个文件)"
            )

        # 累积和，用于二分查找
        self._cumsum = np.cumsum([0] + self.num_pieces_per_file)
        self.total_pieces = int(self._cumsum[-1])

        # 缓存文件句柄，加速读取
        self._file_handles = {}

        # pv 缓存: (file_idx, piece_idx) → waveform_offset
        self._pv_cache = {}
        self._pv_cache_max = 10000

        logger.info(
            f"LigFileIndex: 索引了 {len(self.filepaths)} 个文件, "
            f"共 {self.total_pieces} 个 piece"
        )

    def _locate(self, global_index):
        """
        将全局索引定位到文件索引和文件内 piece 索引

        Args:
            global_index: 全局 piece 索引

        Returns:
            (file_idx, piece_idx_in_file): 文件索引和文件内 piece 索引
        """
        if global_index < 0 or global_index >= self.total_pieces:
            raise IndexError(
                f"全局索引 {global_index} 超出范围 [0, {self.total_pieces})"
            )

        file_idx = int(np.searchsorted(self._cumsum, global_index, side='right')) - 1
        if file_idx < 0:
            file_idx = 0
        piece_idx_in_file = global_index - self._cumsum[file_idx]
        return file_idx, int(piece_idx_in_file)

    def _get_file_handle(self, file_idx):
        """获取缓存的文件句柄（LRU 淘汰策略）"""
        max_handles = 64
        if file_idx not in self._file_handles:
            if len(self._file_handles) >= max_handles:
                oldest_idx = next(iter(self._file_handles.keys()))
                try:
                    self._file_handles[oldest_idx].close()
                except Exception:
                    pass
                del self._file_handles[oldest_idx]
            filepath = self.filepaths[file_idx]
            try:
                self._file_handles[file_idx] = open(filepath, 'rb')
            except OSError as e:
                raise LigFormatError(f"无法打开文件: {filepath}, 错误: {e}")
        # LRU: 将访问过的项移到末尾
        handle = self._file_handles.pop(file_idx)
        self._file_handles[file_idx] = handle
        return self._file_handles[file_idx]

    def _get_piece_waveform_offset(self, file_idx: int, piece_idx_in_file: int) -> int:
        """
        根据 piece 版本计算正确的波形数据偏移量。

        不同版本的 .lig piece 具有不同的头部字节数：
            - pv==1001: 头部 208 字节
            - pv!=1001: 头部 464 字节

        Args:
            file_idx:          文件在 self.filepaths 中的索引
            piece_idx_in_file: 文件内的 piece 索引

        Returns:
            int: 波形数据在文件中的字节偏移量
        """
        cache_key = (file_idx, piece_idx_in_file)
        if cache_key in self._pv_cache:
            return self._pv_cache[cache_key]

        # 读取 piece 版本字段
        piece_start = (
            _LIG_FILE_HEADER_BYTES +
            piece_idx_in_file * _LIG_PIECE_BYTES
        )
        pv_offset = piece_start + _LIG_PV_OFFSET_IN_PIECE

        try:
            fh = self._get_file_handle(file_idx)
            fh.seek(pv_offset)
            pv_bytes = fh.read(_PV_READ_SIZE)
            if len(pv_bytes) < _PV_READ_SIZE:
                hdr_size = _LIG_PIECE_HDR_SIZE_PV1001  # fallback
            else:
                pv = struct.unpack('i', pv_bytes)[0]
                if pv == 1001:
                    hdr_size = _LIG_PIECE_HDR_SIZE_PV1001
                elif pv in (2001, 3001):
                    hdr_size = _LIG_PIECE_HDR_SIZE_PV_OTHER
                else:
                    # Unrecognized / template placeholder (e.g. 0xCDCDCDCD = -842150451)
                    # → fallback to standard 208-byte header (most common)
                    hdr_size = _LIG_PIECE_HDR_SIZE_PV1001
        except (OSError, struct.error):
            hdr_size = _LIG_PIECE_HDR_SIZE_PV1001  # fallback

        waveform_offset = piece_start + hdr_size
        self._pv_cache[cache_key] = waveform_offset

        # LRU 淘汰
        if len(self._pv_cache) > self._pv_cache_max:
            oldest = next(iter(self._pv_cache))
            del self._pv_cache[oldest]

        return waveform_offset

    def read_piece(self, global_index):
        """
        按全局索引读取单个 piece（pv-aware）。

        自动检测 piece 版本并计算正确的波形数据偏移量，
        确保 pv==1001 和 pv!=1001 的 piece 都能被正确读取。

        Args:
            global_index: 全局 piece 索引

        Returns:
            numpy array: (16000,) float32 波形数据

        Raises:
            IndexError: 索引超出范围
            LigFormatError: 读取失败
        """
        file_idx, piece_idx_in_file = self._locate(global_index)

        # 计算波形数据在文件中的字节偏移（pv-aware）
        piece_offset_bytes = self._get_piece_waveform_offset(file_idx, piece_idx_in_file)

        try:
            fh = self._get_file_handle(file_idx)
            fh.seek(piece_offset_bytes)
            raw_bytes = fh.read(_LIG_WAVEFORM_BYTES)
        except (OSError, LigFormatError) as e:
            raise LigFormatError(
                f"读取 piece 失败 (global_index={global_index}): {e}"
            )

        if len(raw_bytes) < _LIG_WAVEFORM_BYTES:
            # 文件不完整，零填充（静默处理）
            waveform = np.zeros(_LIG_WAVEFORM_SAMPLES, dtype=np.float32)
            if len(raw_bytes) >= 2:
                n_valid = len(raw_bytes) // 2
                partial = np.frombuffer(raw_bytes[:n_valid * 2], dtype=np.uint16).astype(np.float32)
                waveform[:n_valid] = partial
        else:
            # 使用 np.frombuffer 替代 struct.unpack，速度提升 10-50x
            waveform = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32)

        return waveform

    def read_piece_raw(self, global_index):
        """
        按全局索引读取单个 piece 的完整原始字节（pv-aware）。

        不同版本的 .lig piece 具有不同的总大小：
            - pv==1001: 208 + 32000 = 32208 字节
            - pv!=1001: 464 + 32000 = 32464 字节

        Args:
            global_index: 全局 piece 索引

        Returns:
            bytes: 完整的 piece 原始字节，或空 bytes 如果读取失败
        """
        file_idx, piece_idx_in_file = self._locate(global_index)
        fh = self._get_file_handle(file_idx)

        # Determine actual piece size by reading pv
        # The pv is at a well-known offset within the piece structure
        waveform_offset = self._get_piece_waveform_offset(file_idx, piece_idx_in_file)
        # piece_start = waveform_offset - header_size
        # For pv==1001: header=208, for pv!=1001: header=464
        header_size = waveform_offset - (_LIG_FILE_HEADER_BYTES + piece_idx_in_file * _LIG_PIECE_BYTES)
        actual_piece_size = header_size + _LIG_WAVEFORM_BYTES
        piece_start = _LIG_FILE_HEADER_BYTES + piece_idx_in_file * actual_piece_size

        try:
            fh.seek(piece_start)
            return fh.read(actual_piece_size)
        except OSError:
            return b''

    def read_pieces_batch(self, global_indices):
        """
        批量按全局索引读取多个 piece（按文件分组、顺序读取，减少 seek 开销）。

        相比逐条 read_piece，在读取同一个文件的多个 piece 时，
        避免了反复 seek 和文件句柄切换。

        Args:
            global_indices: 全局 piece 索引列表

        Returns:
            list of numpy arrays: 各 (16000,) float32 波形，顺序与输入一致
        """
        from collections import defaultdict

        if not global_indices:
            return []

        # 1. 定位 + 按文件分组
        file_groups = defaultdict(list)  # file_idx → [(order, global_idx, piece_idx_in_file)]
        for order, gi in enumerate(global_indices):
            fi, pi = self._locate(gi)
            file_groups[fi].append((order, gi, pi))

        results = [None] * len(global_indices)

        # 2. 逐文件顺序读取
        for fi, items in file_groups.items():
            # 按 piece_idx_in_file 排序，实现顺序读取
            items_sorted = sorted(items, key=lambda x: x[2])  # sort by piece_idx

            try:
                fh = self._get_file_handle(fi)
                for order, gi, pi in items_sorted:
                    # 用 pv-aware 偏移
                    waveform_offset = self._get_piece_waveform_offset(fi, pi)
                    fh.seek(waveform_offset)
                    raw_bytes = fh.read(_LIG_WAVEFORM_BYTES)

                    if len(raw_bytes) < _LIG_WAVEFORM_BYTES:
                        waveform = np.zeros(_LIG_WAVEFORM_SAMPLES, dtype=np.float32)
                        if len(raw_bytes) >= 2:
                            n_valid = len(raw_bytes) // 2
                            partial = np.frombuffer(
                                raw_bytes[:n_valid * 2], dtype=np.uint16
                            ).astype(np.float32)
                            waveform[:n_valid] = partial
                    else:
                        waveform = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32)

                    results[order] = waveform

            except (OSError, LigFormatError) as e:
                raise LigFormatError(
                    f"批量读取 piece 失败 (file_idx={fi}): {e}"
                )

        return results

    def close(self):
        """关闭所有缓存的文件句柄"""
        for fh in self._file_handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._file_handles.clear()

    def __del__(self):
        self.close()

    def __len__(self):
        return self.total_pieces
