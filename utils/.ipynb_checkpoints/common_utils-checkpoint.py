# utils/dta_utils.py

import sys
import torch
import numpy as np

class Logger(object):
    """
    同时将标准输出重定向到控制台和日志文件，便于实验追踪。
    使用方式: sys.stdout = Logger("log.txt")
    """
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_seed(seed=42):
    """
    设置全局随机种子，确保实验可复现。
    包括 PyTorch CPU/GPU 随机数生成器，并关闭 cuDNN 非确定性优化。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_numpy_types(obj):
    """
    将包含 NumPy 类型的数据结构（字典、列表、元组等）转换为 Python 原生类型（int, float, list, dict）
    用于解决 JSON 序列化时遇到的 NumPy 类型不兼容问题（如 float32/float64 无法被 json.dump 序列化）

    参数:
        obj (dict/list/tuple/np.ndarray/np.number): 待转换的数据对象

    返回:
        转换后的 Python 原生类型数据结构

    使用场景:
        在保存模型训练日志（.json 文件）前调用此函数，确保所有数值类型（如 np.float32）转换为 Python float
        例如：fold_train_log = convert_numpy_types(fold_train_log)

    示例:
        >>> import numpy as np
        >>> data = {"loss": np.float32(0.85), "accuracy": np.array([0.92])}
        >>> convert_numpy_types(data)
        {"loss": 0.85, "accuracy": [0.92]}
    """
    # 如果是字典，递归处理所有键值对
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}

    # 如果是列表或元组，递归处理所有元素
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    # 处理 NumPy 整数类型（int32/int64 等）
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)

    # 处理 NumPy 浮点类型（float32/float64 等）
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)

    # 处理 NumPy 数组
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 转换为 Python 列表

    # 其他类型（如 str, int, float）直接返回
    else:
        return obj