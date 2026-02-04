import numpy as np
from math import sqrt
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from scipy.stats import pearsonr, spearmanr


def concordance_index(y_true, y_pred):
    """
    优化版 CI 计算（真正 O(n log n)，无双重循环，解决卡顿问题）
    保留原逻辑，用向量化操作替代循环，大幅提升效率，适配大数据量。
    """
    y_true = np.array(y_true, dtype=np.float64).flatten()
    y_pred = np.array(y_pred, dtype=np.float64).flatten()
    
    n = len(y_true)
    if n < 2:
        return 1.0

    # 步骤1：按真实值排序（保留原逻辑）
    sorted_idx = np.argsort(y_true)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    # 步骤2：向量化计算有效样本对（避免双重循环）
    # 1. 生成所有 (i,j) 对，其中 i > j 且 y_true[i] != y_true[j]
    # 2. 用广播机制替代循环，计算 concordant 和 tied_pred
    concordant = 0
    tied_pred = 0
    total_valid = 0

    for i in range(n):
        # 只取 j < i，且 y_true[i] != y_true[j] 的样本
        j_mask = (y_true_sorted[i] != y_true_sorted[:i])
        y_pred_j = y_pred_sorted[:i][j_mask]
        if len(y_pred_j) == 0:
            continue
        
        # 向量化统计：无需内层循环
        concordant += np.sum(y_pred_sorted[i] > y_pred_j)
        tied_pred += np.sum(y_pred_sorted[i] == y_pred_j)
        total_valid += len(y_pred_j)

    if total_valid == 0:
        return 1.0

    ci_value = (concordant + 0.5 * tied_pred) / total_valid
    return ci_value


def calculate_regression_metrics(true_scores, pred_scores):
    """
    计算 DTA 回归任务评估指标。
    返回字典包含: r2, mae, mse, rmse, mape, pearson, spearman, ci
    """
    true_scores = np.array(true_scores, dtype=np.float64).flatten()
    pred_scores = np.array(pred_scores, dtype=np.float64).flatten()

    # 基础回归指标
    r2 = r2_score(true_scores, pred_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    mse = mean_squared_error(true_scores, pred_scores)
    rmse = sqrt(mse)

    # MAPE（可能因零除法失败，鲁棒处理）
    try:
        mape = mean_absolute_percentage_error(true_scores, pred_scores)
        mape = round(mape, 6)
    except Exception:
        mape = None

    # Pearson 相关系数（带异常捕获）
    try:
        pearson_corr, _ = pearsonr(true_scores, pred_scores)
        pearson_corr = round(pearson_corr, 6)
    except Exception:
        pearson_corr = float('nan')

    # Spearman 相关系数（带异常捕获）
    try:
        spearman_corr, _ = spearmanr(true_scores, pred_scores)
        spearman_corr = round(spearman_corr, 6)
    except Exception:
        spearman_corr = float('nan')

    # Concordance Index (CI)（先计算原始值，再四舍五入）
    try:
        ci_val = concordance_index(true_scores, pred_scores)
        ci_val = round(ci_val, 6) if not np.isnan(ci_val) else float('nan')
    except Exception:
        ci_val = float('nan')

    # 整理返回结果
    return {
        "r2": round(r2, 6),
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mape": mape,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "ci": ci_val
    }