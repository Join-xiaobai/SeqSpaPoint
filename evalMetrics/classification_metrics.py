import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,  # 新增：精确率
    recall_score,     # 新增：召回率
    classification_report
)


def calculate_binary_metrics(true_labels, pred_probs, threshold=0.5):
    """
    计算 DTI 二分类任务指标。
    - pred_probs: 模型输出的正类概率（sigmoid 后），shape=[N]
    - threshold: 分类阈值（默认0.5）
    """
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs).flatten()
    pred_labels = (pred_probs >= threshold).astype(int)

    # 优化1：增加单类别校验，避免 AUC 计算报错
    try:
        auc = roc_auc_score(true_labels, pred_probs)
    except ValueError:
        auc = np.nan
        print("⚠️  真实标签仅包含单一类别，无法计算 AUC")

    try:
        auprc = average_precision_score(true_labels, pred_probs)
    except ValueError:
        auprc = np.nan
        print("⚠️  真实标签仅包含单一类别，无法计算 AUPRC")

    acc = accuracy_score(true_labels, pred_labels)
    
    # 优化：增加 F1 异常捕获
    try:
        f1 = f1_score(true_labels, pred_labels)
    except ValueError:
        f1 = np.nan
        print("⚠️  真实标签仅包含单一类别，无法计算 F1 分数")

    # ========== 新增：精确率 + 召回率（和你的逻辑对齐）==========
    try:
        precision = precision_score(true_labels, pred_labels, zero_division=0)  # zero_division避免除0报错
    except ValueError:
        precision = np.nan
        print("⚠️  真实标签仅包含单一类别，无法计算 精确率")

    try:
        recall = recall_score(true_labels, pred_labels, zero_division=0)
    except ValueError:
        recall = np.nan
        print("⚠️  真实标签仅包含单一类别，无法计算 召回率")
    # ===========================================================

    return {
        "auc": round(auc, 6) if not np.isnan(auc) else np.nan,
        "auprc": round(auprc, 6) if not np.isnan(auprc) else np.nan,
        "accuracy": round(acc, 6),
        "f1": round(f1, 6) if not np.isnan(f1) else np.nan,
        # ========== 新增返回：精确率、召回率（保留round和nan处理）==========
        "precision": round(precision, 6) if not np.isnan(precision) else np.nan,
        "recall": round(recall, 6) if not np.isnan(recall) else np.nan
    }


def calculate_multiclass_metrics(true_labels, pred_logits_or_probs, num_classes, return_report=False):
    """
    计算 MOA 多分类任务指标。
    - pred_logits_or_probs: shape=[N, C]，可以是 logits 或 softmax 概率
    - return_report: 是否返回详细分类报告（默认False）
    """
    true_labels = np.array(true_labels)  # [N]
    pred_array = np.array(pred_logits_or_probs)  # [N, C]

    # 优化2：增加形状校验，避免维度不匹配报错
    if pred_array.shape[1] != num_classes:
        raise ValueError(f"❌ 预测结果维度 {pred_array.shape[1]} 与 num_classes {num_classes} 不匹配！")

    # 优化3：修正概率判断逻辑（核心修复）
    is_prob = (np.all(pred_array >= 0 - 1e-8) and np.all(pred_array <= 1 + 1e-8))  # 概率范围
    row_sums = pred_array.sum(axis=1)
    is_prob &= np.allclose(row_sums, 1, atol=1e-3)  # 行和接近1

    if not is_prob:
        # Apply softmax（增加数值稳定性，避免 exp 爆炸）
        exps = np.exp(pred_array - np.max(pred_array, axis=1, keepdims=True))
        pred_probs = exps / exps.sum(axis=1, keepdims=True)
    else:
        pred_probs = pred_array
        # 修正极小负数为 0，保证概率分布合法
        pred_probs = np.clip(pred_probs, 0, 1)

    pred_labels = np.argmax(pred_probs, axis=1)

    # 优化4：增强标签校验与裁剪
    valid_labels = np.arange(num_classes)
    # 校验标签类型
    if not np.issubdtype(true_labels.dtype, np.integer):
        true_labels = true_labels.astype(int)
        print("⚠️  真实标签非整数类型，已自动转换为 int")
    # 裁剪非法标签并提示
    valid_label_mask = (true_labels >= 0) & (true_labels < num_classes)
    if not np.all(valid_label_mask):
        invalid_count = np.sum(~valid_label_mask)
        print(f"⚠️  发现 {invalid_count} 个非法标签（超出 [0, {num_classes-1}] 范围），已自动裁剪")
    true_labels_clipped = np.clip(true_labels, 0, num_classes - 1)  # 裁剪非法标签

    macro_f1 = f1_score(true_labels_clipped, pred_labels, average='macro', labels=valid_labels, zero_division=0)
    micro_f1 = f1_score(true_labels_clipped, pred_labels, average='micro', zero_division=0)
    acc = accuracy_score(true_labels_clipped, pred_labels)

    # 构建结果字典
    result = {
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "accuracy": round(acc, 6),
        "num_classes": num_classes
    }

    # 可选返回详细分类报告
    if return_report:
        report = classification_report(
            true_labels_clipped, 
            pred_labels, 
            labels=valid_labels, 
            zero_division=0, 
            output_dict=True
        )
        result["classification_report"] = report

    return result