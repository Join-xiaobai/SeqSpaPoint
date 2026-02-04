import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# 假设这两个指标计算文件已存在且格式正确
from .regression_metrics import calculate_regression_metrics
from .classification_metrics import calculate_binary_metrics, calculate_multiclass_metrics

def evaluate_on_test(model, test_loader, device, task_type="dta", num_classes=None, scaler=None, output_index=0):
    """
    通用测试评估函数，支持 DTA / DTI / MOA。
    【核心修改】
    - 统一所有任务的输入格式，对齐 Dataset 输出（point_cloud, drug_emb, target_emb, label）
    - 修复 DTI/MOA 模型输入不匹配问题，支持分离的 drug_emb/target_emb
    - 优化 test_loss 赋值逻辑，保证统计口径一致
    - 增加 num_classes 校验，提升健壮性
    - 完全移除标准化（scaler）相关逻辑，仅保留参数兼容性
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # 初始化损失函数
    if task_type == "dta":
        criterion = nn.MSELoss()
    elif task_type == "dti":
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "moa":
        criterion = nn.CrossEntropyLoss()
        # 校验 MOA 任务的 num_classes 参数
        if num_classes is None or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("❌ MOA 任务必须传入合法的正整数 num_classes！")
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {task_type.upper()} Test Set"):
            # === 统一解包输入，对齐 Dataset 输出（所有任务一致，避免解包报错）===
            point_cloud, drug_emb, target_emb, labels = batch
            
            # 设备迁移（开启 non_blocking 提升效率，仅在 CUDA 可用时生效）
            point_cloud = point_cloud.to(device, non_blocking=True if torch.cuda.is_available() else False)
            drug_emb = drug_emb.to(device, non_blocking=True if torch.cuda.is_available() else False)
            target_emb = target_emb.to(device, non_blocking=True if torch.cuda.is_available() else False)
            labels = labels.to(device, non_blocking=True if torch.cuda.is_available() else False)

            # === 模型前向传播（所有任务统一输入，适配分离的药物/靶点嵌入）===
            outputs = model(point_cloud, drug_emb, target_emb)
            
            # 处理模型输出为元组的情况（提取指定索引的输出）
            if isinstance(outputs, tuple):
                outputs = outputs[output_index]

            # === 按任务类型计算损失、收集预测结果 ===
            if task_type == "dta":
                # DTA：回归任务，调整输出形状并计算 MSE 损失
                outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
                loss = criterion(outputs, labels.float())
                
                # 收集原始预测值
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            elif task_type == "dti":
                # DTI：二分类任务，计算 BCE 损失并转换为概率
                outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
                loss = criterion(outputs, labels.float())
                
                # 用 sigmoid 转换为 0-1 概率
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

            elif task_type == "moa":
                # MOA：多分类任务，计算 CrossEntropy 损失（直接使用 logits）
                loss = criterion(outputs, labels.long())
                
                # 收集 logits 用于后续计算多分类指标（指标函数内部处理 softmax）
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 累计批次损失
            total_loss += loss.item()

    # === 计算平均损失（批次平均，统计口径与训练过程一致）===
    avg_loss = round(total_loss / len(test_loader), 6)

    # ========================
    # DTA：仅计算原始尺度指标（无 scaler）
    # ========================
    if task_type == "dta":
        preds = np.array(all_preds)
        labels = np.array(all_labels)

        # 直接计算所有回归 + 排序指标（原始尺度）
        metrics = calculate_regression_metrics(labels, preds)

        # 统一命名：不加 _raw 后缀，格式对齐分类任务
        metrics = {
            'MSE': metrics['mse'],
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R2': metrics['r2'],
            'Pearson': metrics['pearson'],
            'Spearman': metrics['spearman'],
            'CI': metrics['ci'],
        }

        # 优化 test_loss 赋值：使用批次平均损失（与训练过程一致），而非全局 MSE
        metrics["test_loss"] = avg_loss

        # ===== 调试打印 =====
        print(f"[DEBUG] 标签范围: {labels.min():.2f} ～ {labels.max():.2f}")
        print(f"[DEBUG] 预测范围: {preds.min():.2f} ～ {preds.max():.2f}")
        print(f"[DEBUG] MSE: {metrics['MSE']:.6f} | 批次平均损失: {metrics['test_loss']:.6f}")

    # ========================
    # DTI / MOA：分类任务指标计算
    # ========================
    else:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        if task_type == "dti":
            raw_metrics = calculate_binary_metrics(all_labels, all_preds)
        elif task_type == "moa":
            raw_metrics = calculate_multiclass_metrics(all_labels, all_preds, num_classes)

        # 转换 key 为大写（与 DTA 一致，避免后续 KeyError）
        metrics = {}
        for k, v in raw_metrics.items():
            metrics[k.capitalize()] = v
        metrics["test_loss"] = avg_loss

    return metrics