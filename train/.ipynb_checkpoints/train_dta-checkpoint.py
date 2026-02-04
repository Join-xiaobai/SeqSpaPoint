import sys
import os
import json
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# 从独立模块导入
from model.SeqSpaPoint import SeqSpaPoint
from dataset.interaction_dataset import PointCloudInteractionDataset
from evalMetrics.regression_metrics import calculate_regression_metrics
from utils.common_utils import Logger, set_seed, convert_numpy_types


def soft_high_affinity_focal_rank_loss(
    y_pred, 
    y_true, 
    gamma=1.5, 
    threshold=10.0, 
    scale=2.5,
    max_pairs=4096,
    top_k_ratio=0.25,
    # ========== 新增：可控优化参数（默认关闭，保持原有逻辑） ==========
    use_batch_high_aff=False,  # 是否开启批次内高亲和阈值筛选（补充TopK）
    use_numerical_stability=False,  # 是否开启数值稳定保护（防止NaN）0.5
    dataset_type="kiba"  # 数据集类型（用于差异化适配）
):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    N = y_pred.size(0)
    if N < 2:
        # 新增：数值稳定，添加requires_grad=True防止梯度断流
        if use_numerical_stability:
            return torch.tensor(1.0, device=y_pred.device, requires_grad=True)
        else:
            return torch.tensor(0.0, device=y_pred.device)

    # 改进：确定性采样（保留高亲和力样本 + 低亲和力样本随机采样）
    if N * (N - 1) > max_pairs:
        # 步骤1：筛选高亲和力样本（两种方式可选，兼顾top-K和阈值，确保至少保留2个样本）
        top_k = max(2, int(N * top_k_ratio))  # 至少保留2个，避免无法构建样本对
        _, top_k_indices = torch.topk(y_true, k=top_k, dim=0)  # 按真实亲和力取top-K
        
        # ========== 可选控制：开启批次内高亲和阈值筛选（补充TopK，参数可控） ==========
        if use_batch_high_aff:
            # 基于当前批次的真实值筛选（而非全局固定阈值）
            batch_threshold = np.percentile(y_true.detach().cpu().numpy(), 70)
            batch_threshold = torch.tensor(batch_threshold, device=y_true.device, dtype=torch.float32)
            
            # 筛选批次内超过阈值的高亲和样本
            high_aff_threshold_indices = torch.where(y_true >= batch_threshold)[0]
            # 合并TopK和阈值筛选结果（去重），强化高亲和样本覆盖
            top_k_indices = torch.cat([top_k_indices, high_aff_threshold_indices], dim=0).unique()
            # 兜底：确保至少保留2个样本，防止报错
            top_k_indices = top_k_indices if len(top_k_indices) >= 2 else torch.topk(y_true, k=2, dim=0)[1]

        # 步骤2：筛选低亲和力样本（剩余样本中随机采样，控制总样本数平方≈max_pairs）
        # 构建剩余样本索引（排除高亲和力样本）
        all_indices = torch.arange(N, device=y_true.device)
        remaining_mask = ~torch.isin(all_indices, top_k_indices)
        remaining_indices = all_indices[remaining_mask]
        
        # 计算需要的低亲和力样本数
        total_sample_num = int(np.sqrt(max_pairs)) + 1
        low_aff_sample_num = max(0, total_sample_num - len(top_k_indices))
        
        # 对低亲和力样本随机采样
        if len(remaining_indices) > low_aff_sample_num:
            # 新增：可选数值稳定，固定随机种子保证可复现（参数可控）
            if use_numerical_stability:
                # 修复：生成器仅支持 CPU 设备，先在 CPU 上生成索引，再移回目标设备
                g = torch.Generator(device="cpu")  # 强制 CPU 设备，解决 RuntimeError
                g.manual_seed(42)  # 固定种子，保证采样可复现
                # 1. CPU 上生成随机排列索引 2. 截取需要的数量 3. 转换为与 remaining_indices 相同设备
                perm_indices = torch.randperm(len(remaining_indices), generator=g)[:low_aff_sample_num].to(remaining_indices.device)
                low_aff_indices = remaining_indices[perm_indices]
            else:
                low_aff_indices = remaining_indices[torch.randperm(len(remaining_indices))[:low_aff_sample_num]]
        else:
            low_aff_indices = remaining_indices  # 剩余样本不足时全取

        # 步骤3：合并索引（高亲和力在前，低亲和力在后，去重兜底）
        indices = torch.cat([top_k_indices, low_aff_indices], dim=0).unique()
        N = len(indices)
        
        # 最终兜底：确保至少有2个样本（避免排序损失无法计算）
        if N < 2:
            indices = torch.randperm(y_true.size(0))[:2]

        # 提取采样后的预测值和真实值
        y_pred = y_pred[indices]
        y_true = y_true[indices]

    # 以下逻辑：保留原有核心，新增数值稳定保护（参数可控）
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    
    labels = (true_diff > 0).float()
    valid_mask = (true_diff != 0)
    # 新增：数值稳定，将bool mask转为float，避免乘法类型错误（参数可控）
    if use_numerical_stability:
        valid_mask = valid_mask.float()

    avg_aff = (y_true.unsqueeze(1) + y_true.unsqueeze(0)) / 2.0
    # ========== 可选控制：数据集差异化缩放（参数可控，不影响原有逻辑） ==========
    if use_batch_high_aff:
        # 缩放，保持原有高亲和聚焦效果
        weight = torch.sigmoid(scale * 0.8 * (avg_aff - threshold))
    else:
        # 默认：沿用原有缩放逻辑，向下兼容
        weight = torch.sigmoid(scale * (avg_aff - threshold))

    p = torch.sigmoid(pred_diff)
    pt = labels * p + (1 - labels) * (1 - p)
    
    # 新增：数值稳定，裁剪pt避免极端值导致NaN（参数可控）
    if use_numerical_stability:
        pt = torch.clamp(pt, 1e-8, 1.0 - 1e-8)
        focal_modulation = (1 - pt) ** gamma
    else:
        # 沿用原有逻辑，向下兼容
        focal_modulation = (1 - pt + 1e-8) ** gamma

    bce_loss = F.binary_cross_entropy_with_logits(pred_diff, labels, reduction='none')
    
    loss = (focal_modulation * bce_loss * weight * valid_mask).sum()
    norm = (weight * valid_mask).sum() + 1e-8
    return loss / norm


def train_with_cv(
    FEATURE_CSV_PATH,
    POINT_CLOUD_PLY_PATH,
    BASE_SAVE_DIR,
    task_type="dta",
    label_col="affinity",
    id_cols=None,
    dataset_name="kiba",
    num_folds=5,
    num_epochs=300,
    batch_size=64,
    lr=1e-4,
    warmup_epochs=15,
    weight_decay=1e-5,
    patience=30,
    split_types=["warm", "drug_cold", "target_cold"],
    seed=42,
    dropout_rate = 0.2,
    # ========== 新增：训练阶段可控参数（默认关闭新优化，保持原有逻辑） ==========
    use_dynamic_lambda_warmup=False,  # 是否开启λ_rank warmup
    use_dataset_lambda_clip=False,  # 是否开启数据集差异化λ裁剪区间
    use_batch_threshold=False,  # 是否传递给损失函数，开启批次内高亲和筛选
    use_numerical_stability=False  # 是否传递给损失函数，开启数值稳定保护
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if task_type != "dta":
        raise ValueError(f"train_with_cv 仅支持 task_type='dta'，但收到: {task_type}")

    if id_cols is None:
        id_cols = ["drug_id", "protein_id"]

    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    global_log_path = os.path.join(BASE_SAVE_DIR, "dta_global_train_log.txt")
    sys.stdout = Logger(global_log_path)

    print(f"===== DTA实验开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} =====")
    print(f"  - 计算设备：{device}")
    print(f"  - 数据集名称：{dataset_name}")
    print(f"  - 标签列名：{label_col}")
    print(f"  - ID 列名：{id_cols}")
    print(f"  - 训练轮数：{num_epochs}")
    print(f"  - 批次大小：{batch_size}")
    print(f"  - 初始学习率：{lr}")
    print(f"  - 热身训练（warmup_epochs）轮数：{warmup_epochs}")
    print(f"  - 早停耐心值：{patience}")
    print(f"  - 🚀 联合损失：SmoothL1 + Soft High-Affinity Focal Ranking (γ=2.0)")
    print(f"  - 🔥 λ 调度：{'带Warmup的动态调度' if use_dynamic_lambda_warmup else '原有自适应调度'}")
    print(f"  - 🧠 模型：num_queries=6, dropout={dropout_rate}")
    # 新增：打印可控优化开关状态
    print(f"  - ⚙️  可选优化：批次高亲和筛选={use_batch_threshold} | 数值稳定={use_numerical_stability} | 数据集 λ 裁剪={use_dataset_lambda_clip}")
    print()

    for split_type in split_types:
        print("=" * 80)
        print(f"开始 {split_type.upper()} 场景DTA训练")
        print("=" * 80)

        scenario_dir = os.path.join(BASE_SAVE_DIR, split_type)
        log_dir = os.path.join(scenario_dir, "logs")
        model_dir = os.path.join(scenario_dir, "models")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        try:
            dataset = PointCloudInteractionDataset(
                feature_csv_path=FEATURE_CSV_PATH,
                point_cloud_ply_path=POINT_CLOUD_PLY_PATH,
                task_type="dta",
                split_type=split_type,
                test_size=0.1,
                label_col=label_col,
                id_cols=id_cols,
                standardize_embeddings=False
            )
            drug_dim = dataset.drug_embeddings.shape[1]
            target_dim = dataset.target_embeddings.shape[1]
            print(f"✅ DTA数据集加载成功 | Drug 嵌入维度: {drug_dim} | Target 嵌入维度: {target_dim}")
        except Exception as e:
            print(f"❌ DTA数据集加载失败：{e}")
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            return

        # === 动态计算高亲和力阈值（保留原有逻辑，新增数据集差异化） ===
        all_affinities = np.array(dataset.df[label_col])  # 假设 dataset.df 可访问
        if use_batch_threshold:
            # Davis：适配其分布，全局阈值提高，更聚焦高亲和
            threshold = np.percentile(all_affinities, 75)
        else:
            # 默认：沿用70%分位，保持KIBA原有效果
            threshold = np.percentile(all_affinities, 70)
        print(f"📊 自动设定高亲和力阈值 threshold = {threshold:.2f} ({70 if dataset_name.lower()=='kiba' else 75}% 分位数)")

        all_train_indices = dataset.train_indices
        fixed_test_indices = dataset.test_indices

        def collate_fn(batch):
            point_clouds, drug_embs, target_embs, affinities = zip(*batch)
            point_clouds = torch.stack(point_clouds)
            drug_embs = torch.stack(drug_embs)
            target_embs = torch.stack(target_embs)
            affinities = torch.tensor(affinities, dtype=torch.float32)
            return point_clouds, drug_embs, target_embs, affinities

        fixed_test_subset = Subset(dataset, fixed_test_indices)
        fixed_test_loader = DataLoader(
            fixed_test_subset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        fold_mse, fold_mae, fold_rmse = [], [], []
        fold_r2, fold_pearson, fold_spearman, fold_ci = [], [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_train_indices)):
            print("-" * 80)
            print(f"第 {fold_idx + 1}/{num_folds} 折训练（当前场景：{split_type}）")
            print("-" * 80)

            combined_kfold_indices = np.concatenate([train_idx, val_idx])
            train_orig_idx = [all_train_indices[i] for i in combined_kfold_indices]

            train_subset = Subset(dataset, train_orig_idx)
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn
            )

            model = SeqSpaPoint(
                drug_dim=drug_dim, 
                target_dim=target_dim, 
                k=10, 
                dropout_rate=dropout_rate, 
                num_queries=6
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                    min_lr_ratio = 5e-4  # 从1e-3上调至5e-4，后期保留更高学习率，精细拟合压低MSE
                    return cosine_decay * (1 - min_lr_ratio) + min_lr_ratio
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            fold_best_composite = -float('inf')
            fold_train_log = []
            patience_counter = 0
            lambda_rank = 0.0
            best_model_path = os.path.join(model_dir, f"fold_{fold_idx + 1}_best_composite.pth")
            best_test_ci = -float('inf')
            best_pearson = 0.0

            for epoch in range(num_epochs):
                current_epoch = epoch + 1

                # ========== 训练 ==========
                model.train()
                total_train_loss = total_smoothl1_loss = total_rank_loss = 0.0
                train_pred, train_true = [], []

                with tqdm(train_loader, desc=f"Epoch {current_epoch}/{num_epochs} [Train]") as pbar:
                    for point_cloud, drug_emb, target_emb, affinity in pbar:
                        point_cloud = point_cloud.to(device)
                        drug_emb = drug_emb.to(device)
                        target_emb = target_emb.to(device)
                        affinity = affinity.to(device)

                        optimizer.zero_grad()
                        pred = model(point_cloud, drug_emb, target_emb)

                        smoothl1_loss = F.smooth_l1_loss(pred, affinity, beta=1.0)
                        # ========== 传递可控参数给损失函数（保持原有参数，新增可选参数） ==========
                        ranking_loss = soft_high_affinity_focal_rank_loss(
                            pred, affinity, 
                            gamma=2.0, threshold=threshold, scale=2.0,
                            use_batch_high_aff=use_batch_threshold,  # 新增可控参数
                            use_numerical_stability=use_numerical_stability,  # 新增可控参数
                            dataset_type=dataset_name  # 新增数据集类型参数
                        )

                        loss = smoothl1_loss + lambda_rank * ranking_loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        total_train_loss += loss.item()
                        total_smoothl1_loss += smoothl1_loss.item()
                        total_rank_loss += ranking_loss.item()
                        train_pred.extend(pred.detach().cpu().numpy())
                        train_true.extend(affinity.cpu().numpy())

                        pbar.set_postfix({
                            "smoothl1": f"{smoothl1_loss.item():.6f}",
                            "focal_rank": f"{ranking_loss.item():.6f}",
                            "λ": f"{lambda_rank:.1f}"
                        })

                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                avg_total_loss = round(total_train_loss / len(train_loader), 6)
                avg_smoothl1_loss = round(total_smoothl1_loss / len(train_loader), 6)
                avg_rank_loss = round(total_rank_loss / len(train_loader), 6)
                train_metrics = calculate_regression_metrics(train_true, train_pred)

                # ========== 验证（在固定测试集） ==========
                model.eval()
                test_pred, test_true = [], []
                total_val_smoothl1 = 0.0
                total_val_rank = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    with tqdm(fixed_test_loader, desc=f"Epoch {current_epoch}/{num_epochs} [Test/Val]") as pbar:
                        for point_cloud, drug_emb, target_emb, affinity in pbar:
                            point_cloud = point_cloud.to(device)
                            drug_emb = drug_emb.to(device)
                            target_emb = target_emb.to(device)
                            affinity = affinity.to(device)
                
                            pred = model(point_cloud, drug_emb, target_emb)
                            smoothl1_loss = F.smooth_l1_loss(pred, affinity, beta=1.0)
                            # ========== 验证阶段同样传递可控参数（保持一致） ==========
                            ranking_loss = soft_high_affinity_focal_rank_loss(
                                pred, affinity, 
                                gamma=2.0, threshold=threshold, scale=2.0,
                                use_batch_high_aff=use_batch_threshold,
                                use_numerical_stability=use_numerical_stability,
                                dataset_type=dataset_name
                            )
                
                            total_val_smoothl1 += smoothl1_loss.item()
                            total_val_rank += ranking_loss.item()
                            num_val_batches += 1
                
                            test_pred.extend(pred.cpu().numpy())
                            test_true.extend(affinity.cpu().numpy())

                test_metrics = calculate_regression_metrics(test_true, test_pred)
                current_test_ci = test_metrics.get('ci', -float('inf'))
                if np.isnan(current_test_ci):
                    current_test_ci = -float('inf')
                current_pearson = test_metrics.get('pearson', 0.0)
                if np.isnan(current_pearson) or current_pearson is None:
                    current_pearson = 0.0
                composite_score = 0.6 * current_test_ci + 0.4 * current_pearson

                print(f"\n===== Epoch {current_epoch} 指标汇总 =====")
                print(f"【训练集】 Total Loss：{avg_total_loss} | SmoothL1：{avg_smoothl1_loss} | FocalRank：{avg_rank_loss}")
                print(f"           R2：{train_metrics['r2']} | Pearson：{train_metrics.get('pearson', 'N/A')} | MAE：{train_metrics['mae']} | RMSE：{train_metrics['rmse']}")
                print(f"【测试集】 CI：{current_test_ci:.6f} | Pearson：{test_metrics.get('pearson', 'N/A')} | MSE：{test_metrics['mse']:.6f} | MAE：{test_metrics['mae']:.6f}")
                print(f" 当前学习率：{current_lr:.6f} | λ_rank：{lambda_rank:.1f} | 本轮最优Composite：{fold_best_composite:.6f} (CI={best_test_ci:.4f}, Pearson={best_pearson:.4f})")
                print(f"=====================================\n")

                epoch_log = {
                    "epoch": current_epoch,
                    "train_total_loss": avg_total_loss,
                    "train_smoothl1_loss": avg_smoothl1_loss,
                    "train_rank_loss": avg_rank_loss,
                    "test_metrics": test_metrics,
                    "current_lr": current_lr,
                    "lambda_rank": lambda_rank,
                }
                fold_train_log.append(epoch_log)

                # ———————— 动态 λ_rank（新增参数控制，可选warmup和数据集差异化） ————————
                avg_val_smoothl1 = total_val_smoothl1 / num_val_batches
                avg_val_rank = total_val_rank / num_val_batches 
                lambda_rank = float((avg_val_rank + 1e-6) / (avg_val_smoothl1 + 1e-6))

                # 可选：开启λ_rank warmup，避免前期震荡（参数可控）
                if use_dynamic_lambda_warmup:
                    warmup_factor = min(1.0, current_epoch / warmup_epochs)
                    lambda_rank = lambda_rank * warmup_factor

                if dataset_name.lower() == "kiba":
                    # KIBA：沿用原有区间，保持效果
                    lambda_rank = np.clip(lambda_rank, 1.0, 6.0)
                else:  # davis
                    # Davis：调整区间，增强排序损失权重
                    lambda_rank = np.clip(lambda_rank, 1.2, 6.0)

                # ———————— 早停 & 保存 ————————
                if composite_score > fold_best_composite:
                    fold_best_composite = composite_score
                    # 同步更新最优CI和Pearson（关键修复）
                    best_test_ci = current_test_ci
                    best_pearson = current_pearson
                    best_model_path = os.path.join(model_dir, f"fold_{fold_idx + 1}_best_composite.pth")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"第 {fold_idx + 1} 折最优模型更新！Composite={composite_score:.6f} (CI={best_test_ci:.4f}, Pearson={best_pearson:.4f})，保存路径：{best_model_path}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停触发！连续{patience}轮 Composite 分数未提升，停止本轮训练")
                        break

            # ========== 最终测试 ==========
            log_file_path = os.path.join(log_dir, f"fold_{fold_idx + 1}_train_log.json")
            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(fold_train_log), f, indent=2, ensure_ascii=False)

            best_model = SeqSpaPoint(drug_dim=drug_dim, target_dim=target_dim, k=10, dropout_rate=dropout_rate, num_queries=6).to(device)
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            best_model.eval()

            final_pred, final_true = [], []
            with torch.no_grad():
                for point_cloud, drug_emb, target_emb, affinity in fixed_test_loader:
                    point_cloud = point_cloud.to(device)
                    drug_emb = drug_emb.to(device)
                    target_emb = target_emb.to(device)
                    affinity = affinity.to(device)
                    pred = best_model(point_cloud, drug_emb, target_emb)
                    final_pred.extend(pred.cpu().numpy())
                    final_true.extend(affinity.cpu().numpy())

            final_test_metrics = calculate_regression_metrics(final_true, final_pred)

            metric_keys = ['mse', 'mae', 'rmse', 'r2', 'pearson', 'spearman', 'ci']
            metric_lists = [fold_mse, fold_mae, fold_rmse, fold_r2, fold_pearson, fold_spearman, fold_ci]
            
            for key, lst in zip(metric_keys, metric_lists):
                value = final_test_metrics.get(key, float('nan'))
                lst.append(value)

            print(f"\n===== 第 {fold_idx + 1} 折最终测试结果 =====")
            for k, v in final_test_metrics.items():
                print(f"  {k.upper()}: {v:.6f}" if not isinstance(v, str) else f"  {k.upper()}: {v}")
            print(f"=====================================\n")

            test_log_path = os.path.join(log_dir, f"fold_{fold_idx + 1}_test_log.json")
            with open(test_log_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(final_test_metrics), f, indent=2, ensure_ascii=False)

        # ========== 5折汇总 ==========
        def safe_mean_std(arr):
            arr = np.array(arr, dtype=np.float64)
            return round(np.nanmean(arr), 6), round(np.nanstd(arr), 6)

        metrics_summary = {
            'MSE': safe_mean_std(fold_mse),
            'MAE': safe_mean_std(fold_mae),
            'RMSE': safe_mean_std(fold_rmse),
            'R2': safe_mean_std(fold_r2),
            'Pearson': safe_mean_std(fold_pearson),
            'Spearman': safe_mean_std(fold_spearman),
            'CI': safe_mean_std(fold_ci),
        }

        print("=" * 80)
        print(f"{split_type.upper()} 场景5折交叉验证结果汇总（基于固定10%测试集）")
        for name, (mean, std) in metrics_summary.items():
            print(f"  {name}: {mean} ± {std}")
        print("=" * 80 + "\n")

    print(f"===== DTA实验结束时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} =====")
    print(f"所有DTA实验完成！结果保存至：{BASE_SAVE_DIR}")
    sys.stdout.close()
    sys.stdout = sys.__stdout__