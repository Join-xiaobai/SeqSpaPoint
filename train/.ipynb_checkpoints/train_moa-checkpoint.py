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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# 从独立模块导入（保持原路径风格）
from model.SeqSpaPoint import SeqSpaPoint
from dataset.interaction_dataset import PointCloudInteractionDataset
from evalMetrics.classification_metrics import calculate_binary_metrics
from utils.common_utils import Logger, set_seed, convert_numpy_types


def focal_binary_loss(
    y_pred, 
    y_true, 
    gamma=2.0, 
    alpha=0.25,
    use_numerical_stability=True
):
    """
    MOA专用Focal Loss（解决1:10类别不平衡问题）
    :param y_pred: 模型输出logits (B,)
    :param y_true: 真实标签 (B,)
    :param gamma: 聚焦参数（降低易分类样本权重）
    :param alpha: 类别权重参数（平衡正负样本）
    :param use_numerical_stability: 数值稳定保护（防止sigmoid溢出）
    :return: 标量损失值
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1).float()
    
    # 数值稳定保护：限制logits范围，防止sigmoid计算溢出
    if use_numerical_stability:
        y_pred = torch.clamp(y_pred, -10, 10)
    
    p = torch.sigmoid(y_pred)
    pt = y_true * p + (1 - y_true) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    
    # 类别加权：平衡正负样本比例
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    
    loss = (alpha_t * focal_weight * bce_loss).mean()
    return loss


def train_with_cv(
    FEATURE_CSV_PATH,
    POINT_CLOUD_PLY_PATH,
    BASE_SAVE_DIR,
    task_type="dti",
    label_col="label",
    id_cols=["DrugID", "TargetID"],
    dataset_name="moa_activation",
    num_folds=5,
    num_epochs=100,
    batch_size=64,
    lr=1e-4,
    warmup_epochs=10,
    weight_decay=1e-5,
    patience=15,
    split_types=["warm", "drug_cold", "target_cold"],
    seed=42,
    dropout_rate=0.15,
    # MOA专属优化参数
    use_focal_loss=True,
    use_class_weight=True,
    use_numerical_stability=True,
    focal_gamma=2.0,
    focal_alpha=0.25
):
    # 固定MOA任务配置，设置随机种子保证复现
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = "moa"

    # 创建MOA专属保存目录
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    global_log_path = os.path.join(BASE_SAVE_DIR, "moa_global_train_log.txt")
    sys.stdout = Logger(global_log_path)

    # 打印MOA实验信息（仅保留MOA相关）
    print(f"===== MOA实验开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} =====")
    print(f"  - 计算设备：{device}")
    print(f"  - 数据集名称：{dataset_name}")
    print(f"  - 标签列名：{label_col}")
    print(f"  - ID 列名：{id_cols}")
    print(f"  - K折数：{num_folds}")
    print(f"  - 训练轮数：{num_epochs}")
    print(f"  - 批次大小：{batch_size}")
    print(f"  - 初始学习率：{lr}")
    print(f"  - 热身训练轮数：{warmup_epochs}")
    print(f"  - 早停耐心值：{patience}")
    print(f"  - 损失函数：Focal BCEWithLogits (γ={focal_gamma}, α={focal_alpha})")
    print(f"  - 学习率调度：CosineAnnealing with Warmup")
    print(f"  - 模型：num_queries=6, dropout={dropout_rate}")
    print(f"  - 划分场景：{split_types}")
    print(f"  - 核心逻辑：K折 + 训练集 + 测试集做评估")
    print()

    # 遍历MOA的3种划分场景（warm/drug_cold/target_cold）
    for split_type in split_types:
        print("=" * 80)
        print(f"开始 {split_type.upper()} 场景MOA训练（K={num_folds}折）")
        print("=" * 80)

        # 场景专属目录（MOA命名）
        scenario_dir = os.path.join(BASE_SAVE_DIR, split_type)
        log_dir = os.path.join(scenario_dir, "logs")
        model_dir = os.path.join(scenario_dir, "models")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # 加载MOA专属数据集
        try:
            dataset = PointCloudInteractionDataset(
                feature_csv_path=FEATURE_CSV_PATH,
                point_cloud_ply_path=POINT_CLOUD_PLY_PATH,
                task_type=task_type,
                split_type=split_type,
                test_size=0.1,  # MOA固定10%测试集
                label_col=label_col,
                id_cols=id_cols,
                standardize_embeddings=False  # MOA建议关闭嵌入标准化
            )
            drug_dim = dataset.drug_embeddings.shape[1]
            target_dim = dataset.target_embeddings.shape[1]
            print(f"✅ MOA数据集加载成功 | Drug 嵌入维度: {drug_dim} | Target 嵌入维度: {target_dim}")
            print(f"   全局数据集：训练集{len(dataset.train_indices)} | 测试集{len(dataset.test_indices)} | 总样本{len(dataset)}")
            
            # 打印MOA标签分布（二分类关键信息）
            all_labels = np.array(dataset.df[label_col])
            pos_ratio = (all_labels == 1).sum() / len(all_labels)
            print(f"   正样本比例：{pos_ratio:.2%} | 负样本比例：{1-pos_ratio:.2%}")
        except Exception as e:
            print(f"❌ MOA数据集加载失败：{e}")
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            return

        # MOA全局测试集加载器（整轮实验只创建一次）
        def collate_fn(batch):
            point_clouds, drug_embs, target_embs, labels = zip(*batch)
            point_clouds = torch.stack(point_clouds)
            drug_embs = torch.stack(drug_embs)
            target_embs = torch.stack(target_embs)
            labels = torch.tensor(labels, dtype=torch.float32)  # BCE需要float类型
            return point_clouds, drug_embs, target_embs, labels

        fixed_test_subset = Subset(dataset, dataset.test_indices)
        fixed_test_loader = DataLoader(
            fixed_test_subset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # K折交叉验证（仅对全局训练集做K折划分）
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_metrics = {  # 记录每折的最终测试集指标
            'auc': [], 'auprc': [], 'precision': [], 
            'recall': [], 'f1': [], 'accuracy': []
        }

        # 遍历每一折
        for fold_idx, (_, _) in enumerate(kf.split(dataset.train_indices)):
            # MOA专属：验证集合并入训练集（无单独验证集）
            fold_train_indices = dataset.train_indices
            print("-" * 80)
            print(f"第 {fold_idx + 1}/{num_folds} 折训练 | 训练样本数：{len(fold_train_indices)}")
            print("-" * 80)

            # 构建该折的训练集加载器
            train_subset = Subset(dataset, fold_train_indices)
            
            # MOA类别加权采样（解决1:10类别不平衡）
            if use_class_weight:
                train_labels = [int(dataset.df[label_col].iloc[idx]) for idx in fold_train_indices]
                class_counts = np.bincount(train_labels)
                class_weights = 1.0 / class_counts
                sample_weights = [class_weights[label] for label in train_labels]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=torch.Generator().manual_seed(42)
                )
                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    sampler=sampler,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=collate_fn
                )
            else:
                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=collate_fn
                )

            # 初始化MOA专属模型
            model = SeqSpaPoint(
                drug_dim=drug_dim, 
                target_dim=target_dim, 
                k=10, 
                dropout_rate=dropout_rate, 
                num_queries=6
            ).to(device)

            # 优化器（MOA专属参数）
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # 学习率调度（warmup + cosine decay）
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                    min_lr_ratio = 1e-4
                    return cosine_decay * (1 - min_lr_ratio) + min_lr_ratio
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            # 训练状态初始化（按MOA核心指标AUPRC跟踪最优）
            fold_best_auprc = 0.0
            fold_train_log = []
            patience_counter = 0
            best_model_path = os.path.join(model_dir, f"fold_{fold_idx + 1}_best_auprc.pth")

            # 开始MOA训练
            for epoch in range(num_epochs):
                current_epoch = epoch + 1

                # ========== 训练阶段 ==========
                model.train()
                total_train_loss = 0.0
                train_pred_probs, train_true = [], []

                with tqdm(train_loader, desc=f"Epoch {current_epoch}/{num_epochs} [Train]") as pbar:
                    for point_cloud, drug_emb, target_emb, labels in pbar:
                        # 数据移至设备
                        point_cloud = point_cloud.to(device)
                        drug_emb = drug_emb.to(device)
                        target_emb = target_emb.to(device)
                        labels = labels.to(device)

                        # 前向传播
                        optimizer.zero_grad()
                        pred_logits = model(point_cloud, drug_emb, target_emb)

                        # 计算MOA专属Focal Loss
                        if use_focal_loss:
                            loss = focal_binary_loss(
                                pred_logits, labels,
                                gamma=focal_gamma,
                                alpha=focal_alpha,
                                use_numerical_stability=use_numerical_stability
                            )
                        else:
                            loss = F.binary_cross_entropy_with_logits(pred_logits, labels)

                        # 反向传播
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        # 记录损失和预测结果
                        total_train_loss += loss.item()
                        pred_probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
                        train_pred_probs.extend(pred_probs)
                        train_true.extend(labels.cpu().numpy())

                        # 进度条显示（MOA专属）
                        pbar.set_postfix({
                            "loss": f"{loss.item():.6f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.6e}"
                        })

                # 学习率更新
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                # 计算训练集指标
                avg_train_loss = round(total_train_loss / len(train_loader), 6)
                train_metrics = calculate_binary_metrics(train_true, train_pred_probs)

                # ========== 测试阶段（MOA专属评估） ==========
                model.eval()
                test_pred_probs, test_true = [], []
                total_test_loss = 0.0

                with torch.no_grad():
                    with tqdm(fixed_test_loader, desc=f"Epoch {current_epoch}/{num_epochs} [Test]") as pbar:
                        for point_cloud, drug_emb, target_emb, labels in pbar:
                            # 数据移至设备
                            point_cloud = point_cloud.to(device)
                            drug_emb = drug_emb.to(device)
                            target_emb = target_emb.to(device)
                            labels = labels.to(device)

                            # 前向传播
                            pred_logits = model(point_cloud, drug_emb, target_emb)

                            # 计算损失
                            if use_focal_loss:
                                loss = focal_binary_loss(
                                    pred_logits, labels,
                                    gamma=focal_gamma,
                                    alpha=focal_alpha,
                                    use_numerical_stability=use_numerical_stability
                                )
                            else:
                                loss = F.binary_cross_entropy_with_logits(pred_logits, labels)

                            total_test_loss += loss.item()

                            # 记录预测结果
                            pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                            test_pred_probs.extend(pred_probs)
                            test_true.extend(labels.cpu().numpy())

                            pbar.set_postfix({"test_loss": f"{loss.item():.6f}"})

                # 计算测试集指标
                avg_test_loss = round(total_test_loss / len(fixed_test_loader), 6)
                test_metrics = calculate_binary_metrics(test_true, test_pred_probs)
                current_test_auprc = test_metrics['auprc']  # MOA核心评估指标

                # 打印本轮指标（MOA专属）
                print(f"\n===== Epoch {current_epoch} 指标汇总（MOA） =====")
                print(f"【训练集】 Loss：{avg_train_loss} | AUROC：{train_metrics['auc']:.4f} | AUPRC：{train_metrics['auprc']:.4f} | precision：{train_metrics['precision']:.4f} | recall：{train_metrics['recall']:.4f} | F1：{train_metrics['f1']:.4f} | Acc：{train_metrics['accuracy']:.4f}")
                print(f"【测试集】 Loss：{avg_test_loss} | AUROC：{test_metrics['auc']:.4f} | AUPRC：{test_metrics['auprc']:.4f} | precision：{test_metrics['precision']:.4f} | recall：{test_metrics['recall']:.4f} | F1：{test_metrics['f1']:.4f} | Acc：{test_metrics['accuracy']:.4f}")
                print(f" 当前学习率：{current_lr:.6e} | 本轮最优测试集AUPRC：{fold_best_auprc:.4f}")

                # 记录日志（MOA专属）
                epoch_log = {
                    "epoch": current_epoch,
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "current_lr": current_lr
                }
                fold_train_log.append(epoch_log)

                # ========== 早停 & 模型保存（MOA专属） ==========
                if current_test_auprc > fold_best_auprc:
                    fold_best_auprc = current_test_auprc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"第 {fold_idx + 1} 折最优模型更新！测试集AUPRC={fold_best_auprc:.4f}，保存路径：{best_model_path}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停触发！连续{patience}轮测试集AUPRC未提升，停止本轮训练")
                        break

            # ========== 该折最终处理（MOA专属） ==========
            # 1. 保存训练日志
            log_file_path = os.path.join(log_dir, f"fold_{fold_idx + 1}_train_log.json")
            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(fold_train_log), f, indent=2, ensure_ascii=False)

            # 2. 找到该折最优测试集指标
            best_test_metrics = None
            for log in fold_train_log:
                if log["test_metrics"]["auprc"] == fold_best_auprc:
                    best_test_metrics = log["test_metrics"]
                    break
            # 兜底：用最后一轮测试集指标
            if best_test_metrics is None:
                best_test_metrics = fold_train_log[-1]["test_metrics"]

            # 3. 记录该折最优测试集指标
            fold_metrics['auc'].append(best_test_metrics['auc'])
            fold_metrics['auprc'].append(best_test_metrics['auprc'])
            fold_metrics['precision'].append(best_test_metrics['precision'])
            fold_metrics['recall'].append(best_test_metrics['recall'])
            fold_metrics['f1'].append(best_test_metrics['f1'])
            fold_metrics['accuracy'].append(best_test_metrics['accuracy'])

            # 4. 打印该折最终结果（MOA专属）
            print(f"\n===== 第 {fold_idx + 1} 折最终结果（MOA测试集最优AUPRC） =====")
            print(f"  最优AUPRC：{fold_best_auprc:.4f}")
            print(f"  AUROC: {best_test_metrics['auc']:.4f} | AUPRC: {best_test_metrics['auprc']:.4f}")
            print(f"  precision: {best_test_metrics['precision']:.4f} | recall: {best_test_metrics['recall']:.4f} | F1: {best_test_metrics['f1']:.4f} | Accuracy: {best_test_metrics['accuracy']:.4f}")
            print(f"=====================================\n")

            # 5. 保存该折最优测试集指标日志
            test_log_path = os.path.join(log_dir, f"fold_{fold_idx + 1}_best_test_log.json")
            with open(test_log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "best_test_metrics": convert_numpy_types(best_test_metrics),
                    "fold_best_auprc": fold_best_auprc
                }, f, indent=2, ensure_ascii=False)

        # ========== 场景内K折结果汇总（MOA专属） ==========
        def safe_mean_std(arr):
            arr = np.array(arr, dtype=np.float64)
            return round(np.nanmean(arr), 4), round(np.nanstd(arr), 4)

        metrics_summary = {
            'AUROC': safe_mean_std(fold_metrics['auc']),
            'AUPRC': safe_mean_std(fold_metrics['auprc']),
            'Precision': safe_mean_std(fold_metrics['precision']),
            'Recall': safe_mean_std(fold_metrics['recall']),
            'F1': safe_mean_std(fold_metrics['f1']),
            'Accuracy': safe_mean_std(fold_metrics['accuracy'])
        }

        print("=" * 80)
        print(f"{split_type.upper()} 场景{num_folds}折交叉验证结果汇总（MOA测试集）")
        for name, (mean, std) in metrics_summary.items():
            print(f"  {name}: {mean} ± {std}")
        print("=" * 80 + "\n")

    # MOA实验结束
    print(f"===== MOA实验结束时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} =====")
    print(f"所有MOA实验完成！结果保存至：{BASE_SAVE_DIR}")
    sys.stdout.close()
    sys.stdout = sys.__stdout__