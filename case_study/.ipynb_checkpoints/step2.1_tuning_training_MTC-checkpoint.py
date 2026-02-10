import os
import sys
import json
import time
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ===================== 修复PyTorch 2.6 torch.load安全限制 =====================
def fix_torch_load_security():
    """解决PyTorch 2.6+ torch.load的weights_only和numpy scalar反序列化限制"""
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np._core.multiarray.scalar])
    except (ImportError, AttributeError):
        pass

fix_torch_load_security()

# ===================== 核心配置（仅保留微调相关参数） =====================
class Config:
    # 1. 数据路径（保持不变）
    MTC_FEATURE_PATH = "./data_preprocessing/dta/mtc/log_and_file/dta_features.csv"
    POINT_CLOUD_PLY_PATH = "./data_preprocessing/dta/mtc/log_and_file/dta_point_cloud.ply"
    PRETRAINED_MODEL_PATH = "../result/dta_experiment/kiba/drug_cold/models/fold_1_best_composite.pth"
    
    # 2. 输出路径（仅保留微调输出路径，删除预测输出路径）
    OUTPUT_DIR = "./step2_mtc_drug_cold_finetune_results"  
    
    # 3. 微调参数（核心调整）
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    LR = 1e-4  # 【调整1】从1e-5→1e-4，解决梯度更新不明显
    WARMUP_EPOCHS = 5
    PATIENCE = 30  # 【调整2】从20→30，给模型更多学习时间
    NUM_FOLDS = 5
    SEED = 42
    DROPOUT_RATE = 0.2
    WEIGHT_DECAY = 1e-4  # 【新增】添加权重衰减，防止过拟合
    
    # 4. 模型参数（保持不变）
    DRUG_DIM = 384
    TARGET_DIM = 1280
    K = 10
    NUM_QUERIES = 6
    
    # 5. 仅保留微调相关的设备参数，删除预测相关参数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AFFINITY_MAX = 4.5
    AFFINITY_MIN = 0.0  # 约束亲和力最小值

# ===================== 1. 加载模型（仅保留预训练模型加载） =====================
def load_pretrained_model(config):
    """加载预训练模型（强制开启所有参数梯度更新）"""
    from model.SeqSpaPoint import SeqSpaPoint
    model = SeqSpaPoint(
        drug_dim=config.DRUG_DIM,
        target_dim=config.TARGET_DIM,
        k=config.K,
        dropout_rate=config.DROPOUT_RATE,
        num_queries=config.NUM_QUERIES
    ).to(config.DEVICE)
    
    # 加载权重（保持不变）
    checkpoint = torch.load(
        config.PRETRAINED_MODEL_PATH, 
        map_location=config.DEVICE,
        weights_only=False
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # 【核心调整】强制开启所有参数的梯度更新，解决参数冻结
    for param in model.parameters():
        param.requires_grad = True
    
    print(f"✅ 预训练模型加载完成：{config.PRETRAINED_MODEL_PATH}")
    print(f"✅ 所有模型参数已开启梯度更新")
    return model

# ===================== 2. 加载数据集（仅保留微调模式） =====================
def load_mtc_dataset(config):
    """加载MTC数据集（仅用于微调，删除预测模式）"""
    from dataset.interaction_dataset import PointCloudInteractionDataset
    
    # 预处理（保持不变）
    mtc_df = pd.read_csv(config.MTC_FEATURE_PATH, encoding="utf-8")
    mtc_df["drug_id"] = mtc_df["drug_id"].astype(str)
    mtc_df["protein_id"] = mtc_df["protein_id"].astype(str)
    mtc_df["affinity"] = pd.to_numeric(mtc_df["affinity"], errors="coerce").fillna(0.0)
    
    # 初始化数据集（仅用于微调）
    dataset = PointCloudInteractionDataset(
        feature_csv_path=config.MTC_FEATURE_PATH,
        point_cloud_ply_path=config.POINT_CLOUD_PLY_PATH,
        task_type="dta",
        split_type="drug_cold",
        test_size=0.1,
        label_col="affinity",
        id_cols=["drug_id", "protein_id"],
        standardize_embeddings=False
    )
    
    # collate_fn（保持不变）
    def collate_fn(batch):
        point_clouds, drug_embs, target_embs, affinities = zip(*batch)
        point_clouds = torch.stack(point_clouds)
        drug_embs = torch.stack(drug_embs)
        target_embs = torch.stack(target_embs)
        affinities = torch.tensor(affinities, dtype=torch.float32)
        return point_clouds, drug_embs, target_embs, affinities
    
    train_subset = Subset(dataset, dataset.train_indices)
    test_subset = Subset(dataset, dataset.test_indices)
    
    # 打印drug_cold划分详情
    train_drugs = set([dataset.df.iloc[i]['drug_id'] for i in dataset.train_indices])
    test_drugs = set([dataset.df.iloc[i]['drug_id'] for i in dataset.test_indices])
    print(f"🔍 Drug-cold划分详情：")
    print(f"   - 训练集药物数：{len(train_drugs)} | 样本数：{len(train_subset)}")
    print(f"   - 测试集药物数：{len(test_drugs)} | 样本数：{len(test_subset)}")
    print(f"   - 测试集亲和力范围：{dataset.df.iloc[dataset.test_indices]['affinity'].min():.4f} ~ {dataset.df.iloc[dataset.test_indices]['affinity'].max():.4f}")
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        pin_memory=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_subset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn, 
        pin_memory=False,
        num_workers=0
    )
    
    print(f"✅ MTC数据集加载完成（drug_cold微调）：")
    print(f"   - 训练集样本数：{len(train_subset)}")
    print(f"   - 测试集样本数：{len(test_subset)}")
    return train_loader, test_loader

# ===================== 3. 损失函数（保持不变） =====================
def soft_high_affinity_focal_rank_loss(y_pred, y_true, gamma=1.5, threshold=2.0, scale=2.5):
    """适配低亲和力样本"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    N = y_pred.size(0)
    if N < 2:
        return torch.tensor(0.0, device=y_pred.device)
    
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    labels = (true_diff > 0).float()
    valid_mask = (true_diff != 0).float()
    
    avg_aff = (y_true.unsqueeze(1) + y_true.unsqueeze(0)) / 2.0
    weight = torch.sigmoid(scale * (avg_aff - threshold))
    
    p = torch.sigmoid(pred_diff)
    pt = labels * p + (1 - labels) * (1 - p)
    focal_modulation = (1 - pt + 1e-8) ** gamma
    
    bce_loss = F.binary_cross_entropy_with_logits(pred_diff, labels, reduction='none')
    loss = (focal_modulation * bce_loss * weight * valid_mask).sum()
    norm = (weight * valid_mask).sum() + 1e-8
    return loss / norm

# 低亲和力约束损失：防止预测值为负
def low_affinity_constraint_loss(y_pred, min_aff=0.0):
    below_min_mask = (y_pred < min_aff).float()
    penalty = torch.square(y_pred - min_aff) * below_min_mask
    return torch.mean(penalty)

# ===================== 4. 微调逻辑（保持不变） =====================
def finetune_model(config):
    """微调逻辑（约束预测值+优化损失权重）"""
    # 创建输出目录（保持不变）
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "logs"), exist_ok=True)
    
    # 加载数据和模型（保持不变）
    train_loader, test_loader = load_mtc_dataset(config)
    model = load_pretrained_model(config)
    
    # 优化器（保持不变）
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LR, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度（保持不变）
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        else:
            progress = (epoch - config.WARMUP_EPOCHS) / max(1, config.NUM_EPOCHS - config.WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 微调记录（保持不变）
    best_mse = float('inf')
    patience_counter = 0
    train_log = []
    
    print("\n🚀 开始MTC-DTA微调（drug_cold，100轮，按MSE保存最优）：")
    print(f"   - 设备：{config.DEVICE}")
    print(f"   - 学习率：{config.LR}")
    print(f"   - 输出目录：{config.OUTPUT_DIR}")
    print("="*80)
    
    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]") as pbar:
            for batch_idx, (point_cloud, drug_emb, target_emb, affinity) in enumerate(pbar):
                # 移设备（保持不变）
                point_cloud = point_cloud.to(config.DEVICE, non_blocking=True)
                drug_emb = drug_emb.to(config.DEVICE, non_blocking=True)
                target_emb = target_emb.to(config.DEVICE, non_blocking=True)
                affinity = affinity.to(config.DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                pred = model(point_cloud, drug_emb, target_emb)
                
                # 约束预测值范围
                pred_clamped = torch.clamp(pred, config.AFFINITY_MIN, config.AFFINITY_MAX)
                
                # 计算损失
                smoothl1_loss = F.smooth_l1_loss(pred_clamped, affinity)
                rank_loss = soft_high_affinity_focal_rank_loss(pred_clamped, affinity)
                low_aff_loss = low_affinity_constraint_loss(pred, config.AFFINITY_MIN)
                loss = smoothl1_loss + 0.5 * rank_loss + 0.5 * low_aff_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        # 验证阶段
        model.eval()
        test_mse = 0.0
        total_samples = 0
        with torch.no_grad():
            for point_cloud, drug_emb, target_emb, affinity in test_loader:
                # 移设备（保持不变）
                point_cloud = point_cloud.to(config.DEVICE, non_blocking=True)
                drug_emb = drug_emb.to(config.DEVICE, non_blocking=True)
                target_emb = target_emb.to(config.DEVICE, non_blocking=True)
                affinity = affinity.to(config.DEVICE, non_blocking=True)
                
                pred = model(point_cloud, drug_emb, target_emb)
                pred_clamped = torch.clamp(pred, config.AFFINITY_MIN, config.AFFINITY_MAX)
                
                # 计算MSE
                mse = F.mse_loss(pred_clamped, affinity, reduction='sum')
                test_mse += mse.item()
                total_samples += affinity.size(0)
        
        # 计算平均MSE
        test_mse /= total_samples
        scheduler.step()
        
        # 保存最优模型
        if test_mse < best_mse:
            best_mse = test_mse
            patience_counter = 0
            torch.save({
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "mse": best_mse,
                "optimizer": optimizer.state_dict()
            }, os.path.join(config.OUTPUT_DIR, "models", "mtc_drug_cold_best_model.pth"))
            print(f"📌 最优模型更新：Epoch {epoch+1}，MSE = {best_mse:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"⚠️  早停触发（{config.PATIENCE}轮MSE无提升），结束微调")
                break
        
        # 记录日志
        log = {
            "epoch": epoch+1,
            "train_loss": total_loss / len(train_loader),
            "test_mse": test_mse,
            "best_mse": best_mse,
            "lr": optimizer.param_groups[0]['lr']
        }
        train_log.append(log)
        
        print(f"Epoch {epoch+1} | 训练损失：{log['train_loss']:.6f} | 测试MSE：{test_mse:.6f} | 最优MSE：{best_mse:.6f}")
    
    # 保存日志和模型
    with open(os.path.join(config.OUTPUT_DIR, "logs", "finetune_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "models", "mtc_drug_cold_final_model.pth"))
    
    print("\n🎉 微调完成！结果保存至：")
    print(f"   - 最优模型（MSE）：{os.path.join(config.OUTPUT_DIR, 'models', 'mtc_drug_cold_best_model.pth')}")
    print(f"   - 最优MSE值：{best_mse:.6f}")
    print(f"   - 训练日志：{os.path.join(config.OUTPUT_DIR, 'logs', 'finetune_log.json')}")

# ===================== 5. 主函数（仅保留微调逻辑） =====================
def main():
    # 设置随机种子（保持不变）
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
    
    # 添加路径（保持不变）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, ".."))
    sys.path.append("./model")
    sys.path.append("./dataset")
    
    # 初始化配置（保持不变）
    config = Config()
    
    # 仅执行微调，删除预测相关代码
    finetune_model(config)
    
    print("\n🎉 微调任务全部完成！所有结果已保存至：", config.OUTPUT_DIR)

if __name__ == "__main__":
    main()