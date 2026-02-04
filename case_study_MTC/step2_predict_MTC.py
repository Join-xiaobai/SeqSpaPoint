import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ===================== 核心配置（根据你的实际路径修改） =====================
class Config:
    # 1. 输入输出路径
    MTC_BENCHMARK_PATH = "./data/mtc_dti_pred_input/mtc_dta_benchmark_with_embeddings.csv"  # 你已生成的基准集路径
    POINT_CLOUD_PLY_PATH = "../data_preprocessing/dta/davis/log_and_file/dta_point_cloud.ply"  # 点云文件路径
    MODEL_PATH = "../result/dta_experiment/davis/drug_cold/models/fold_1_best_composite.pth"  # 预训练模型路径
    PREDICTION_OUTPUT_DIR = "./mtc_prediction_results"  # 预测结果输出目录
    
    # 2. 模型参数（必须与训练时一致）
    DRUG_DIM = 384  # ChemBERTa嵌入维度
    TARGET_DIM = 1280  # ESM-2嵌入维度
    K = 10  # PointNet的K近邻数
    DROPOUT_RATE = 0.2
    NUM_QUERIES = 6
    
    # 3. 运行配置
    BATCH_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTI_THRESHOLD = 0.5  # DTI分类阈值（0.5为通用值）
    AFFINITY_MAX = 10000.0  # DAVIS数据集亲和力最大值

# ===================== 第一步：预处理MTC基准集（关键修复） =====================
def preprocess_mtc_benchmark(config):
    """预处理MTC基准集：只保留模型需要的列，移除字符串列，避免类型转换错误"""
    # 加载原始基准集
    mtc_df = pd.read_csv(config.MTC_BENCHMARK_PATH)
    print(f"📋 原始MTC基准集列名：{mtc_df.columns.tolist()[:10]}...")
    
    # 1. 提取需要的列：ID列 + 嵌入列 + 标签列（移除SMILES/序列等字符串列）
    # 嵌入列筛选
    drug_emb_cols = [col for col in mtc_df.columns if col.startswith("emb_drug_")]
    target_emb_cols = [col for col in mtc_df.columns if col.startswith("emb_target_")]
    
    # 核心列（仅保留数值/ID列，移除所有字符串列）
    core_cols = ["drug_id", "protein_id", "affinity"] + drug_emb_cols + target_emb_cols
    mtc_processed_df = mtc_df[core_cols].copy()
    
    # 2. 确保ID列类型正确
    mtc_processed_df["drug_id"] = mtc_processed_df["drug_id"].astype("int64")
    mtc_processed_df["protein_id"] = mtc_processed_df["protein_id"].astype("str")
    
    # 3. 确保所有嵌入列都是浮点数
    for col in drug_emb_cols + target_emb_cols:
        mtc_processed_df[col] = pd.to_numeric(mtc_processed_df[col], errors="coerce").fillna(0.0)
    
    # 4. 保存预处理后的临时文件（只用于模型预测）
    temp_path = "./data/mtc_dti_pred_input/mtc_benchmark_numeric_only.csv"
    mtc_processed_df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ MTC基准集预处理完成：")
    print(f"   - 移除字符串列（drug_smiles/target_sequence等）")
    print(f"   - 保留列数：{len(mtc_processed_df.columns)}（ID+嵌入+标签）")
    print(f"   - 样本数：{len(mtc_processed_df)}")
    print(f"   - 临时文件路径：{temp_path}")
    
    return mtc_df, temp_path  # 返回原始df（用于结果生成）和预处理文件路径

# ===================== 第二步：设置模型导入路径 =====================
def setup_model_path():
    """将model模块所在目录加入Python搜索路径"""
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(current_file_path))
    sys.path.append(root_dir)
    
    # 导入核心模型和数据集类
    global SeqSpaPoint, PointCloudInteractionDataset
    from model.SeqSpaPoint import SeqSpaPoint
    from dataset.interaction_dataset import PointCloudInteractionDataset

# ===================== 第三步：加载MTC数据集（使用预处理文件） =====================
def load_mtc_dataset(config, temp_benchmark_path):
    """加载预处理后的MTC基准集，避免字符串列导致的类型错误"""
    # 初始化数据集（使用预处理后的数值文件）
    dataset = PointCloudInteractionDataset(
        feature_csv_path=temp_benchmark_path,  # 使用预处理后的文件
        point_cloud_ply_path=config.POINT_CLOUD_PLY_PATH,
        task_type="dta",
        split_type="warm", # 不划分，仅传参避免报错
        test_size=0.01,  # 合法值，仅为通过参数校验
        label_col="affinity",
        id_cols=["drug_id", "protein_id"],  # 匹配预处理文件的列名
        standardize_embeddings=False
    )
    
    # 强制使用全部数据
    all_indices = list(range(len(dataset)))
    dataset.train_indices = all_indices
    dataset.test_indices = []
    full_dataset = Subset(dataset, all_indices)
    
    # 自定义批量处理函数
    def collate_fn(batch):
        point_clouds, drug_embs, target_embs, affinities = zip(*batch)
        point_clouds = torch.stack(point_clouds)
        drug_embs = torch.stack(drug_embs)
        target_embs = torch.stack(target_embs)
        affinities = torch.tensor(affinities, dtype=torch.float32)
        return point_clouds, drug_embs, target_embs, affinities
    
    # 创建DataLoader
    dataloader = DataLoader(
        full_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✅ MTC数据集加载完成：")
    print(f"   - 总预测样本数：{len(full_dataset)}")
    print(f"   - 点云文件路径：{config.POINT_CLOUD_PLY_PATH}")
    return dataset, dataloader

# ===================== 第四步：加载预训练模型 =====================
def load_pretrained_model(config):
    """加载预训练的SeqSpaPoint模型，设置为评估模式"""
    model = SeqSpaPoint(
        drug_dim=config.DRUG_DIM,
        target_dim=config.TARGET_DIM,
        k=config.K,
        dropout_rate=config.DROPOUT_RATE,
        num_queries=config.NUM_QUERIES
    ).to(config.DEVICE)
    
    # 加载模型权重
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ 预训练模型加载成功：")
    print(f"   - 模型路径：{config.MODEL_PATH}")
    print(f"   - 运行设备：{config.DEVICE}")
    return model

# ===================== 第五步：执行预测并生成结果 =====================
def predict_mtc_dta_dti(config, original_mtc_df, model, dataloader):
    """对MTC基准集执行预测，输出亲和力和DTI概率"""
    # 初始化结果存储列表
    all_pred_affinity = []
    all_dti_prob = []
    
    # 禁用梯度计算
    with torch.no_grad():
        with tqdm(dataloader, desc="MTC基准集预测进度") as pbar:
            for batch_idx, (point_cloud, drug_emb, target_emb, true_aff) in enumerate(pbar):
                # 数据移到指定设备
                point_cloud = point_cloud.to(config.DEVICE)
                drug_emb = drug_emb.to(config.DEVICE)
                target_emb = target_emb.to(config.DEVICE)
                
                # 模型预测亲和力
                pred_affinity = model(point_cloud, drug_emb, target_emb)
                
                # 计算DTI概率（归一化后sigmoid）
                norm_pred_aff = torch.clamp(pred_affinity / config.AFFINITY_MAX, 0, 1)
                dti_prob = torch.sigmoid(norm_pred_aff * 10)
                
                # 收集结果
                all_pred_affinity.extend(pred_affinity.cpu().numpy().flatten().tolist())
                all_dti_prob.extend(dti_prob.cpu().numpy().flatten().tolist())
    
    # 构建结果DataFrame（结合原始df的完整信息）
    result_df = pd.DataFrame({
        "sample_id": [f"MTC-DTA-{i+1:05d}" for i in range(len(original_mtc_df))],
        "drug_id": original_mtc_df["drug_id"].values,
        "target_id": original_mtc_df["protein_id"].values,
        "target_gene": original_mtc_df["target_gene"].values,
        "drug_smiles": original_mtc_df["drug_smiles"].values,  # 还原SMILES
        "target_sequence": original_mtc_df["target_sequence"].values,  # 还原序列
        "true_affinity": original_mtc_df["affinity"].values,
        "pred_affinity": all_pred_affinity,
        "dti_probability": all_dti_prob,
        "dti_label": (np.array(all_dti_prob) >= config.DTI_THRESHOLD).astype(int)
    })
    
    # 保存预测结果
    os.makedirs(config.PREDICTION_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.PREDICTION_OUTPUT_DIR, "mtc_dta_dti_predictions.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 打印统计信息
    print(f"\n✅ MTC基准集预测完成！")
    print(f"   - 结果保存路径：{output_path}")
    print(f"\n📊 预测结果统计：")
    print(f"   - 总预测样本数：{len(result_df)}")
    print(f"   - 真实亲和力范围：{result_df['true_affinity'].min():.4f} ~ {result_df['true_affinity'].max():.4f}")
    print(f"   - 预测亲和力范围：{result_df['pred_affinity'].min():.4f} ~ {result_df['pred_affinity'].max():.4f}")
    print(f"   - DTI概率范围：{result_df['dti_probability'].min():.4f} ~ {result_df['dti_probability'].max():.4f}")
    print(f"   - 预测为相互作用（DTI=1）的样本数：{result_df['dti_label'].sum()}")
    print(f"   - 预测为无相互作用（DTI=0）的样本数：{len(result_df) - result_df['dti_label'].sum()}")
    return result_df

# ===================== 主函数（一键执行预测） =====================
def main():
    print("=" * 80)
    print("          MTC-DTA基准集预测（加载预训练模型）")
    print("=" * 80)
    
    try:
        # 1. 初始化配置
        config = Config()
        
        # 2. 预处理MTC基准集（核心修复：移除字符串列）
        print("\n🔧 预处理MTC基准集（移除字符串列）...")
        original_mtc_df, temp_benchmark_path = preprocess_mtc_benchmark(config)
        
        # 3. 设置模型导入路径
        print("\n🔧 设置模型导入路径...")
        setup_model_path()
        
        # 4. 加载MTC数据集（使用预处理文件）
        print("\n📥 加载MTC数据集...")
        dataset, dataloader = load_mtc_dataset(config, temp_benchmark_path)
        
        # 5. 加载预训练模型
        print("\n🔧 加载预训练模型...")
        model = load_pretrained_model(config)
        
        # 6. 执行预测
        print("\n🚀 开始MTC基准集预测...")
        predict_mtc_dta_dti(config, original_mtc_df, model, dataloader)
        
        # 清理临时文件
        if os.path.exists(temp_benchmark_path):
            os.remove(temp_benchmark_path)
            print(f"\n🗑️  清理临时文件：{temp_benchmark_path}")
        
        print("\n🎉 MTC基准集预测全部完成！结果已保存至 mtc_prediction_results/")
    except Exception as e:
        print(f"\n❌ 预测失败：{str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()