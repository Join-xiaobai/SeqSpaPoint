import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

# ===================== 核心配置（仅需改这部分） =====================
class Config:
    # 1. 数据路径（原始MTC数据集，和微调时一致）
    MTC_FEATURE_PATH = "./data_preprocessing/dta/mtc/log_and_file/dta_features.csv"
    POINT_CLOUD_PLY_PATH = "./data_preprocessing/dta/mtc/log_and_file/dta_point_cloud.ply"
    
    # 2. 微调后模型路径
    FINETUNED_MODEL_PATH = "./step2_mtc_drug_cold_finetune_results/models/mtc_drug_cold_best_model.pth"
    
    # 3. 预测结果输出路径
    PREDICTION_OUTPUT_DIR = "./step2_mtc_drug_cold_finetune_results/predictions_finetuned"
    
    # 4. 模型参数（必须和微调时一致）
    DRUG_DIM = 384
    TARGET_DIM = 1280
    K = 10
    NUM_QUERIES = 6
    DROPOUT_RATE = 0.2
    
    # 5. 预测参数（核心修复：基于pIC50逻辑重构）
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    DTI_THRESHOLD = 0.5  # DTI分类阈值
    # 核心修复1：明确原始IC50（nM）的合理范围（基于你的数据集）
    RAW_IC50_MIN = 0.0    # 原始IC50最小值（nM）
    RAW_IC50_MAX = 4.5    # 原始IC50最大值（nM）
    # 核心修复2：pIC50的合理范围（5~12是行业标准）
    PIC50_MIN = 5.0       
    PIC50_MAX = 12.0
    SEED = 42
    DTI_SCALE_FACTOR = 4  # 修复：放大系数从8→4，避免概率过度饱和
    # 新增：Monte Carlo Dropout采样次数（置信度估计）
    MC_DROPOUT_SAMPLES = 5

# ===================== 1. 加载微调后的最优模型 =====================
def load_finetuned_model(config):
    """加载微调后基于MSE最优的模型（仅预测，无训练）"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, ".."))
    sys.path.append("./model")
    sys.path.append("./dataset")
    
    from model.SeqSpaPoint import SeqSpaPoint
    model = SeqSpaPoint(
        drug_dim=config.DRUG_DIM,
        target_dim=config.TARGET_DIM,
        k=config.K,
        dropout_rate=config.DROPOUT_RATE,
        num_queries=config.NUM_QUERIES
    ).to(config.DEVICE)
    
    checkpoint = torch.load(
        config.FINETUNED_MODEL_PATH, 
        map_location=config.DEVICE,
        weights_only=False
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ 微调后最优模型加载完成：{config.FINETUNED_MODEL_PATH}")
    return model

# ===================== 2. 加载MTC数据集（仅用于预测） =====================
def load_mtc_dataset_for_prediction(config):
    """加载全部MTC数据集用于预测（和微调时的数据集加载逻辑一致）"""
    from dataset.interaction_dataset import PointCloudInteractionDataset
    
    mtc_df = pd.read_csv(config.MTC_FEATURE_PATH, encoding="utf-8")
    mtc_df["drug_id"] = mtc_df["drug_id"].astype(str)
    mtc_df["protein_id"] = mtc_df["protein_id"].astype(str)
    mtc_df["affinity"] = pd.to_numeric(mtc_df["affinity"], errors="coerce").fillna(0.0)
    
    # 核心修复：提前计算真实pIC50（避免后续重复计算）
    # 替换0值为极小值，避免log10(0)报错
    mtc_df["affinity_safe"] = mtc_df["affinity"].replace(0, 1e-10)
    # 原始IC50（nM）转pIC50：pIC50 = -log10(IC50 × 10^-9)
    mtc_df["true_pIC50"] = -np.log10(mtc_df["affinity_safe"] * 1e-9)
    # 过滤异常pIC50值
    mtc_df["true_pIC50"] = mtc_df["true_pIC50"].clip(config.PIC50_MIN, config.PIC50_MAX)
    
    dataset = PointCloudInteractionDataset(
        feature_csv_path=config.MTC_FEATURE_PATH,
        point_cloud_ply_path=config.POINT_CLOUD_PLY_PATH,
        task_type="dta",
        split_type="drug_cold",
        test_size=0.01,
        label_col="affinity",
        id_cols=["drug_id", "protein_id"],
        standardize_embeddings=False
    )
    
    def collate_fn(batch):
        point_clouds, drug_embs, target_embs, affinities = zip(*batch)
        point_clouds = torch.stack(point_clouds)
        drug_embs = torch.stack(drug_embs)
        target_embs = torch.stack(target_embs)
        affinities = torch.tensor(affinities, dtype=torch.float32)
        return point_clouds, drug_embs, target_embs, affinities
    
    all_indices = list(range(len(dataset)))
    full_subset = Subset(dataset, all_indices)
    dataloader = DataLoader(
        full_subset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn, 
        pin_memory=False,
        num_workers=0
    )
    
    print(f"✅ MTC数据集加载完成（仅预测）：")
    print(f"   - 总预测样本数：{len(full_subset)}")
    print(f"   - 真实IC50范围：{mtc_df['affinity'].min():.4f} ~ {mtc_df['affinity'].max():.4f} nM")
    print(f"   - 真实pIC50范围：{mtc_df['true_pIC50'].min():.4f} ~ {mtc_df['true_pIC50'].max():.4f}")
    return mtc_df, dataloader

# ===================== 3. 核心预测函数（彻底修复数值转换+列名规范+置信度） =====================
def predict_mtc_dta(config, original_mtc_df, model, dataloader):
    """
    最终版核心修复：
    1. 明确区分「模型原始输出」和「有物理意义的预测值」
    2. 列名全量规范化，标注物理意义+单位
    3. 补充pIC50→IC50(nM)的反向转换，便于验证
    4. 修复log10(≤0)溢出风险（二次clamp防护）
    5. 新增Monte Carlo Dropout置信度估计
    """
    all_model_raw_output = []    # 模型未转换的原始输出（无物理意义）
    all_pred_pIC50 = []          # 模型最终预测的pIC50（核心结果，有物理意义）
    all_pred_pIC50_std = []      # 预测pIC50的标准差（置信度）
    all_pred_IC50_nM = []        # 预测pIC50转回的IC50(nM)（便于对比）
    all_dti_prob = []            # DTI概率（基于pIC50计算）
    
    with torch.no_grad():
        with tqdm(dataloader, desc="MTC-DTA微调后模型预测进度") as pbar:
            for batch_idx, (point_cloud, drug_emb, target_emb, true_aff) in enumerate(pbar):
                # 移数据到GPU
                point_cloud = point_cloud.to(config.DEVICE)
                drug_emb = drug_emb.to(config.DEVICE)
                target_emb = target_emb.to(config.DEVICE)
                true_aff = true_aff.to(config.DEVICE)
                
                # 1. 获取模型原始输出（无物理意义的中间值）
                model_raw_output = model(point_cloud, drug_emb, target_emb)
                all_model_raw_output.extend(model_raw_output.cpu().numpy().flatten().tolist())
                
                # 2. 模型原始输出 → 预测pIC50（有物理意义的核心结果）
                # 步骤1：先将模型原始输出映射到IC50(nM)范围（0~4.5）
                pred_IC50_nM = torch.clamp(model_raw_output, config.RAW_IC50_MIN, config.RAW_IC50_MAX)
                
                # 步骤2：核心修复——二次clamp防护，彻底杜绝log10(≤0)
                # 强制最小值1e-12（对应pIC50=12.0），最大值1e6（对应pIC50=3.0）
                pred_IC50_nM_safe = torch.clamp(pred_IC50_nM, min=1e-12, max=1e6)
                
                # 步骤3：标准转换公式得到pIC50
                pred_pIC50 = -torch.log10(pred_IC50_nM_safe * 1e-9)
                pred_pIC50 = torch.clamp(pred_pIC50, config.PIC50_MIN, config.PIC50_MAX)
                
                # 3. 新增：Monte Carlo Dropout置信度估计
                model.train()  # 启用Dropout采样
                pred_pIC50_samples = []
                for _ in range(config.MC_DROPOUT_SAMPLES):
                    sample_raw_output = model(point_cloud, drug_emb, target_emb)
                    sample_IC50_nM = torch.clamp(sample_raw_output, config.RAW_IC50_MIN, config.RAW_IC50_MAX)
                    sample_IC50_nM_safe = torch.clamp(sample_IC50_nM, min=1e-12, max=1e6)
                    sample_pIC50 = -torch.log10(sample_IC50_nM_safe * 1e-9)
                    sample_pIC50 = torch.clamp(sample_pIC50, config.PIC50_MIN, config.PIC50_MAX)
                    pred_pIC50_samples.append(sample_pIC50.cpu().numpy())
                model.eval()  # 恢复eval模式
                # 计算标准差（置信度：值越小，预测越可信）
                pred_pIC50_std = np.std(np.stack(pred_pIC50_samples), axis=0)
                
                # 4. 预测pIC50 → 转回IC50(nM)（便于和真实值对比）
                pred_IC50_nM_from_pIC50 = 10 ** (-pred_pIC50 / 1) / 1e-9
                pred_IC50_nM_from_pIC50 = torch.clamp(pred_IC50_nM_from_pIC50, config.RAW_IC50_MIN, config.RAW_IC50_MAX)
                
                # 5. 基于pIC50计算DTI概率（更合理）
                norm_pred_pIC50 = (pred_pIC50 - config.PIC50_MIN) / (config.PIC50_MAX - config.PIC50_MIN)
                dti_prob = torch.sigmoid(norm_pred_pIC50 * config.DTI_SCALE_FACTOR)
                
                # 收集结果
                all_pred_pIC50.extend(pred_pIC50.cpu().numpy().flatten().tolist())
                all_pred_pIC50_std.extend(pred_pIC50_std.flatten().tolist())
                all_pred_IC50_nM.extend(pred_IC50_nM_from_pIC50.cpu().numpy().flatten().tolist())
                all_dti_prob.extend(dti_prob.cpu().numpy().flatten().tolist())
    
    # 构建结果DataFrame（列名100%规范，标注物理意义+单位）
    result_df = pd.DataFrame({
        # 基础信息
        "sample_id": [f"MTC-DTA-{i+1:05d}" for i in range(len(original_mtc_df))],
        "drug_id": original_mtc_df["drug_id"].values,
        "target_id": original_mtc_df["protein_id"].values,
        
        # 真实值（列名明确标注单位）
        "true_IC50_nM": original_mtc_df["affinity"].values,          # 实验测得的IC50 (nM)
        "true_pIC50": original_mtc_df["true_pIC50"].values,          # 实验值转换的pIC50
        
        # 模型输出（明确区分原始输出和有物理意义的预测值）
        "model_raw_output": all_model_raw_output,                    # 模型未转换的原始输出（无单位）
        "pred_pIC50": all_pred_pIC50,                                # 模型最终预测的pIC50（核心结果）
        "pred_pIC50_std": all_pred_pIC50_std,                        # 预测pIC50的标准差（置信度）
        "pred_IC50_nM": all_pred_IC50_nM,                            # 预测pIC50转回的IC50 (nM)
        
        # DTI相关
        "dti_probability": all_dti_prob,
        "dti_label": (np.array(all_dti_prob) >= config.DTI_THRESHOLD).astype(int)
    })
    
    # 保存预测结果
    os.makedirs(config.PREDICTION_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.PREDICTION_OUTPUT_DIR, "mtc_dta_finetuned_predictions.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 核心评估指标：真实vs预测的相关性（基于有物理意义的数值）
    valid_mask = result_df['true_IC50_nM'] > 0
    corr_ic50 = result_df.loc[valid_mask, 'true_IC50_nM'].corr(result_df.loc[valid_mask, 'pred_IC50_nM'])
    corr_pIC50 = result_df.loc[valid_mask, 'true_pIC50'].corr(result_df.loc[valid_mask, 'pred_pIC50'])
    
    # 打印统计信息（突出有物理意义的数值）
    print(f"\n✅ MTC-DTA微调后模型预测完成！")
    print(f"   - 结果保存路径：{output_path}")
    print(f"\n📊 MTC-DTA预测结果统计（列名规范化后）：")
    print(f"   - 总预测样本数：{len(result_df)}")
    print(f"   - 模型原始输出范围：{result_df['model_raw_output'].min():.4f} ~ {result_df['model_raw_output'].max():.4f}（无单位）")
    print(f"   - 真实IC50范围：{result_df['true_IC50_nM'].min():.4f} ~ {result_df['true_IC50_nM'].max():.4f} nM")
    print(f"   - 预测IC50范围：{result_df['pred_IC50_nM'].min():.4f} ~ {result_df['pred_IC50_nM'].max():.4f} nM")
    print(f"   - 真实pIC50范围：{result_df['true_pIC50'].min():.4f} ~ {result_df['true_pIC50'].max():.4f}")
    print(f"   - 预测pIC50范围：{result_df['pred_pIC50'].min():.4f} ~ {result_df['pred_pIC50'].max():.4f}")
    print(f"   - 预测pIC50标准差（置信度）：{result_df['pred_pIC50_std'].min():.4f} ~ {result_df['pred_pIC50_std'].max():.4f}")
    print(f"   - DTI概率范围：{result_df['dti_probability'].min():.4f} ~ {result_df['dti_probability'].max():.4f}")
    print(f"   - 真实vs预测IC50相关系数：{corr_ic50:.4f}")
    print(f"   - 真实vs预测pIC50相关系数：{corr_pIC50:.4f}（核心评估指标）")
    print(f"   - 预测为相互作用（DTI=1）的样本数：{result_df['dti_label'].sum()}")
    print(f"   - 预测为无相互作用（DTI=0）的样本数：{len(result_df) - result_df['dti_label'].sum()}")
    
    # 打印关键样本验证（第26行）
    if len(result_df) >= 26:
        sample_26 = result_df.iloc[25]  # 索引从0开始
        print(f"\n🔍 关键样本验证（第26行）：")
        print(f"   - 真实IC50(nM)：{sample_26['true_IC50_nM']:.4f}")
        print(f"   - 真实pIC50：{sample_26['true_pIC50']:.4f}")
        print(f"   - 模型原始输出：{sample_26['model_raw_output']:.4f}（无单位）")
        print(f"   - 预测pIC50：{sample_26['pred_pIC50']:.4f}（核心结果）")
        print(f"   - 预测pIC50标准差：{sample_26['pred_pIC50_std']:.4f}（置信度）")
        print(f"   - 预测IC50(nM)：{sample_26['pred_IC50_nM']:.4f}（反向转换）")
    
    return result_df

# ===================== 4. 主函数（独立运行预测+单位一致性校验） =====================
def main():
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
    
    # 初始化配置
    config = Config()
    
    # 加载模型和数据
    model = load_finetuned_model(config)
    original_mtc_df, dataloader = load_mtc_dataset_for_prediction(config)
    
    # 执行预测（最终版修复）
    result_df = predict_mtc_dta(config, original_mtc_df, model, dataloader)
    
    # 核心增强：自动校验单位一致性（NC强制要求）
    print(f"\n🔍 NC投稿级单位一致性校验：")
    # 过滤0值避免校验误差
    valid_mask = result_df['true_IC50_nM'] > 0
    # 校验IC50→pIC50转换可逆性
    reverse_calc_IC50 = 10 ** (-result_df['true_pIC50'] / 1) / 1e-9
    is_conversion_valid = np.allclose(
        result_df.loc[valid_mask, 'true_IC50_nM'],
        reverse_calc_IC50.loc[valid_mask],
        atol=1e-3
    )
    
    if is_conversion_valid:
        print(f"✅ IC50↔pIC50 转换可逆性校验通过（误差<1e-3）")
    else:
        print(f"❌ IC50↔pIC50 转换可逆性校验失败！")
    
    # 校验核心列是否存在
    required_cols = ['true_IC50_nM', 'true_pIC50', 'pred_pIC50', 'pred_IC50_nM']
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if not missing_cols:
        print(f"✅ 核心列完整性校验通过（所有NC要求列均存在）")
    else:
        print(f"❌ 核心列完整性校验失败！缺失列：{missing_cols}")
    
    print("\n🎉 全部预测完成！最终结果已保存至：", config.PREDICTION_OUTPUT_DIR)

if __name__ == "__main__":
    main()