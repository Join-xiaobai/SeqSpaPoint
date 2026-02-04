import os
import json
import re
import pickle
import pandas as pd
import numpy as np

# ===================== 全局配置（只需改这里） =====================
# 基础路径
BASE_DIR = "./data"
DAVIS_DIR = os.path.join(BASE_DIR, "davis")
TRAIN_FEATURE_PATH = "../data_preprocessing/dta/davis/log_and_file/dta_features.csv"  # 嵌入特征文件路径
OUTPUT_DIR = os.path.join(BASE_DIR, "mtc_dti_pred_input")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MTC核心靶点基因（仅保留这5个）
MTC_CORE_GENES = ["RET", "KDR", "MET", "MAP2K1", "MAP2K2"]

# 最终输出文件名（带嵌入特征）
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mtc_dta_benchmark_with_embeddings.csv")

# ===================== 工具函数 =====================
def load_davis_raw():
    """加载原始DAVIS数据（单行JSON+pickle），返回亲和力DataFrame（统一用protein_id）"""
    # 1. 加载药物SMILES
    drug_path = os.path.join(DAVIS_DIR, "ligands_can.txt")
    with open(drug_path, "r", encoding="utf-8") as f:
        drug_dict = json.loads(f.read().strip())
    drug_dict = {did: smi.strip() for did, smi in drug_dict.items() if smi.strip()}
    drug_list = sorted(drug_dict.keys())

    # 2. 加载靶点序列（统一列名为protein_id）
    target_path = os.path.join(DAVIS_DIR, "proteins.txt")
    with open(target_path, "r", encoding="utf-8") as f:
        target_dict = json.loads(f.read().strip())
    target_dict = {tid: seq.strip() for tid, seq in target_dict.items() if seq.strip()}
    target_list = sorted(target_dict.keys())

    # 3. 加载亲和力矩阵
    y_path = os.path.join(DAVIS_DIR, "Y")
    with open(y_path, "rb") as f:
        Y = pickle.load(f, encoding="latin1")
    Y = np.array(Y, dtype=np.float64)
    
    # 验证维度
    assert Y.shape == (len(drug_list), len(target_list)), \
        f"矩阵维度不匹配！矩阵{Y.shape} vs 药物{len(drug_list)}×靶点{len(target_list)}"

    # 构建DataFrame（核心：靶点列名统一为protein_id）
    affinity_df = pd.DataFrame(Y, index=drug_list, columns=target_list).stack().reset_index()
    affinity_df.columns = ["drug_id", "protein_id", "affinity"]  # 统一用protein_id
    
    # 关键修复1：将drug_id转为整数类型（匹配训练特征文件）
    affinity_df["drug_id"] = pd.to_numeric(affinity_df["drug_id"], errors="coerce").astype("int64")
    # 关键修复2：确保protein_id是字符串类型（DAVIS靶点ID都是字符串）
    affinity_df["protein_id"] = affinity_df["protein_id"].astype("str")
    
    # 过滤异常值
    affinity_df = affinity_df[
        (affinity_df["affinity"] > 0) & 
        (affinity_df["affinity"].notna()) &
        (np.isfinite(affinity_df["affinity"])) &
        (affinity_df["drug_id"].notna())  # 过滤转换失败的drug_id
    ]

    print(f"✅ Davis原始数据加载完成：")
    print(f"   - 药物总数：{len(drug_dict)}")
    print(f"   - 靶点总数：{len(target_dict)}")
    print(f"   - 有效亲和力数据：{len(affinity_df)} 条")
    print(f"   - drug_id类型：{affinity_df['drug_id'].dtype}")
    print(f"   - protein_id类型：{affinity_df['protein_id'].dtype}")
    return drug_dict, target_dict, affinity_df

def filter_mtc_samples(drug_dict, target_dict, affinity_df):
    """筛选MTC核心靶点和相关药物，返回MTC基准集（带SMILES/序列）"""
    # 1. 匹配MTC核心靶点（protein_id）
    mtc_protein_ids = []
    gene_pattern = {gene: re.compile(rf"\b{gene}\b", re.IGNORECASE) for gene in MTC_CORE_GENES}

    for tid, seq in target_dict.items():
        for gene, pattern in gene_pattern.items():
            if pattern.search(tid) or pattern.search(seq):
                if tid in affinity_df["protein_id"].values:
                    mtc_protein_ids.append(tid)
                    break

    mtc_protein_ids = sorted(list(set(mtc_protein_ids)))
    print(f"\n🔍 匹配到MTC核心靶点（有亲和力数据）：{len(mtc_protein_ids)} 个")
    for i, tid in enumerate(mtc_protein_ids, 1):
        print(f"   {i}. {tid}")
    
    if not mtc_protein_ids:
        raise ValueError("❌ 未匹配到任何MTC核心靶点，请检查基因名")

    # 2. 匹配MTC相关药物
    mtc_drug_ids = affinity_df[
        affinity_df["protein_id"].isin(mtc_protein_ids)
    ]["drug_id"].unique()
    mtc_drug_ids = sorted(list(set(mtc_drug_ids)))
    print(f"\n🔍 匹配到MTC相关药物：{len(mtc_drug_ids)} 个（展示前10个）")
    for i, did in enumerate(mtc_drug_ids[:10], 1):
        print(f"   {i}. {did}")
    if len(mtc_drug_ids) > 10:
        print(f"   ... 共{len(mtc_drug_ids)}个药物")

    # 3. 构建MTC基准集（带SMILES和序列）
    mtc_affinity_df = affinity_df[
        (affinity_df["drug_id"].isin(mtc_drug_ids)) & 
        (affinity_df["protein_id"].isin(mtc_protein_ids))
    ].copy()

    # 添加药物SMILES（注意drug_dict的key是字符串，需要转换）
    def get_drug_smiles(did):
        return drug_dict.get(str(did), "")
    mtc_affinity_df["drug_smiles"] = mtc_affinity_df["drug_id"].apply(get_drug_smiles)
    
    # 添加靶点序列
    mtc_affinity_df["target_sequence"] = mtc_affinity_df["protein_id"].map(target_dict)
    
    # 添加靶点基因名标注
    def get_target_gene(tid):
        for gene in MTC_CORE_GENES:
            if gene.lower() in tid.lower():
                return gene
        return "Unknown"
    mtc_affinity_df["target_gene"] = mtc_affinity_df["protein_id"].apply(get_target_gene)

    # 重新排序列
    mtc_affinity_df = mtc_affinity_df[
        ["drug_id", "drug_smiles", "protein_id", "target_sequence", "target_gene", "affinity"]
    ].reset_index(drop=True)

    print(f"\n✅ MTC基准集筛选完成：")
    print(f"   - 总样本数：{len(mtc_affinity_df)} 条")
    print(f"   - 涉及药物数：{mtc_affinity_df['drug_id'].nunique()}")
    print(f"   - 涉及靶点数：{mtc_affinity_df['protein_id'].nunique()}")
    print(f"   - 亲和力范围：{mtc_affinity_df['affinity'].min():.4f} ~ {mtc_affinity_df['affinity'].max():.4f}")
    return mtc_affinity_df

def add_embeddings_to_mtc(mtc_df):
    """为MTC基准集补全嵌入特征（emb_drug_*/emb_target_*）"""
    # 1. 加载训练特征文件（已确认列名：drug_id, protein_id, emb_drug_*, emb_target_*）
    if not os.path.exists(TRAIN_FEATURE_PATH):
        raise FileNotFoundError(f"❌ 嵌入特征文件不存在：{TRAIN_FEATURE_PATH}")
    
    train_df = pd.read_csv(TRAIN_FEATURE_PATH)
    print(f"\n📥 加载嵌入特征文件：{len(train_df)} 条样本")
    print(f"   - 训练文件drug_id类型：{train_df['drug_id'].dtype}")
    print(f"   - 训练文件protein_id类型：{train_df['protein_id'].dtype}")
    
    # 关键修复3：统一训练特征文件的列类型（确保和mtc_df一致）
    train_df["drug_id"] = pd.to_numeric(train_df["drug_id"], errors="coerce").astype("int64")
    train_df["protein_id"] = train_df["protein_id"].astype("str")
    
    # 过滤训练文件中无效的行
    train_df = train_df[train_df["drug_id"].notna()]

    # 2. 提取嵌入列
    drug_emb_cols = [col for col in train_df.columns if col.startswith("emb_drug_")]
    target_emb_cols = [col for col in train_df.columns if col.startswith("emb_target_")]
    
    if not drug_emb_cols or not target_emb_cols:
        raise ValueError("❌ 嵌入特征文件缺少emb_drug_*或emb_target_*列")
    
    print(f"   - 药物嵌入列数：{len(drug_emb_cols)}（ChemBERTa）")
    print(f"   - 靶点嵌入列数：{len(target_emb_cols)}（ESM-2）")

    # 3. 合并嵌入特征（按drug_id+protein_id匹配）
    merge_cols = ["drug_id", "protein_id"] + drug_emb_cols + target_emb_cols
    train_emb_df = train_df[merge_cols].drop_duplicates(subset=["drug_id", "protein_id"])

    mtc_with_emb_df = pd.merge(
        mtc_df,
        train_emb_df,
        on=["drug_id", "protein_id"],
        how="left"
    )

    # 检查匹配失败的样本
    missing_mask = mtc_with_emb_df[drug_emb_cols[0]].isna() | mtc_with_emb_df[target_emb_cols[0]].isna()
    missing_count = missing_mask.sum()
    
    if missing_count > 0:
        print(f"\n⚠️  警告：{missing_count} 个样本未匹配到嵌入特征（可能是训练集中无对应药物/靶点）")
        # 过滤掉无嵌入的样本（保证后续预测可用）
        mtc_with_emb_df = mtc_with_emb_df[~missing_mask].reset_index(drop=True)
        print(f"   - 过滤后有效样本数：{len(mtc_with_emb_df)}")
    else:
        print(f"\n✅ 所有样本均匹配到嵌入特征！")

    # 4. 保存最终文件
    mtc_with_emb_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ 最终MTC基准集（带嵌入特征）已保存：")
    print(f"   - 保存路径：{FINAL_OUTPUT_FILE}")
    print(f"   - 最终样本数：{len(mtc_with_emb_df)}")
    print(f"   - 总列数：{len(mtc_with_emb_df.columns)}（基础信息+嵌入特征）")
    
    return mtc_with_emb_df

# ===================== 主函数（一键执行） =====================
def main():
    print("=" * 80)
    print("          一键构建MTC-DTA基准集（带嵌入特征）")
    print("=" * 80)
    
    try:
        # 步骤1：加载原始DAVIS数据（统一用protein_id，修复类型问题）
        drug_dict, target_dict, affinity_df = load_davis_raw()
        
        # 步骤2：筛选MTC核心靶点/药物，生成基准集
        mtc_df = filter_mtc_samples(drug_dict, target_dict, affinity_df)
        
        # 步骤3：补全嵌入特征，生成最终文件
        final_df = add_embeddings_to_mtc(mtc_df)
        
        print("\n🎉 全流程执行完成！")
        print(f"📌 最终文件：{FINAL_OUTPUT_FILE}")
        print(f"📌 列名预览：{final_df.columns.tolist()[:10]}...（包含emb_drug_*/emb_target_*）")
        
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈
        raise

if __name__ == "__main__":
    main()