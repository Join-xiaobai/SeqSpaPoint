import pandas as pd
import numpy as np
import open3d as o3d
import os
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import itertools  # 用于生成 drug-protein 组合
from tqdm import tqdm  # 如果你没装 tqdm，pip install tqdm
import random

import warnings
warnings.filterwarnings("ignore")  # 忽略警告

# 全局设置随机种子，保证结果可复现
def set_global_seed(seed: int = 0):
    random.seed(seed)          # Python 内置随机数
    np.random.seed(seed)       # Numpy 随机数
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python Hash 种子（可选，增强确定性）

set_global_seed(42)

# ===================== 路径配置（核心修改：改为可动态配置）=====================
# 1. 保留原有默认路径逻辑（兼容旧调用）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_INPUT_DIR = os.path.join(current_script_dir, "..", "data", "{mode}", "{data_name}")
DEFAULT_BASE_OUTPUT_DIR = os.path.join(current_script_dir, "{mode}", "{data_name}", "log_and_file")
# 转换为绝对路径，消除相对路径歧义
DEFAULT_BASE_INPUT_DIR = os.path.abspath(DEFAULT_BASE_INPUT_DIR)
DEFAULT_BASE_OUTPUT_DIR = os.path.abspath(DEFAULT_BASE_OUTPUT_DIR)

# 2. 新增路径获取函数（供外部调用时自定义路径）
def get_base_paths(mode: str, data_name: str, input_dir: str = None, output_dir: str = None) -> tuple[str, str]:
    """
    获取输入/输出基础路径（支持自定义，兼容默认）
    
    Args:
        mode: 任务模式（dti/dta/moa）
        data_name: 数据集名
        input_dir: 自定义输入目录（None则用默认）
        output_dir: 自定义输出目录（None则用默认）
    
    Returns:
        tuple: (input_dir_abs, output_dir_abs)
    """
    # 优先使用自定义路径，否则用默认路径
    if input_dir is None:
        input_dir_abs = DEFAULT_BASE_INPUT_DIR.format(mode=mode, data_name=data_name)
    else:
        input_dir_abs = os.path.abspath(input_dir)
    
    if output_dir is None:
        output_dir_abs = DEFAULT_BASE_OUTPUT_DIR.format(mode=mode, data_name=data_name)
    else:
        output_dir_abs = os.path.abspath(output_dir)
    
    # 确保目录存在
    os.makedirs(input_dir_abs, exist_ok=True)
    os.makedirs(output_dir_abs, exist_ok=True)
    
    return input_dir_abs, output_dir_abs

DTI_FILES = {
    "drug": "drug_smiles.txt",
    "protein": "protein_seq.txt",
    "interaction": "dti.txt"
}
DTA_FILES = {
    "drug": "ligands_can.txt",
    "protein": "proteins.txt",
    "interaction": "Y"
}
MOA_FILES = {
    "drug": "drug_smi.txt",
    "dti": "dti.txt",
    "gene": "tar_gene.txt",
    "seq": "tar_seq.txt"
}

DRUG_EMB_FILENAME = "drug_embeddings.pkl"
TARGET_EMB_FILENAME = "target_embeddings.pkl"

# ===================== 辅助工具函数（无修改）=====================
def is_valid_smiles(smi: str) -> bool:
    """校验 SMILES 序列有效性（保留常见有效字符）"""
    if not smi or len(smi.strip()) == 0:
        return False
    valid_chars = set("CNOHSPFClBrI[]()=+-#@*.0123456789 ")
    return all(c in valid_chars for c in smi.strip())

def is_valid_protein_seq(seq: str) -> bool:
    """校验蛋白序列有效性（保留标准氨基酸和 X）"""
    if not seq or len(seq.strip()) == 0:
        return False
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWYX")
    return all(c in valid_amino_acids for c in seq.strip().upper())

def normalize_column(col: pd.Series) -> np.ndarray:
    """带异常值处理的 min-max 归一化"""
    col = col.astype(float).values
    # 第一步：处理 nan/inf，填充为列均值
    col = np.nan_to_num(col, nan=np.nan, posinf=np.nan, neginf=np.nan)
    col_mean = np.nanmean(col)
    col = np.where(np.isfinite(col), col, col_mean)
    # 第二步：min-max 归一化
    min_val, max_val = col.min(), col.max()
    denom = np.maximum(max_val - min_val, 1e-8)  # 避免除 0
    return (col - min_val) / denom

# ===================== 批量嵌入生成函数（无修改）=====================
def generate_drug_embeddings_batch(drug_df, tokenizer, model, device, batch_size=32):
    """批量生成药物 ChemBERTa 嵌入，替代逐条处理，提速 10~20 倍"""
    drug_emb_dict = {}
    # 只保留有效 SMILES
    valid_drugs = drug_df[drug_df["smiles"].apply(is_valid_smiles)].reset_index(drop=True)
    drug_ids = valid_drugs["drug_id"].tolist()
    smiles_list = valid_drugs["smiles"].tolist()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Generating drug embeddings (batch)"):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_ids = drug_ids[i:i+batch_size]

            inputs = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for did, emb in zip(batch_ids, embs):
                drug_emb_dict[did] = emb.flatten()
    return drug_emb_dict


def generate_target_embeddings_batch(prot_df, tokenizer, model, device, batch_size=8):
    """批量生成靶点 ESM-2 嵌入，替代逐条处理， Hetionet 提速最明显"""
    target_emb_dict = {}
    # 只保留有效蛋白序列
    valid_prots = prot_df[prot_df["seq"].apply(is_valid_protein_seq)].reset_index(drop=True)
    target_ids = valid_prots["target_id"].tolist()
    seq_list = valid_prots["seq"].tolist()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(seq_list), batch_size), desc="Generating target embeddings (batch)"):
            batch_seqs = seq_list[i:i+batch_size]
            batch_ids = target_ids[i:i+batch_size]

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)

            outputs = model(** inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size())

            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embs = (sum_embeddings / sum_mask).cpu().numpy()

            for pid, emb in zip(batch_ids, embs):
                target_emb_dict[pid] = emb.flatten()
    return target_emb_dict

# ===================== 数据读取函数（无修改）=====================
def read_drug_df(input_dir: str) -> pd.DataFrame:
    """读取 DTA 模式药物文件并转换为 DataFrame（新增原始索引）"""
    file_path = os.path.join(input_dir, "ligands_can.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"药物文件不存在：{file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # JSON 格式补全
    if not content.startswith('{'):
        content = '{' + content
    if not content.endswith('}'):
        content = content + '}'
    
    try:
        drug_dict = json.loads(content)
    except json.JSONDecodeError as e:
        raise Exception(f"JSON 解析失败：{e}")
    
    # 核心修改1：保留原始索引（按文件中的顺序）
    df = pd.DataFrame(
        list(drug_dict.items()), 
        columns=["drug_id", "smiles"],
        index=range(len(drug_dict))  # 原始索引（0~N）
    )
    # 仅删除空值，保留原始索引（保证与亲和力矩阵对齐）
    df = df[["drug_id", "smiles"]].dropna(subset=["drug_id", "smiles"])
    # 序列有效性过滤
    df["is_valid"] = df["smiles"].apply(is_valid_smiles)
    # 核心修改2：记录有效药物的原始索引
    valid_df = df[df["is_valid"]].drop(columns=["is_valid"]).copy()
    valid_df["original_index"] = valid_df.index  # 新增列：原始文件中的索引
    print(f"✅ 解析 ligands_can.txt 为 JSON → 原始药物数: {len(df)}，有效药物数: {len(valid_df)}")
    return valid_df  # 返回带original_index的df

def read_protein_df(input_dir: str) -> pd.DataFrame:
    """读取 DTA 模式蛋白文件并转换为 DataFrame"""
    file_path = os.path.join(input_dir, "proteins.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"蛋白文件不存在：{file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # JSON 格式补全
    if not content.startswith('{'):
        content = '{' + content
    if not content.endswith('}'):
        content = content + '}'
    
    try:
        prot_dict = json.loads(content)
    except json.JSONDecodeError as e:
        raise Exception(f"JSON 解析失败：{e}")
    
    df = pd.DataFrame(list(prot_dict.items()), columns=["protein_id", "sequence"])
    # 仅删除空值，保留原始索引（保证与亲和力矩阵对齐）
    df = df[["protein_id", "sequence"]].dropna(subset=["protein_id", "sequence"])
    # 序列有效性过滤
    df["is_valid"] = df["sequence"].apply(is_valid_protein_seq)
    df = df[df["is_valid"]].drop(columns=["is_valid"])
    print(f"✅ 解析 proteins.txt 为 JSON → 有效蛋白数: {len(df)}")
    return df

def read_interaction_file(interaction_file_path: str, mode: str) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """读取交互文件（适配 DTI/DTA/MOA 模式）"""
    if not os.path.exists(interaction_file_path):
        raise FileNotFoundError(f"未找到交互文件：{interaction_file_path}")
    
    mode = mode.lower()
    if mode == "dta":
        try:
            with open(interaction_file_path, "rb") as f:
                affinity_matrix = pickle.load(f, encoding="latin1")
            affinity_matrix = np.array(affinity_matrix, dtype=np.float32)
            total_nan = np.isnan(affinity_matrix).sum()
            total_inf = np.isinf(affinity_matrix).sum()
            print(f"✅ {mode.upper()}模式：Y文件读取成功！矩阵形状：{affinity_matrix.shape}，NaN总数：{total_nan}，Inf总数：{total_inf}")
            return None, affinity_matrix
        except Exception as e:
            raise Exception(f"❌ DTA模式读取失败：{e}")
    
    elif mode == "dti":
        try:
            dti_data = pd.read_csv(
                interaction_file_path, 
                sep="\t", 
                header=None, 
                names=["drug_id", "interaction_type", "protein_id"], 
                encoding="latin1"
            )
            valid_dti = dti_data.dropna(subset=["drug_id", "protein_id"])
            valid_dti = valid_dti[
                (valid_dti["drug_id"].str.strip() != "") & 
                (valid_dti["protein_id"].str.strip() != "")
            ]
            print(f"✅ {mode.upper()}模式：dti.txt读取成功！有效关联数：{len(valid_dti)}")
            return valid_dti, None
        except Exception as e:
            raise Exception(f"❌ DTI模式读取失败：{e}")
    
    elif mode == "moa":
        try:
            moa_interaction = pd.read_csv(
                interaction_file_path, 
                sep="\t", 
                header=0, 
                encoding="latin1"
            )
            moa_interaction.columns = ["DrugID", "TargetID"]
            moa_interaction = moa_interaction.dropna().reset_index(drop=True)
            moa_interaction = moa_interaction[
                (moa_interaction["DrugID"].str.strip() != "") & 
                (moa_interaction["TargetID"].str.strip() != "")
            ].reset_index(drop=True)
            print(f"✅ MOA模式：交互文件读取成功！有效关联数：{len(moa_interaction)}")
            return moa_interaction, None
        except Exception as e:
            raise Exception(f"❌ MOA模式交互文件读取失败：{e}")
    
    else:
        raise ValueError("模式仅支持'dti'、'dta'、'moa'！")

def read_raw_files(mode: str, input_dir: str, dataset_name: str) -> tuple:
    """读取原始数据（按模式分类处理，修复 ID 对齐、异常值）"""
    mode = mode.lower()
    input_dir = os.path.abspath(input_dir)
    
    if mode == "dti":
        # 读取 DTI 模式数据
        drug_path = os.path.join(input_dir, DTI_FILES["drug"])
        prot_path = os.path.join(input_dir, DTI_FILES["protein"])
        interact_path = os.path.join(input_dir, DTI_FILES["interaction"])
        
        drugs = pd.read_csv(drug_path, sep="\t", header=None, names=["drug_id", "smiles"])
        proteins = pd.read_csv(prot_path, sep="\t", header=None, names=["protein_id", "sequence"])
        interaction_data, _ = read_interaction_file(interact_path, mode)
        
        # 过滤有效数据（保证 ID 存在）
        valid_interaction = interaction_data[
            (interaction_data["drug_id"].isin(drugs["drug_id"])) &
            (interaction_data["protein_id"].isin(proteins["protein_id"]))
        ].copy().reset_index(drop=True)
        
        # 序列有效性过滤
        drugs["is_valid"] = drugs["smiles"].apply(is_valid_smiles)
        proteins["is_valid"] = proteins["sequence"].apply(is_valid_protein_seq)
        drugs = drugs[drugs["is_valid"]].drop(columns=["is_valid"])
        proteins = proteins[proteins["is_valid"]].drop(columns=["is_valid"])
        
        print(f"📌 DTI模式：过滤后有效关联数：{len(valid_interaction)}")
        return drugs, proteins, valid_interaction
    
    elif mode == "dta":
        # 读取 DTA 模式数据（修复 ID 对齐、DAVIS 数据处理）
        drugs = read_drug_df(input_dir)
        proteins = read_protein_df(input_dir)
        interact_path = os.path.join(input_dir, DTA_FILES["interaction"])
        _, affinity_matrix = read_interaction_file(interact_path, mode)
        
        # 核心修改：仅对MTC数据集做索引对齐，其他数据集跳过（保证兼容性）
        if dataset_name.lower() == "mtc":
            # 用有效药物的原始索引对齐Y矩阵
            valid_original_indices = drugs["original_index"].tolist()
            # 截取Y矩阵中有效药物对应的行
            affinity_matrix = affinity_matrix[valid_original_indices, :]
            print(f"🔍 MTC数据集专属：按有效药物索引裁剪Y矩阵 → 新形状：{affinity_matrix.shape}")
        
        # 严格校验矩阵形状与药物/蛋白数量匹配
        if affinity_matrix.shape != (len(drugs), len(proteins)):
            raise ValueError(
                f"DTA矩阵形状{affinity_matrix.shape}与药物/蛋白数量({len(drugs)}/{len(proteins)})不匹配！\n"
                f"请检查 ligands_can.txt、proteins.txt 与 Y 矩阵的顺序一致性"
            )
        
        # 处理亲和力矩阵（适配 DAVIS 数据集，规避 0/极小值导致的 log 异常）
        print(f"🔍 原始 Y 矩阵范围: [{np.nanmin(affinity_matrix):.1f}, {np.nanmax(affinity_matrix):.1f}]")
        affinity_np = np.array(affinity_matrix, dtype=np.float32)
        
        if dataset_name.lower() == "davis":
            min_threshold = 1e-12  # 极小值阈值，避免 log 异常
            affinity_np = np.where(
                (affinity_np <= min_threshold) | (affinity_np > 1e9),  # 过滤异常值
                np.nan, 
                -np.log10(np.clip(affinity_np / 1e9, min_threshold, 1.0))
            )
            # 额外过滤对数运算后的 inf
            affinity_np = np.nan_to_num(affinity_np, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
        # 提取有效数据（过滤 NaN/Inf）
        valid_mask = np.isfinite(affinity_np)
        valid_drug_indices, valid_protein_indices = np.where(valid_mask)
        
        interaction_list = []
        for i, j in zip(valid_drug_indices, valid_protein_indices):
            affinity = affinity_np[i, j]
            interaction_list.append({
                "drug_id": drugs.iloc[i]["drug_id"],
                "protein_id": proteins.iloc[j]["protein_id"],
                "affinity": affinity
            })
        
        valid_interaction = pd.DataFrame(interaction_list)
        print(f"📌 DTA模式：成功提取{len(valid_interaction)}条无NaN有效药物-蛋白亲和力数据")
        
        if dataset_name.lower() == "davis":
            aff_min = valid_interaction["affinity"].min()
            aff_max = valid_interaction["affinity"].max()
            print(f"✅ DAVIS 亲和力范围: [{aff_min:.4f}, {aff_max:.4f}]")
        
        return drugs, proteins, valid_interaction
    
    else:  # moa
        # 读取 MOA 模式数据（仅保留基础读取，负采样交给extract_features）
        drug_path = os.path.join(input_dir, MOA_FILES["drug"])
        dti_path = os.path.join(input_dir, MOA_FILES["dti"])
        gene_path = os.path.join(input_dir, MOA_FILES["gene"])
        seq_path = os.path.join(input_dir, MOA_FILES["seq"])
        
        drugs = pd.read_csv(drug_path, sep="\t", header=0, encoding="latin1")
        drugs.columns = ["DrugID", "smi"]
        dti_df = pd.read_csv(dti_path, sep="\t", header=0, encoding="latin1")
        dti_df.columns = ["DrugID", "TargetID"]
        genes = pd.read_csv(gene_path, sep="\t", header=0, encoding="latin1")
        genes.columns = ["TargetID", "Gene_symbol"]
        seq_df = pd.read_csv(seq_path, sep="\t", header=0, encoding="latin1")
        seq_df.columns = ["TargetID", "seq"]
        
        # 基础数据清洗（仅保留有效数据）
        genes = genes.dropna().reset_index(drop=True)
        genes = genes[genes["Gene_symbol"].str.strip() != ""].reset_index(drop=True)
        genes["gene_id"] = genes["Gene_symbol"].astype(str)
        
        seq_df = seq_df.dropna().reset_index(drop=True)
        seq_df = seq_df[seq_df["seq"].str.strip() != ""].reset_index(drop=True)
        seq_df["is_valid"] = seq_df["seq"].apply(is_valid_protein_seq)
        seq_df = seq_df[seq_df["is_valid"]].drop(columns=["is_valid"])
        
        drugs["is_valid"] = drugs["smi"].apply(is_valid_smiles)
        drugs = drugs[drugs["is_valid"]].drop(columns=["is_valid"])
        
        # 仅合并有效正样本（负采样在extract_features中处理）
        valid_drug_target = dti_df[dti_df["DrugID"].isin(drugs["DrugID"])].reset_index(drop=True)
        valid_drug_target_gene = pd.merge(
            valid_drug_target, 
            genes[["TargetID", "gene_id"]], 
            on="TargetID", 
            how="inner"
        )
        valid_interaction = pd.merge(
            valid_drug_target_gene, 
            seq_df[["TargetID", "seq"]], 
            on="TargetID", 
            how="inner"
        ).reset_index(drop=True)
        
        co_occur_df = valid_interaction.groupby(['gene_id', 'TargetID']).size().reset_index(name='gene_target_co_occur')
        valid_interaction = pd.merge(valid_interaction, co_occur_df, on=['gene_id', 'TargetID'], how='left')
        
        print(f"📌 MOA模式：过滤后有效正样本数：{len(valid_interaction)}")
        return drugs, dti_df, genes, seq_df, valid_interaction

# ===================== 嵌入生成与加载（无修改）=====================
def generate_embeddings(mode: str, input_dir: str, drug_emb_path: str, target_emb_path: str):
    """生成药物/靶点嵌入（优化 ESM2 提取方式，增加序列校验）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 使用设备: {device}")
    
    # 模型本地路径（基于脚本目录构建）
    CHEMBERTA_LOCAL_PATH = os.path.join(current_script_dir, "models", "chemberta")
    ESM2_LOCAL_PATH = os.path.join(current_script_dir, "models", "esm2_t33_650M_UR50D")
    
    # 模型存在性校验
    if not os.path.exists(CHEMBERTA_LOCAL_PATH):
        raise FileNotFoundError(
            f"❌ 未找到 ChemBERTa 本地模型目录：{CHEMBERTA_LOCAL_PATH}\n"
            "💡 请确保已将 ChemBERTa 模型（如 seyonec/ChemBERTa-zinc-base-v1）\n"
            "   下载至 ./data_preprocessing/models/chemberta/"
        )
    if not os.path.exists(ESM2_LOCAL_PATH):
        raise FileNotFoundError(
            f"❌ 未找到 ESM-2 本地模型目录：{ESM2_LOCAL_PATH}\n"
            "💡 请确保已将 ESM-2 模型（facebook/esm2_t33_650M_UR50D）\n"
            "   下载至 ./data_preprocessing/models/esm2_t33_650M_UR50D/"
        )
    
    # === 药物嵌入生成（ChemBERTa）===
    print("🧪 正在生成药物嵌入 (ChemBERTa-77M-MTR，本地加载)...")
    if mode.lower() == "dta":
        drug_df = read_drug_df(input_dir)
    else:
        drug_file = os.path.join(input_dir, MOA_FILES["drug"] if mode.lower() == "moa" else DTI_FILES["drug"])
        drug_df = pd.read_csv(
            drug_file, 
            sep="\t", 
            header=None if mode.lower() == "dti" else 0, 
            names=["drug_id", "smiles"]
        )
        if mode.lower() == "moa":
            drug_df.columns = ["DrugID", "smiles"]
            drug_df.rename(columns={"DrugID": "drug_id"}, inplace=True)
    
    # 加载 tokenizer 和模型
    tokenizer_drug = AutoTokenizer.from_pretrained(CHEMBERTA_LOCAL_PATH)
    model_drug = AutoModel.from_pretrained(CHEMBERTA_LOCAL_PATH).to(device).eval()
    
    # ===== 替换为批量生成药物嵌入（优化 Hetionet 速度）=====
    drug_emb_dict = generate_drug_embeddings_batch(
        drug_df, tokenizer_drug, model_drug, device, batch_size=32
    )
    
    # 保存药物嵌入
    with open(drug_emb_path, "wb") as f:
        pickle.dump(drug_emb_dict, f)
    print(f"✅ 药物嵌入已保存至: {drug_emb_path} | 共 {len(drug_emb_dict)} 个")
    
    # === 靶点嵌入生成（ESM2，优化为平均池化）===
    print("🧬 正在生成靶点嵌入 (ESM-2 650M，本地加载)...")
    if mode.lower() == "dta":
        prot_df = read_protein_df(input_dir)
        prot_df.rename(columns={"protein_id": "target_id", "sequence": "seq"}, inplace=True)
    elif mode.lower() == "dti":
        prot_file = os.path.join(input_dir, DTI_FILES["protein"])
        prot_df = pd.read_csv(prot_file, sep="\t", header=None, names=["target_id", "seq"])
    else:  # moa
        seq_file = os.path.join(input_dir, MOA_FILES["seq"])
        prot_df = pd.read_csv(seq_file, sep="\t", header=0, encoding="latin1")
        prot_df.columns = ["target_id", "seq"]
    
    # 加载 tokenizer 和模型（替换为 AutoModel，最优特征提取）
    tokenizer_prot = AutoTokenizer.from_pretrained(ESM2_LOCAL_PATH)
    model_prot = AutoModel.from_pretrained(
        ESM2_LOCAL_PATH,
        output_hidden_states=True
    ).to(device).eval()
    
    # ===== 替换为批量生成靶点嵌入（Hetionet 核心提速）=====
    target_emb_dict = generate_target_embeddings_batch(
        prot_df, tokenizer_prot, model_prot, device, batch_size=8
    )
    
    # 保存靶点嵌入
    with open(target_emb_path, "wb") as f:
        pickle.dump(target_emb_dict, f)
    print(f"✅ 靶点嵌入已保存至: {target_emb_path} | 共 {len(target_emb_dict)} 个")

def ensure_embeddings_exist(drug_emb_path: str, target_emb_path: str, mode: str, input_dir: str):
    """检查嵌入文件是否存在，缺失则自动生成"""
    if not os.path.exists(drug_emb_path) or not os.path.exists(target_emb_path):
        print("⚠️ 嵌入文件缺失，正在自动生成...")
        generate_embeddings(mode, input_dir, drug_emb_path, target_emb_path)
    else:
        print("✅ 嵌入文件已存在，跳过生成")

def load_drug_embeddings(embed_file_path: str) -> dict:
    """加载药物嵌入字典（增加空字典校验）"""
    if not os.path.exists(embed_file_path):
        raise FileNotFoundError(f"药物嵌入文件不存在：{embed_file_path}")
    
    with open(embed_file_path, "rb") as f:
        drug_emb_dict = pickle.load(f)
    
    if not isinstance(drug_emb_dict, dict) or len(drug_emb_dict) == 0:
        raise ValueError("❌ 药物嵌入字典为空或格式错误！")
    
    print(f"✅ 药物嵌入加载成功！数量：{len(drug_emb_dict)}")
    return drug_emb_dict

def load_target_embeddings(embed_file_path: str) -> dict:
    """加载靶点嵌入字典（增加空字典校验）"""
    if not os.path.exists(embed_file_path):
        raise FileNotFoundError(f"靶点嵌入文件不存在：{embed_file_path}")
    
    with open(embed_file_path, "rb") as f:
        target_emb_dict = pickle.load(f)
    
    if not isinstance(target_emb_dict, dict) or len(target_emb_dict) == 0:
        raise ValueError("❌ 靶点嵌入字典为空或格式错误！")
    
    print(f"✅ 靶点嵌入加载成功！数量：{len(target_emb_dict)}")
    return target_emb_dict

# ===================== 特征提取（无修改）=====================
def extract_features(*args, mode: str, drug_emb_dict=None, target_emb_dict=None):
    """提取组合特征（嵌入+统计特征，优化缺失值填充）"""
    mode = mode.lower()
    if drug_emb_dict is None or target_emb_dict is None:
        raise ValueError("❌ 药物/靶点嵌入字典不能为空！")
    
    # === 数据初始化 ===
    if mode == "moa":
        # 原有 MOA 逻辑（不动）
        drugs, dti_df, genes, seq_df, interaction_data = args
        id_col_drug = "DrugID"
        id_col_target = "TargetID"
        data = interaction_data.copy()
        drug_len_map = drugs.set_index(id_col_drug)["smi"].apply(lambda x: len(str(x).strip()))
        target_len_map = seq_df.set_index(id_col_target)["seq"].apply(lambda x: len(str(x).strip()))
        data["drug_complexity"] = data[id_col_drug].map(drug_len_map)
        data["target_length"] = data[id_col_target].map(target_len_map)
        length_col = "target_length"
    else:
        # DTI/DTA 模式（重点修改这里）
        drugs, proteins, interaction_data = args
        id_col_drug = "drug_id"
        id_col_target = "protein_id"
        data = pd.merge(interaction_data, drugs[[id_col_drug]], on=id_col_drug, how="inner")
        data = pd.merge(data, proteins[[id_col_target]], on=id_col_target, how="inner")
        drug_len_map = drugs.set_index(id_col_drug)["smiles"].apply(lambda x: len(str(x).strip()))
        prot_len_map = proteins.set_index(id_col_target)["sequence"].apply(lambda x: len(str(x).strip()))
        data["drug_complexity"] = data[id_col_drug].map(drug_len_map)
        data["protein_length"] = data[id_col_target].map(prot_len_map)
        length_col = "protein_length"
    
    # === 统计特征生成 ===
    data["length_ratio"] = data["drug_complexity"] / np.maximum(data[length_col], 1e-8)
    data = data.fillna(0.0)  # 填充统计特征空值
    
    # === 嵌入维度获取与均值计算（用于缺失值填充）===
    def get_emb_mean(emb_dict: dict) -> np.ndarray:
        """计算嵌入字典的均值（过滤无效嵌入）"""
        emb_vals = []
        for emb in emb_dict.values():
            if emb is not None and len(emb) > 0 and np.isfinite(emb).all():
                emb_vals.append(emb)
        if len(emb_vals) == 0:
            raise ValueError("❌ 嵌入字典中无有效嵌入数据！")
        return np.array(emb_vals).mean(axis=0)
    
    drug_dim = len(next(iter(drug_emb_dict.values())))
    target_dim = len(next(iter(target_emb_dict.values())))
    drug_emb_mean = get_emb_mean(drug_emb_dict)
    target_emb_mean = get_emb_mean(target_emb_dict)
    
    # === 嵌入提取与缺失值填充 ===
    def get_emb_or_mean(idx: str, emb_dict: dict, dim: int, emb_mean: np.ndarray) -> np.ndarray:
        """获取嵌入，缺失则返回均值"""
        emb = emb_dict.get(idx)
        if emb is not None and len(emb) == dim and np.isfinite(emb).all():
            return emb
        return emb_mean
    
    # 提取药物/靶点嵌入
    drug_embs = np.stack(data[id_col_drug].apply(
        lambda x: get_emb_or_mean(x, drug_emb_dict, drug_dim, drug_emb_mean)
    ).values)
    target_embs = np.stack(data[id_col_target].apply(
        lambda x: get_emb_or_mean(x, target_emb_dict, target_dim, target_emb_mean)
    ).values)
    combined_embs = np.concatenate([drug_embs, target_embs], axis=1)
    
    # === 特征合并 ===
    drug_emb_cols = [f"emb_drug_{i}" for i in range(drug_dim)]
    target_emb_cols = [f"emb_target_{i}" for i in range(target_dim)]
    emb_cols = drug_emb_cols + target_emb_cols
    
    emb_df = pd.DataFrame(combined_embs, columns=emb_cols, index=data.index)
    data = pd.concat([data, emb_df], axis=1)

    # ========== 优化：DTI 负采样（避免生成全量组合，Hetionet 专用）==========
    if mode == "dti":
        data["label"] = 1
        positive_data = data.copy()
        print(f"✅ DTI 正样本数：{len(positive_data)}")

        all_drugs = positive_data["drug_id"].unique()
        all_proteins = positive_data["protein_id"].unique()
        positive_pairs = set(zip(positive_data["drug_id"], positive_data["protein_id"]))

        np.random.seed(42)
        needed = len(positive_data)
        negative_pairs = []

        # 批量随机采样，不生成全量笛卡尔积
        while len(negative_pairs) < needed:
            batch_size = min(1000, needed - len(negative_pairs))
            sampled_drugs = np.random.choice(all_drugs, size=batch_size, replace=True)
            sampled_prots = np.random.choice(all_proteins, size=batch_size, replace=True)

            candidates = [(d, p) for d, p in zip(sampled_drugs, sampled_prots)
                          if (d, p) not in positive_pairs]
            negative_pairs.extend(candidates[:needed - len(negative_pairs)])

        # 构建负样本 DF
        negative_df = pd.DataFrame(negative_pairs[:needed], columns=["drug_id", "protein_id"])

        # 补全统计特征
        negative_df["drug_complexity"] = negative_df["drug_id"].map(drug_len_map).fillna(drug_len_map.mean())
        negative_df["protein_length"] = negative_df["protein_id"].map(prot_len_map).fillna(prot_len_map.mean())
        negative_df["length_ratio"] = negative_df["drug_complexity"] / np.maximum(negative_df["protein_length"], 1e-8)
        negative_df = negative_df.fillna(0.0)

        # 补全嵌入
        negative_drug_embs = np.stack(negative_df["drug_id"].apply(
            lambda x: get_emb_or_mean(x, drug_emb_dict, drug_dim, drug_emb_mean)
        ).values)
        negative_target_embs = np.stack(negative_df["protein_id"].apply(
            lambda x: get_emb_or_mean(x, target_emb_dict, target_dim, target_emb_mean)
        ).values)
        negative_combined_embs = np.concatenate([negative_drug_embs, negative_target_embs], axis=1)

        negative_emb_df = pd.DataFrame(negative_combined_embs, columns=emb_cols, index=negative_df.index)
        negative_df = pd.concat([negative_df, negative_emb_df], axis=1)
        negative_df["label"] = 0

        data = pd.concat([positive_data, negative_df], ignore_index=True)
        print(f"✅ DTI 负样本数：{len(negative_df)} | 总样本数：{len(data)}")
    # ==========================================================================

    # ========== 复用DTI逻辑：MOA 负采样（和DTI仅列名不同）==========
    elif mode == "moa":
        # 新增：显式重新生成len_map，避免作用域问题
        drug_len_map = drugs.set_index("DrugID")["smi"].apply(lambda x: len(str(x).strip()))
        target_len_map = seq_df.set_index("TargetID")["seq"].apply(lambda x: len(str(x).strip()))

        data["label"] = 1  # 正样本标签（dti.csv中的对）
        positive_data = data.copy()
        print(f"✅ MOA 正样本数：{len(positive_data)}")
    
        # 仅改列名：drug_id → DrugID，protein_id → TargetID
        all_drugs = positive_data["DrugID"].unique()
        all_targets = positive_data["TargetID"].unique()
        positive_pairs = set(zip(positive_data["DrugID"], positive_data["TargetID"]))
    
        np.random.seed(42)
        needed = len(positive_data) * 10  # MOA负样本是正样本10倍（和data_split_moa.py对齐）
        negative_pairs = []
    
        # 批量随机采样（复用DTI的批量逻辑，仅改变量名）
        while len(negative_pairs) < needed:
            batch_size = min(1000, needed - len(negative_pairs))
            sampled_drugs = np.random.choice(all_drugs, size=batch_size, replace=True)
            sampled_targets = np.random.choice(all_targets, size=batch_size, replace=True)
    
            candidates = [(d, t) for d, t in zip(sampled_drugs, sampled_targets)
                          if (d, t) not in positive_pairs]
            negative_pairs.extend(candidates[:needed - len(negative_pairs)])
    
        # 构建负样本 DF（仅改列名）
        negative_df = pd.DataFrame(negative_pairs[:needed], columns=["DrugID", "TargetID"])

        # 新增：补充gene_id字段（从genes表映射）
        gene_map = genes.set_index("TargetID")["gene_id"].to_dict()
        negative_df["gene_id"] = negative_df["TargetID"].map(gene_map).fillna("unknown")
    
        # 补全统计特征（复用DTI逻辑，仅改列名）
        negative_df["drug_complexity"] = negative_df["DrugID"].map(drug_len_map).fillna(drug_len_map.mean())
        negative_df["target_length"] = negative_df["TargetID"].map(target_len_map).fillna(target_len_map.mean())
        negative_df["length_ratio"] = negative_df["drug_complexity"] / np.maximum(negative_df["target_length"], 1e-8)
        negative_df = negative_df.fillna(0.0)
    
        # 补全嵌入（完全复用DTI逻辑，仅改列名）
        negative_drug_embs = np.stack(negative_df["DrugID"].apply(
            lambda x: get_emb_or_mean(x, drug_emb_dict, drug_dim, drug_emb_mean)
        ).values)
        negative_target_embs = np.stack(negative_df["TargetID"].apply(
            lambda x: get_emb_or_mean(x, target_emb_dict, target_dim, target_emb_mean)
        ).values)
        negative_combined_embs = np.concatenate([negative_drug_embs, negative_target_embs], axis=1)
    
        negative_emb_df = pd.DataFrame(negative_combined_embs, columns=emb_cols, index=negative_df.index)
        negative_df = pd.concat([negative_df, negative_emb_df], axis=1)
        negative_df["label"] = 0  # 负样本标签
    
        # 合并正负样本（复用DTI逻辑）
        data = pd.concat([positive_data, negative_df], ignore_index=True)
        print(f"✅ MOA 负样本数：{len(negative_df)} | 总样本数：{len(data)}")
    # ==========================================================================
    
    # === 数据清洗 ===
    data = data.dropna(subset=[id_col_drug, id_col_target])
    print(f"✅ 特征提取完成：附加 {drug_dim}+{target_dim}={combined_embs.shape[1]} 维语义嵌入")
    return data

# ===================== 点云生成（无修改）=====================
def generate_point_cloud(data, mode: str, output_dir: str):
    """生成点云文件与特征文件（优化归一化、数据清洗）"""
    mode = mode.lower()
    os.makedirs(output_dir, exist_ok=True, mode=0o755)
    
    # === 配置参数 ===
    if mode == "dta":
        point_cols = ["drug_complexity", "protein_length", "length_ratio"]
        id_cols = ["drug_id", "protein_id"]
        label_col = "affinity"
    elif mode == "dti":
        point_cols = ["drug_complexity", "protein_length", "length_ratio"]
        id_cols = ["drug_id", "protein_id"]
        label_col = "label"  # DTI 分类标签
    else:  # moa
        point_cols = ["drug_complexity", "target_length", "length_ratio"]
        id_cols = ["DrugID", "TargetID"]
        if "gene_id" in data.columns:
            id_cols.append("gene_id")
        label_col = "label"  # 复用DTI的label列逻辑
    
    # === 归一化生成点云数据 ===
    norm_cols = []
    for col in point_cols:
        if col in data.columns:
            norm_col_name = f"{col}_norm"
            data[norm_col_name] = normalize_column(data[col])
            norm_cols.append(norm_col_name)
        else:
            raise ValueError(f"点云特征列缺失：{col}")
    
    # 生成点云
    points = np.column_stack([data[col].values for col in norm_cols]).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # === 保存文件 ===
    suffix = {"dti": "dti", "dta": "dta", "moa": "moa"}[mode]
    pcd_path = os.path.join(output_dir, f"{suffix}_point_cloud.ply")
    feature_path = os.path.join(output_dir, f"{suffix}_features.csv")
    
    # 保存点云
    o3d.io.write_point_cloud(pcd_path, pcd)
    
    # 保存特征文件
    emb_cols = [col for col in data.columns if col.startswith("emb_")]
    save_cols = id_cols + emb_cols
    
    if label_col and label_col in data.columns:
        save_cols.append(label_col)
    
    # 数据清洗（过滤空 ID、无效数据）
    data_clean = data[save_cols].copy().dropna().reset_index(drop=True)
    if mode in ("dta", "dti"):
        data_clean = data_clean[
            (data_clean["drug_id"].str.strip() != "") &
            (data_clean["protein_id"].str.strip() != "")
        ].reset_index(drop=True)
    elif mode == "moa":
        data_clean = data_clean[
            (data_clean["DrugID"].str.strip() != "") &
            (data_clean["TargetID"].str.strip() != "")
        ].reset_index(drop=True)
    
    data_clean.to_csv(feature_path, index=False, encoding="utf-8")
    
    # === 打印结果 ===
    has_label = label_col in data_clean.columns if label_col else False
    print(f"\n=== 点云生成完成 ===")
    print(f"点云数量：{len(points)} | 点云文件：{pcd_path}")
    print(f"特征文件（含嵌入{'和标签' if has_label else ''}）：{feature_path}")
    return len(points)

# ===================== 主流程函数（核心修改：支持自定义路径）=====================
def run_preprocessing(mode: str, dataset_name: str, input_dir: str = None, output_dir: str = None):
    """
    可编程调用的预处理统一入口。
    无全局状态，所有路径和配置通过参数传递。
    
    Args:
        mode: 任务模式（dti/dta/moa）
        dataset_name: 数据集名
        input_dir: 自定义输入目录（None则用默认路径）
        output_dir: 自定义输出目录（None则用默认路径）
    """
    # 参数校验
    if mode.lower() not in ("dti", "dta", "moa"):
        raise ValueError("❌ 模式仅支持 'dti'、'dta'、'moa'！")
    
    # 核心修改：使用新的路径函数获取输入/输出目录
    input_dir_abs, output_dir_abs = get_base_paths(mode, dataset_name, input_dir, output_dir)
    
    drug_emb_path = os.path.join(output_dir_abs, DRUG_EMB_FILENAME)
    target_emb_path = os.path.join(output_dir_abs, TARGET_EMB_FILENAME)
    
    # 执行预处理流程
    print(f"=== 开始处理：{mode.upper()}模式 | {dataset_name}数据集 ===")
    print(f"输入目录：{input_dir_abs}")
    print(f"输出目录：{output_dir_abs}")
    
    try:
        ensure_embeddings_exist(drug_emb_path, target_emb_path, mode, input_dir_abs)
        raw_data = read_raw_files(mode, input_dir_abs, dataset_name)
        drug_emb_dict = load_drug_embeddings(drug_emb_path)
        target_emb_dict = load_target_embeddings(target_emb_path)
        feature_data = extract_features(*raw_data, mode=mode, drug_emb_dict=drug_emb_dict, target_emb_dict=target_emb_dict)
        point_count = generate_point_cloud(feature_data, mode=mode, output_dir=output_dir_abs)
        print(f"\n🎉 处理完成！共生成 {point_count} 个点的 {mode.upper()} 点云")
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ===================== 独立运行入口（兼容旧逻辑）=====================
if __name__ == "__main__":
    # 默认值用于直接运行脚本
    DEFAULT_MODE = "dta"
    DEFAULT_DATASET = "davis"
    print(f"🚀 直接运行预处理模块：{DEFAULT_MODE.upper()} / {DEFAULT_DATASET}")
    run_preprocessing(DEFAULT_MODE, DEFAULT_DATASET)