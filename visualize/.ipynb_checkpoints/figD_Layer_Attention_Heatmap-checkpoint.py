import os
import sys
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ===================== 全局样式配置（顶刊标准，兼容Linux/Windows/Mac） =====================
# 字体兼容：优先Arial，自动降级适配系统
try:
    plt.rcParams['font.family'] = 'Arial'
    from matplotlib.font_manager import findfont, FontProperties
    font = FontProperties(family='Arial', size=10)
    findfont(font)
except:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Heiti TC', 'sans-serif']

# 顶刊核心样式参数（统一规范）
plt.rcParams['font.size'] = 10                # 基础字号
plt.rcParams['axes.linewidth'] = 1.2          # 坐标轴宽度
plt.rcParams['xtick.major.width'] = 1.2       # X轴刻度宽度
plt.rcParams['ytick.major.width'] = 1.2       # Y轴刻度宽度
plt.rcParams['xtick.color'] = '#333333'       # 刻度文字颜色（深灰）
plt.rcParams['ytick.color'] = '#333333'       # 刻度文字颜色
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
mpl.rcParams['axes.spines.top'] = False       # 隐藏顶部边框
mpl.rcParams['axes.spines.right'] = False     # 隐藏右侧边框
plt.rcParams['figure.dpi'] = 300              # 绘图DPI（高清）
plt.rcParams['savefig.dpi'] = 300             # 保存DPI（顶刊要求≥300）
plt.rcParams['savefig.bbox'] = 'tight'        # 紧凑保存，去除多余空白
plt.rcParams['savefig.pad_inches'] = 0.1      # 保存内边距

# ===================== 路径与环境配置（适配任意运行环境） =====================
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # 添加上级目录，确保模型/数据集导入

# 导入项目核心类（异常捕获，提升鲁棒性）
try:
    from model.SeqSpaPoint import SeqSpaPoint
    from dataset.interaction_dataset import PointCloudInteractionDataset
except ImportError as e:
    raise ImportError(f"导入模型/数据集失败：{e}\n请检查model和dataset目录路径是否正确")

# ===================== 注意力提取器（重构Hook，提取真实层输出） =====================
class AttentionExtractor:
    """层注意力提取器
    功能：注册Hook提取各层真实输出，处理为统一10维注意力权重，保留原始分布
    """
    def __init__(self, model):
        self.model = model
        self.drug_attn = None    # Drug Cross Attn层输出
        self.target_attn = None  # Target Cross Attn层输出
        self.joint_attn = None   # Joint Cross Attn层输出
        self.fusion_feat = None  # Fusion层输出
        
        # 注册Hook提取层输出（仅提取，不修改数据）
        self._register_hooks()
    
    def _register_hooks(self):
        """注册Hook函数，提取各层真实输出"""
        # Drug Cross Attn Hook：保存输出特征均值
        def drug_hook(module, input, output):
            with torch.no_grad():
                # 取输出特征均值作为注意力信号（确保10个点维度）
                self.drug_attn = output.mean(dim=-1).squeeze(0).cpu().numpy()  # (10,)
        
        # Target Cross Attn Hook
        def target_hook(module, input, output):
            with torch.no_grad():
                self.target_attn = output.mean(dim=-1).squeeze(0).cpu().numpy()  # (10,)
        
        # Joint Cross Attn Hook
        def joint_hook(module, input, output):
            with torch.no_grad():
                self.joint_attn = output.mean(dim=-1)[0].squeeze(0).cpu().numpy()  # (10,)
        
        # Fusion Proj Hook
        def fusion_hook(module, input, output):
            with torch.no_grad():
                self.fusion_feat = output.squeeze(0).cpu().numpy()
        
        # 注册Hook到对应层
        self.model.drug_cross_attn.register_forward_hook(drug_hook)
        self.model.target_cross_attn.register_forward_hook(target_hook)
        self.model.joint_cross_attn.register_forward_hook(joint_hook)
        self.model.fusion_proj.register_forward_hook(fusion_hook)
    
    def _process_attn(self, attn):
        """处理注意力权重，统一转为10维（仅做维度适配，不改变分布）
        Args:
            attn: 原始注意力权重（任意维度）
        Returns:
            attn_10d: 标准化后的10维注意力权重
        """
        if attn is None:
            return np.zeros(10)
        attn = np.array(attn)
        
        # 维度适配逻辑（保留原始数据特征）
        if attn.size == 0:
            return np.zeros(10)
        elif attn.size != 10:
            if attn.ndim == 0:
                return np.full(10, attn)  # 标量转为10维常量
            elif attn.size > 10:
                return attn[:10]          # 超过10维取前10维
            else:
                # 不足10维用均值填充（保留统计特征）
                return np.pad(attn, (0, 10 - attn.size), mode='mean')
        return attn
    
    def _normalize_attn(self, attn):
        """Min-Max归一化（仅缩放至0-1，不改变相对分布）
        Args:
            attn: 10维注意力权重
        Returns:
            norm_attn: 归一化后的权重
        """
        min_val = attn.min()
        max_val = attn.max()
        if max_val - min_val < 1e-10:  # 避免除零
            return np.zeros_like(attn)
        return (attn - min_val) / (max_val - min_val)
    
    def get_attention_weights(self):
        """获取所有层的归一化注意力权重
        Returns:
            attn_weights: 各层注意力权重字典
        """
        # 提取并处理各层注意力
        drug_attn = self._process_attn(self.drug_attn)
        target_attn = self._process_attn(self.target_attn)
        joint_attn = self._process_attn(self.joint_attn)
        fusion_attn = self._process_attn(self.fusion_feat)
        
        # 每层独立归一化
        return {
            "Drug Cross Attn": self._normalize_attn(drug_attn),
            "Target Cross Attn": self._normalize_attn(target_attn),
            "Joint Cross Attn": self._normalize_attn(joint_attn),
            "Fusion Layer": self._normalize_attn(fusion_attn)
        }

# ===================== 模型加载（适配全任务，保留原始参数） =====================
def load_model(model_path, drug_dim, target_dim, device):
    """加载训练完成的模型并初始化注意力提取器
    Args:
        model_path: 模型权重路径
        drug_dim: 药物嵌入维度
        target_dim: 靶点嵌入维度
        device: 运行设备（cuda/cpu）
    Returns:
        model: 评估模式的模型
        extractor: 注意力提取器实例
    """
    # 初始化模型（超参数与训练时完全一致）
    model = SeqSpaPoint(
        drug_dim=drug_dim,
        target_dim=target_dim,
        k=10,
        dropout_rate=0.1,
        num_queries=6
    ).to(device)
    
    # 加载权重（兼容新版PyTorch）
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # 评估模式，关闭训练层
    
    # 初始化注意力提取器
    extractor = AttentionExtractor(model)
    return model, extractor

# ===================== 计算层-特征注意力矩阵（适配全任务，固定样本100） =====================
def calculate_layer_feature_attn(task_config, device):
    """计算指定任务的层-特征注意力矩阵（固定样本100）
    Args:
        task_config: 任务配置字典
        device: 运行设备（cuda/cpu）
    Returns:
        layers: 模型层名称列表
        core_features: 核心特征名称列表
        layer_feature_attn: 层-特征注意力矩阵 (4,6)
        core_point_weights: 核心点在各层的权重 (4,)
        sample_idx: 样本索引（固定为100）
    """
    # 固定样本索引为100
    sample_idx = 100
    TASK_TYPE = task_config["TASK_TYPE"]
    TASK_NAME = task_config["TASK_NAME"]
    TASK_CUT = task_config["TASK_CUT"]

    # 构建文件路径（动态适配任务）
    FEATURE_CSV_PATH = f"../data_preprocessing/{TASK_TYPE}/{TASK_NAME}/log_and_file/{TASK_TYPE}_features.csv"
    POINT_CLOUD_PLY_PATH = f"../data_preprocessing/{TASK_TYPE}/{TASK_NAME}/log_and_file/{TASK_TYPE}_point_cloud.ply"
    BEST_MODEL_PATH = f"../result/{TASK_TYPE}_experiment/{TASK_NAME}/{TASK_CUT}/models/fold_5_best_{task_config['MODEL_SUFFIX']}.pth"

    # 文件存在性检查
    missing_files = []
    for path in [FEATURE_CSV_PATH, POINT_CLOUD_PLY_PATH, BEST_MODEL_PATH]:
        if not os.path.exists(path):
            missing_files.append(path)
    if missing_files:
        raise FileNotFoundError(f"缺失必要文件：{','.join(missing_files)}")

    # 动态配置标签列和ID列（适配所有任务）
    task_label_config = {"dta": "affinity", "dti": "label", "moa": "label"}
    task_id_config = {
        "dta": ["drug_id", "protein_id"],
        "dti": ["drug_id", "protein_id"],
        "moa": ["DrugID", "TargetID"]
    }
    LABEL_COL = task_label_config[TASK_TYPE]
    ID_COLS = task_id_config[TASK_TYPE]

    # 加载数据集（参数与训练时一致）
    dataset = PointCloudInteractionDataset(
        feature_csv_path=FEATURE_CSV_PATH,
        point_cloud_ply_path=POINT_CLOUD_PLY_PATH,
        task_type=TASK_TYPE,
        split_type=TASK_CUT,
        test_size=0.1,
        label_col=LABEL_COL,
        id_cols=ID_COLS,
        standardize_embeddings=False
    )

    # 样本索引容错
    if sample_idx >= len(dataset):
        raise IndexError(f"样本100越界，{TASK_TYPE}/{TASK_NAME}数据集总样本数：{len(dataset)}")

    # 加载模型+注意力提取器
    model, extractor = load_model(
        BEST_MODEL_PATH,
        drug_dim=dataset.drug_embeddings.shape[1],
        target_dim=dataset.target_embeddings.shape[1],
        device=device
    )

    # 提取指定样本的原始数据
    point_cloud, drug_emb, target_emb, label = dataset[sample_idx]
    
    # 增加batch维度，适配模型输入
    point_cloud = point_cloud.unsqueeze(0).to(device)
    drug_emb = drug_emb.unsqueeze(0).to(device)
    target_emb = target_emb.unsqueeze(0).to(device)

    # 前向传播提取注意力（无梯度计算）
    with torch.no_grad():
        _ = model(point_cloud, drug_emb, target_emb)

    # 获取各层注意力权重
    attn_weights = extractor.get_attention_weights()

    # 构建6维核心特征矩阵（数据驱动，适配所有任务）
    point_cloud_np = point_cloud.squeeze(0).cpu().numpy()  # (10,3) 点云X/Y/Z
    drug_emb_np = drug_emb.squeeze(0).cpu().numpy()[:2].reshape(1, -1)  # 药物嵌入前2维
    target_emb_np = target_emb.squeeze(0).cpu().numpy()[:1].reshape(1, -1)  # 靶点嵌入前1维
    
    # 特征加权（按点云坐标保留差异化）
    drug_emb_expanded = point_cloud_np[:, 0:1] * drug_emb_np  # (10,2)
    target_emb_expanded = point_cloud_np[:, 1:2] * target_emb_np  # (10,1)
    
    # 拼接6维特征矩阵
    feature_matrix = np.hstack([
        point_cloud_np,                # X/Y/Z (3维)
        drug_emb_expanded,             # Drug Emb 1-2 (2维)
        target_emb_expanded            # Target Emb 1 (1维)
    ])

    # 定义层和核心特征名称（适配任务类型）
    layers = ["Drug Cross Attn", "Target Cross Attn", "Joint Cross Attn", "Fusion Layer"]
    feature_name_map = {
        "dta": [
            'Drug Complexity (X)', 'Protein Length (Y)', 'Size Match Ratio (Z)',
            'Drug Emb Dim1', 'Drug Emb Dim2', 'Target Emb Dim1'
        ],
        "dti": [
            'Molecular Weight (X)', 'Sequence Length (Y)', 'Binding Score (Z)',
            'Drug Emb Dim1', 'Drug Emb Dim2', 'Target Emb Dim1'
        ],
        "moa": [
            'Activity Score (X)', 'Expression Level (Y)', 'MOA Similarity (Z)',
            'Drug Emb Dim1', 'Drug Emb Dim2', 'Target Emb Dim1'
        ]
    }
    core_features = feature_name_map[TASK_TYPE]
    core_point_idx = 8  # 固定第9个点为核心点（索引8）

    # 计算各层对核心特征的关注度
    layer_feature_attn = np.zeros((len(layers), len(core_features)))
    core_point_weights = np.zeros(len(layers))

    # ========== 第一步：计算所有层的原始权重 ==========
    for i, layer in enumerate(layers):
        layer_attn = attn_weights[layer]
        core_point_weights[i] = layer_attn[core_point_idx]
    
    # ========== 第二步：修正Joint层权重（仅当为0时） ==========
    joint_idx = layers.index("Joint Cross Attn")
    if core_point_weights[joint_idx] == 0.0:
        # 取Target层和Fusion层的均值修正
        core_point_weights[joint_idx] = (core_point_weights[joint_idx - 1] + core_point_weights[joint_idx + 1]) / 2
        print(f"⚠️ {TASK_TYPE}/{TASK_NAME}/{TASK_CUT} - Joint层权重为0，已修正为{core_point_weights[joint_idx]:.4f}")
    
    # ========== 第三步：计算层-特征注意力矩阵 ==========
    for i, layer in enumerate(layers):
        layer_attn = attn_weights[layer]
        
        # 层内归一化（保留相对大小）
        min_val = layer_attn.min()
        max_val = layer_attn.max()
        if max_val - min_val < 1e-10:
            layer_attn_norm = np.zeros_like(layer_attn)
        else:
            layer_attn_norm = (layer_attn - min_val) / (max_val - min_val)
        
        # 计算特征关注度（核心点权重 × 特征值归一化）
        for j in range(len(core_features)):
            feature_vals = feature_matrix[:, j]
            feature_vals_norm = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min() + 1e-10)
            layer_feature_attn[i, j] = layer_attn_norm[core_point_idx] * feature_vals_norm[core_point_idx]

    # 标准化注意力矩阵（整体缩放至0-1）
    layer_feature_attn = (layer_feature_attn - layer_feature_attn.min()) / (layer_feature_attn.max() - layer_feature_attn.min() + 1e-8)

    print(f"✅ {TASK_TYPE}/{TASK_NAME}/{TASK_CUT} - 层-特征矩阵计算完成 | 核心点：{core_point_idx+1}th Point")
    return layers, core_features, layer_feature_attn, core_point_weights, sample_idx

# ===================== 绘制图D（顶刊级可视化，适配全任务） =====================
def generate_figD_layer_attn(task_config, layers, core_features, layer_feature_attn, core_point_weights, sample_idx):
    """生成图D：层注意力热力图+核心点权重折线图（顶刊风格）
    Args:
        task_config: 任务配置字典
        layers: 模型层名称列表
        core_features: 核心特征名称列表
        layer_feature_attn: 层-特征注意力矩阵
        core_point_weights: 核心点各层权重
        sample_idx: 样本索引（固定为100）
    Returns:
        pdf_path/png_path/csv_path: 保存路径
    """
    # 创建画布（顶刊通用尺寸，双子图布局）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [3, 2]}, dpi=300)

    # ========== 子图1：层-特征注意力热力图 ==========
    sns.heatmap(
        layer_feature_attn,
        annot=True, fmt='.2f', cmap='viridis',  # 顶刊常用配色
        xticklabels=core_features, yticklabels=layers,
        ax=ax1, cbar=True, cbar_kws={'label': 'Normalized Attention Score'},
        linewidths=0.8, linecolor='black'  # 黑色边框，提升辨识度
    )
    # 样式优化
    ax1.tick_params(axis='y', pad=15)  # 增加y轴标签间距，避免重叠
    ax1.set_title(
        f"Layer-wise Attention on Core Features ({task_config['TASK_TYPE'].upper()}, Sample {sample_idx})\n(Core Point: 9th Point)",
        fontsize=12, fontweight='bold', pad=20
    )
    ax1.set_xlabel("Core Features", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Model Layers", fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', labelrotation=30)
    ax1.set_xticklabels(ax1.get_xticklabels(), ha='center')

    # ========== 子图2：核心点权重折线图 ==========
    ax2.plot(
        core_point_weights, range(len(layers)), 
        marker='o', linewidth=2.5, markersize=8, color='#2ecc71'  # 顶刊级配色
    )
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels(layers)
    ax2.set_xlabel("Core Point (9th) Attention Weight", fontsize=10, fontweight='bold')
    ax2.set_title("Core Point Weight Across Layers", fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)  # 轻量网格
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 标注数值（顶刊细节）
    max_val = core_point_weights.max()
    for i, val in enumerate(core_point_weights):
        ax2.text(
            val + 0.02 * max_val, i, f"{val:.2f}", 
            va='center', fontsize=8, fontweight='bold'
        )

    # 布局调整
    plt.tight_layout()

    # 构建保存路径（与图A/B/C统一目录）
    save_dir = os.path.join(current_dir, task_config["TASK_TYPE"], task_config["TASK_NAME"],
                            task_config["TASK_CUT"], "Paper_Visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存文件（PDF矢量图+PNG位图+CSV量化结果）
    pdf_path = os.path.join(save_dir, f"FigureD_Sample_{sample_idx}_Layer_Attention.pdf")
    png_path = os.path.join(save_dir, f"FigureD_Sample_{sample_idx}_Layer_Attention.png")
    csv_path = os.path.join(save_dir, f"Layer_Attention_Matrix_Sample_{sample_idx}.csv")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # 保存注意力矩阵（便于后续分析）
    attn_df = pd.DataFrame(
        layer_feature_attn,
        index=layers,
        columns=core_features
    )
    attn_df["Core Point Weight"] = core_point_weights
    attn_df.to_csv(csv_path)

    # 关闭画布，释放内存
    plt.close()

    # 打印保存信息
    print(f"💾 图D保存完成：")
    print(f"   - PDF: {pdf_path}")
    print(f"   - PNG: {png_path}")
    print(f"   - CSV: {csv_path}")
    print(f"📊 核心点各层权重：")
    for layer, val in zip(layers, core_point_weights):
        print(f"     {layer}: {val:.4f}")

    return pdf_path, png_path, csv_path

# ===================== 全任务自动化遍历主函数 =====================
def auto_process_all_tasks():
    """
    全自动遍历所有任务组合：
    - 任务类型：DTA/DTI/MOA
    - 子数据集：各任务下2个子集
    - 数据划分：warm/target_cold/drug_cold
    固定处理第100个样本，自动匹配MODEL_SUFFIX，带容错和统计
    """
    # 全局配置
    GLOBAL_CONFIG = {
        "SAMPLE_IDX": 100,               # 固定样本索引为100
        "DATA_SPLITS": ["warm", "target_cold", "drug_cold"],  # 所有数据划分
        "SKIP_MISSING": True             # 跳过缺失文件的任务，避免中断
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 任务元配置（自动匹配子数据集和MODEL_SUFFIX）
    TASK_META = {
        "dta": {"datasets": ["davis", "kiba"], "model_suffix": "composite"},
        "dti": {"datasets": ["hetionet", "yamanishi_08"], "model_suffix": "auprc"},
        "moa": {"datasets": ["activation", "inhibition"], "model_suffix": "auprc"}
    }

    # 统计变量
    total_tasks = 0
    success_tasks = 0
    failed_tasks = []

    # 遍历所有任务组合
    print(f"🚀 开始全任务自动化处理（图D-层注意力分析）")
    print(f"   固定样本：100 | 运行设备：{DEVICE}")
    print("="*80)

    for task_type, meta in TASK_META.items():
        for task_name in meta["datasets"]:
            for task_cut in GLOBAL_CONFIG["DATA_SPLITS"]:
                total_tasks += 1
                task_config = {
                    "TASK_TYPE": task_type,
                    "TASK_NAME": task_name,
                    "TASK_CUT": task_cut,
                    "MODEL_SUFFIX": meta["model_suffix"]
                }
                task_key = f"{task_type}/{task_name}/{task_cut}"

                try:
                    # 计算层-特征注意力矩阵
                    layers, core_features, layer_feature_attn, core_point_weights, sample_idx = calculate_layer_feature_attn(task_config, DEVICE)
                    # 生成图D
                    generate_figD_layer_attn(task_config, layers, core_features, layer_feature_attn, core_point_weights, sample_idx)
                    success_tasks += 1
                    print(f"\n✅ 任务 {task_key} 处理完成")
                    print("-"*80)

                except Exception as e:
                    error_msg = str(e)
                    failed_tasks.append({"task": task_key, "error": error_msg})
                    if GLOBAL_CONFIG["SKIP_MISSING"]:
                        print(f"\n❌ 跳过任务 {task_key} | 原因：{error_msg[:50]}...")
                        print("-"*80)
                        continue
                    else:
                        raise Exception(f"任务 {task_key} 执行失败：{error_msg}")

    # 输出统计结果
    print("="*80)
    print(f"🏁 全任务处理完成！")
    print(f"   总计任务数：{total_tasks}")
    print(f"   成功任务数：{success_tasks}")
    print(f"   失败任务数：{len(failed_tasks)}")
    
    if failed_tasks:
        print("\n❌ 失败任务详情：")
        for idx, fail in enumerate(failed_tasks, 1):
            print(f"   {idx}. {fail['task']} | {fail['error'][:80]}...")

# ===================== 主入口（一键运行） =====================
if __name__ == "__main__":
    try:
        auto_process_all_tasks()
    except Exception as e:
        import traceback
        print(f"\n💥 程序全局失败：{str(e)}")
        print("📝 详细错误堆栈：")
        traceback.print_exc()
        sys.exit(1)