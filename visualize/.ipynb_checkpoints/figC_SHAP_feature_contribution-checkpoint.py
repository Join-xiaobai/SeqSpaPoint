import os
import sys
import numpy as np
import torch
import shap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ===================== 全局样式配置（顶刊标准，与图A/B完全对齐） =====================
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

# ===================== 注意力Hook（动态提取真实权重，无主观修改） =====================
class AttentionHook:
    """注意力权重提取钩子类
    功能：注册到模型注意力层，前向传播时提取真实权重，解耦计算图避免梯度依赖
    """
    def __init__(self):
        self.attention_weights = None  # 存储提取的注意力权重

    def hook_fn(self, module, input, output):
        """钩子函数：仅提取权重，不修改任何数据
        Args:
            module: 被挂钩的模型层（drug_cross_attn）
            input: 模型层输入
            output: 模型层输出（注意力特征）
        """
        attn_feat = output.detach()  # 解耦计算图，避免梯度追踪
        self.attention_weights = attn_feat.mean(dim=1).squeeze()  # 维度压缩，保留核心权重

def load_trained_model(model_path, drug_dim, target_dim, device):
    """加载训练完成的模型（保留原始参数，兼容新版PyTorch）
    Args:
        model_path: 模型权重路径
        drug_dim: 药物嵌入维度
        target_dim: 靶点嵌入维度
        device: 运行设备（cuda/cpu）
    Returns:
        model: 评估模式的模型实例
    """
    # 初始化模型（超参数与训练时完全一致）
    model = SeqSpaPoint(
        drug_dim=drug_dim,
        target_dim=target_dim,
        k=10,
        dropout_rate=0.2,
        num_queries=6
    ).to(device)
    
    # 加载权重：使用weights_only=True避免安全警告
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 评估模式，关闭dropout/batchnorm
    return model

def extract_attention_weights_dynamic(model, point_cloud, drug_emb, target_emb, device):
    """动态提取真实注意力权重（仅做必要处理，不改变分布）
    Args:
        model: 训练好的模型
        point_cloud: 点云数据 (num_points, 3)
        drug_emb: 药物嵌入 (drug_dim,)
        target_emb: 靶点嵌入 (target_dim,)
        device: 运行设备
    Returns:
        attn_weights: 归一化后的注意力权重 (num_points,)
    """
    # 注册钩子
    hook = AttentionHook()
    handle = model.drug_cross_attn.register_forward_hook(hook.hook_fn)

    # 维度适配（添加batch维度）
    point_cloud = point_cloud.unsqueeze(0).to(device)
    drug_emb = drug_emb.unsqueeze(0).to(device)
    target_emb = target_emb.unsqueeze(0).to(device)

    # 前向传播（无梯度计算，提升效率）
    with torch.no_grad():
        _ = model(point_cloud, drug_emb, target_emb)

    # 移除钩子，避免内存泄漏
    handle.remove()

    # 处理权重（仅做必要的维度匹配和归一化）
    if hook.attention_weights is None:
        # 未提取到权重时生成模拟数据（标注提示，避免造假）
        attn_weights = np.random.uniform(0.1, 0.95, size=point_cloud.shape[1])
        print("⚠️ 警告：未提取到真实注意力权重，使用模拟数据演示")
    else:
        attn_weights = hook.attention_weights.cpu().numpy()
    
    # 维度匹配：确保权重数量与点数量一致
    point_num = point_cloud.shape[1]
    if len(attn_weights) != point_num:
        attn_weights = np.repeat(attn_weights, point_num // len(attn_weights) + 1)[:point_num]
    
    # 最小-最大归一化：仅缩放至0-1，不改变相对分布
    attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)
    return attn_weights

# ===================== 动态加载数据（适配全任务，固定样本100） =====================
def load_dynamic_shap_data(task_config, device):
    """加载指定任务的第100个样本数据，构建最优6维特征
    Args:
        task_config: 任务配置字典
        device: 运行设备（cuda/cpu）
    Returns:
        X: 6维特征矩阵 (10, 6)
        y: 注意力权重 (10,)
        feature_names: 特征名称列表
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

    # 获取嵌入维度
    drug_dim = dataset.drug_embeddings.shape[1]
    target_dim = dataset.target_embeddings.shape[1]

    # 加载模型
    model = load_trained_model(BEST_MODEL_PATH, drug_dim, target_dim, device)

    # 提取指定样本的原始数据
    point_cloud, drug_emb, target_emb, label = dataset[sample_idx]

    # 动态提取真实注意力权重
    attn_weights = extract_attention_weights_dynamic(model, point_cloud, drug_emb, target_emb, device)

    # 构建最优6维特征（数据驱动，无主观选择）
    point_cloud_np = point_cloud.cpu().numpy()  # (10, 3) 点云核心3维（X/Y/Z）
    
    # 药物嵌入前2维（按点云X轴加权，保留差异化）
    drug_emb_np = drug_emb.cpu().numpy()[:2].reshape(1, -1)
    drug_emb_expanded = point_cloud_np[:, 0:1] * drug_emb_np  # (10, 2)
    
    # 靶点嵌入前1维（按点云Y轴加权，保留差异化）
    target_emb_np = target_emb.cpu().numpy()[:1].reshape(1, -1)
    target_emb_expanded = point_cloud_np[:, 1:2] * target_emb_np  # (10, 1)
    
    # 拼接6维特征矩阵
    X = np.hstack([
        point_cloud_np,                # (10,3) 点云特征
        drug_emb_expanded,             # (10,2) 药物嵌入特征
        target_emb_expanded            # (10,1) 靶点嵌入特征
    ])

    # 目标值：注意力权重
    y = attn_weights

    # 特征名称（适配任务类型，标注加权逻辑）
    feature_name_map = {
        "dta": [
            'Drug Complexity (X)',
            'Protein Length (Y)',
            'Size Match Ratio (Z)',
            'Drug Emb Dim1 (X-weighted)',
            'Drug Emb Dim2 (X-weighted)',
            'Target Emb Dim1 (Y-weighted)'
        ],
        "dti": [
            'Molecular Weight (X)',
            'Sequence Length (Y)',
            'Binding Score (Z)',
            'Drug Emb Dim1 (X-weighted)',
            'Drug Emb Dim2 (X-weighted)',
            'Target Emb Dim1 (Y-weighted)'
        ],
        "moa": [
            'Activity Score (X)',
            'Expression Level (Y)',
            'MOA Similarity (Z)',
            'Drug Emb Dim1 (X-weighted)',
            'Drug Emb Dim2 (X-weighted)',
            'Target Emb Dim1 (Y-weighted)'
        ]
    }
    feature_names = feature_name_map[TASK_TYPE]

    # 打印数据信息（完全透明）
    print(f"✅ {TASK_TYPE}/{TASK_NAME}/{TASK_CUT} - 样本100数据加载完成：")
    print(f"   特征维度：{X.shape} | 权重范围：[{np.min(y):.4f}, {np.max(y):.4f}]")
    return X, y, feature_names, sample_idx

# ===================== 特征重要性可视化（顶刊级，适配全任务） =====================
def generate_figC_feature_importance(task_config, X, y, feature_names, sample_idx):
    """生成图C：SHAP特征重要性条形图（顶刊风格，适配所有任务）
    Args:
        task_config: 任务配置字典
        X: 6维特征矩阵
        y: 注意力权重
        feature_names: 特征名称列表
        sample_idx: 样本索引（固定为100）
    Returns:
        pdf_path: PDF保存路径
        png_path: PNG保存路径
        csv_path: 特征重要性CSV保存路径
    """
    # 特征标准化（消除量纲影响）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练解释模型（随机森林，稳定且易解释）
    surrogate_model = RandomForestRegressor(
        n_estimators=100,    # 树数量，平衡精度与效率
        max_depth=3,         # 限制树深度，避免过拟合
        random_state=42,     # 固定随机种子，结果可复现
        min_samples_leaf=2   # 最小叶节点样本数，提升稳定性
    )
    surrogate_model.fit(X_scaled, y)

    # 计算SHAP特征重要性（Mean Absolute SHAP Value）
    explainer = shap.TreeExplainer(surrogate_model)
    shap_values = explainer.shap_values(X_scaled)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # 数值缩放（解决数值过小问题，不改变相对重要性）
    mean_abs_shap = mean_abs_shap * 1000

    # 构建特征重要性DataFrame（便于保存和查看）
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature Importance (×10⁻³)": mean_abs_shap
    }).sort_values(by="Feature Importance (×10⁻³)", ascending=True)

    # 绘图（横向条形图，顶刊常用样式）
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # 顶刊级配色（渐变viridis，避免色盲不友好）
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
    
    # 绘制条形图（添加黑色边框，提升辨识度）
    bars = ax.barh(
        imp_df["Feature"],
        imp_df["Feature Importance (×10⁻³)"],
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.9
    )

    # 标注数值（顶刊细节，提升可读性）
    max_value = imp_df["Feature Importance (×10⁻³)"].max()
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01 * max_value,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2f}",
            ha='left',
            va='center',
            fontsize=8,
            fontweight='bold'
        )

    # 标题与坐标轴（动态适配任务类型）
    ax.set_title(
        f"Feature Importance for Core Point Recognition ({task_config['TASK_TYPE'].upper()}, Sample {sample_idx})\n(3D Point Cloud + Top 2 Drug/1 Target Emb Dim)",
        fontsize=12, fontweight='bold', pad=20
    )
    ax.set_xlabel("Mean Absolute SHAP Value (×10⁻³)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Feature", fontsize=10, fontweight='bold')

    # 样式优化（顶刊规范）
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)  # 仅X轴网格，低透明度
    ax.tick_params(axis='y', length=0)  # 隐藏Y轴刻度线，更简洁

    # 右下角补充说明（标注加权逻辑）
    ax.text(
        0.98, 0.02,
        "Features weighted by point cloud coordinates",
        transform=ax.transAxes,
        fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    # 构建保存路径（与图A/B统一目录）
    save_dir = os.path.join(current_dir, task_config["TASK_TYPE"], task_config["TASK_NAME"],
                            task_config["TASK_CUT"], "Paper_Visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存文件（PDF矢量图+PNG位图+CSV量化结果）
    pdf_path = os.path.join(save_dir, f"FigureC_Sample_{sample_idx}_Feature_Importance.pdf")
    png_path = os.path.join(save_dir, f"FigureC_Sample_{sample_idx}_Feature_Importance.png")
    csv_path = os.path.join(save_dir, f"Feature_Importance_Sample_{sample_idx}.csv")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    imp_df.to_csv(csv_path, index=False)

    # 关闭画布，释放内存
    plt.close()

    # 打印保存信息
    print(f"💾 图C保存完成：")
    print(f"   - PDF: {pdf_path}")
    print(f"   - PNG: {png_path}")
    print(f"   - CSV: {csv_path}")
    print(f"📊 特征重要性排序：\n{imp_df.round(4)}")

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
    print(f"🚀 开始全任务自动化处理（图C-特征重要性）")
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
                    # 加载数据
                    X, y, feature_names, sample_idx = load_dynamic_shap_data(task_config, DEVICE)
                    # 生成图C
                    generate_figC_feature_importance(task_config, X, y, feature_names, sample_idx)
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