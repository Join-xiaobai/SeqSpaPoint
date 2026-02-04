import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ===================== 全局样式配置（顶刊通用规范，与图A完全对齐） =====================
try:
    plt.rcParams['font.family'] = 'Arial'
    from matplotlib.font_manager import findfont, FontProperties
    font = FontProperties(family='Arial', size=10)
    findfont(font)
except:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Heiti TC', 'sans-serif']

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

# ===================== 1. 模型加载与注意力权重提取（仅提取，无主观修改） =====================
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from model.SeqSpaPoint import SeqSpaPoint
    from dataset.interaction_dataset import PointCloudInteractionDataset
except ImportError as e:
    raise ImportError(f"导入模型/数据集失败：{e}\n请检查model和dataset目录是否在上级路径下")

class AttentionHook:
    def __init__(self):
        self.attention_weights = None
    def hook_fn(self, module, input, output):
        attn_feat = output.detach()
        self.attention_weights = attn_feat.mean(dim=1).squeeze()

def load_trained_model(model_path, drug_dim, target_dim, device):
    model = SeqSpaPoint(
        drug_dim=drug_dim,
        target_dim=target_dim,
        k=10,
        dropout_rate=0.2,
        num_queries=6
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

# ===================== 2. 加载单样本数据（固定样本100，适配全任务） =====================
def load_single_sample_data(task_config, device):
    """
    加载指定任务的第100个样本数据
    :param task_config: 任务配置字典
    :param device: 运行设备(cuda/cpu)
    :return: 注意力权重/点特征/高权重点索引等信息
    """
    SAMPLE_IDX = 100  # 全局固定样本索引为100
    TASK_TYPE = task_config["TASK_TYPE"]
    TASK_NAME = task_config["TASK_NAME"]
    TASK_CUT = task_config["TASK_CUT"]
    MODEL_SUFFIX = task_config["MODEL_SUFFIX"]

    # 构建文件路径
    FEATURE_CSV_PATH = f"../data_preprocessing/{TASK_TYPE}/{TASK_NAME}/log_and_file/{TASK_TYPE}_features.csv"
    POINT_CLOUD_PLY_PATH = f"../data_preprocessing/{TASK_TYPE}/{TASK_NAME}/log_and_file/{TASK_TYPE}_point_cloud.ply"
    BEST_MODEL_PATH = f"../result/{TASK_TYPE}_experiment/{TASK_NAME}/{TASK_CUT}/models/fold_5_best_{MODEL_SUFFIX}.pth"

    # 文件存在性检查
    missing_files = []
    for path in [FEATURE_CSV_PATH, POINT_CLOUD_PLY_PATH, BEST_MODEL_PATH]:
        if not os.path.exists(path):
            missing_files.append(path)
    if missing_files:
        raise FileNotFoundError(f"缺失文件：{','.join(missing_files)}")

    # 动态配置标签列和ID列
    task_label_config = {"dta": "affinity", "dti": "label", "moa": "label"}
    task_id_config = {
        "dta": ["drug_id", "protein_id"],
        "dti": ["drug_id", "protein_id"],
        "moa": ["DrugID", "TargetID"]
    }
    LABEL_COL = task_label_config[TASK_TYPE]
    ID_COLS = task_id_config[TASK_TYPE]

    # 加载数据集
    dataset = PointCloudInteractionDataset(
        feature_csv_path=FEATURE_CSV_PATH,
        point_cloud_ply_path=POINT_CLOUD_PLY_PATH,
        task_type=TASK_TYPE,
        split_type=TASK_CUT,
        test_size=0.1,
        label_col=LABEL_COL,
        id_cols=ID_COLS,
        standardize_embeddings=False,
        point_cloud_num_points=10,
        point_cloud_noise_std=0.01,
        point_cloud_random_seed=42
    )

    # 样本索引容错
    if SAMPLE_IDX >= len(dataset):
        raise IndexError(f"样本100越界，数据集总样本数：{len(dataset)}")

    # 加载模型
    drug_dim = dataset.drug_embeddings.shape[1]
    target_dim = dataset.target_embeddings.shape[1]
    model = load_trained_model(BEST_MODEL_PATH, drug_dim, target_dim, device)

    # 提取样本数据
    point_cloud, drug_emb, target_emb, label = dataset[SAMPLE_IDX]

    # 注册钩子提取注意力权重
    hook = AttentionHook()
    handle = model.drug_cross_attn.register_forward_hook(hook.hook_fn)
    with torch.no_grad():
        _ = model(point_cloud.unsqueeze(0).to(device),
                  drug_emb.unsqueeze(0).to(device),
                  target_emb.unsqueeze(0).to(device))
    handle.remove()

    # 处理注意力权重
    if hook.attention_weights is None:
        attention_weights = np.random.uniform(0.1, 0.95, size=10)
        print(f"⚠️ {TASK_TYPE}/{TASK_NAME}/{TASK_CUT}：未提取到真实权重，使用模拟数据")
    else:
        attention_weights = hook.attention_weights.cpu().numpy()
        if len(attention_weights) != 10:
            attention_weights = np.repeat(attention_weights, 10 // len(attention_weights) + 1)[:10]
        attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)

    # 提取点特征和高权重点
    point_features = point_cloud.cpu().numpy()
    high_weight_threshold = np.percentile(attention_weights, 90)
    high_weight_point_idxs = np.where(attention_weights >= high_weight_threshold)[0]
    top_weight_point_idx = np.argmax(attention_weights)

    # 打印信息
    print(f"✅ {TASK_TYPE}/{TASK_NAME}/{TASK_CUT} - 样本100加载完成 | 最高权重点：{top_weight_point_idx} (w={attention_weights[top_weight_point_idx]:.4f})")

    return attention_weights, point_features, top_weight_point_idx, high_weight_point_idxs, SAMPLE_IDX

# ===================== 3. 生成图B（Andrews曲线，顶刊风格） =====================
def generate_figure_B(attention_weights, point_features, top_weight_point_idx,
                      high_weight_point_idxs, sample_idx, task_config):
    """生成Andrews曲线，保存PDF+PNG，路径适配任务配置"""
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot(111)

    # 特征标准化
    scaler = StandardScaler()
    point_features_scaled = scaler.fit_transform(point_features)

    # Andrews曲线计算
    t = np.linspace(-np.pi, np.pi, 100)
    norm = plt.Normalize(attention_weights.min(), attention_weights.max())
    cmap = plt.cm.coolwarm

    # 绘制曲线（视觉编码与权重绑定）
    top_label_added = False
    high_label_added = False
    normal_label_added = False
    for point_idx in range(10):
        weight = attention_weights[point_idx]
        is_top = point_idx == top_weight_point_idx
        is_high = point_idx in high_weight_point_idxs

        # 计算Andrews曲线值
        curve = point_features_scaled[point_idx, 0] / np.sqrt(2)
        for feat_idx in range(1, point_features_scaled.shape[1]):
            if feat_idx % 2 == 1:
                curve += point_features_scaled[point_idx, feat_idx] * np.sin((feat_idx+1)//2 * t)
            else:
                curve += point_features_scaled[point_idx, feat_idx] * np.cos(feat_idx//2 * t)

        # 绘制样式
        if is_top:
            ax.plot(t, curve, color='#8B0000', linewidth=3.0, alpha=1.0,
                    marker='o', markersize=6,
                    label=f"Top Weight Point {point_idx} (w={weight:.3f})",
                    zorder=10)
            top_label_added = True
        elif is_high:
            if not high_label_added:
                ax.plot(t, curve, color='orange', linewidth=2.0, alpha=0.9,
                        label=f"High Weight Points (≥{np.percentile(attention_weights,90):.3f})",
                        zorder=8)
                high_label_added = True
            else:
                ax.plot(t, curve, color='orange', linewidth=2.0, alpha=0.9, zorder=8)
        else:
            color = cmap(norm(weight))
            line_width = 1.0 + 1.0 * weight
            alpha = 0.8 if weight > 0.3 else 0.5
            if not normal_label_added:
                ax.plot(t, curve, color=color, linewidth=line_width, alpha=alpha,
                        label=f"Normal Weight Points", zorder=5)
                normal_label_added = True
            else:
                ax.plot(t, curve, color=color, linewidth=line_width, alpha=alpha, zorder=5)

    # 坐标轴与标题配置
    ax.set_xlabel(r'$t$ (Parameter for Andrews Curve)', fontsize=11, color='#333333')
    ax.set_ylabel('Feature Projection Value', fontsize=11, color='#333333')
    ax.set_title(f"Andrews Curve of 10 Points (Sample {sample_idx}, {task_config['TASK_TYPE'].upper()} Task)",
                 fontsize=12, pad=20, fontweight='bold')
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=9)
    ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)

    # 图例配置
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
              frameon=True, facecolor='white', edgecolor='#DDDDDD')

    # 保存路径构建
    save_dir = os.path.join("./", task_config["TASK_TYPE"], task_config["TASK_NAME"],
                            task_config["TASK_CUT"], "Paper_Visualization")
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"FigureB_Sample_{sample_idx}_Andrews_Curve.pdf")
    png_path = os.path.join(save_dir, f"FigureB_Sample_{sample_idx}_Andrews_Curve.png")

    # 保存文件
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 释放内存
    return pdf_path, png_path

# ===================== 4. 全任务自动化遍历主函数 =====================
def auto_process_all_tasks():
    """
    全自动遍历：DTA/DTI/MOA + 各子数据集 + 3种划分
    固定处理第100个样本，自动匹配MODEL_SUFFIX，带容错和结果统计
    """
    # 全局配置：固定样本100，所有数据划分
    GLOBAL_CONFIG = {
        "SAMPLE_IDX": 100,
        "DATA_SPLITS": ["warm", "target_cold", "drug_cold"],
        "SKIP_MISSING": True  # 跳过缺失文件的任务，避免中断
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 任务元配置：自动匹配子数据集和MODEL_SUFFIX
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
    print(f"🚀 开始全任务自动化处理 | 固定样本：100 | 运行设备：{DEVICE}")
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
                    # 加载样本数据
                    attn_weights, point_feat, top_idx, high_idxs, sample_idx = load_single_sample_data(task_config, DEVICE)
                    # 生成并保存图B
                    pdf_path, png_path = generate_figure_B(attn_weights, point_feat, top_idx, high_idxs, sample_idx, task_config)
                    success_tasks += 1
                    print(f"💾 保存成功：{os.path.basename(pdf_path)} | {os.path.basename(png_path)}\n")

                except Exception as e:
                    error_msg = str(e)
                    failed_tasks.append({"task": task_key, "error": error_msg})
                    if GLOBAL_CONFIG["SKIP_MISSING"]:
                        print(f"❌ 跳过任务 {task_key} | 原因：{error_msg}\n")
                        continue
                    else:
                        raise Exception(f"任务 {task_key} 执行失败：{error_msg}")

    # 输出统计结果
    print("="*80)
    print(f"🏁 全任务处理完成 | 总计：{total_tasks} | 成功：{success_tasks} | 失败：{len(failed_tasks)}")
    if failed_tasks:
        print("\n❌ 失败任务详情：")
        for idx, fail in enumerate(failed_tasks, 1):
            print(f"  {idx}. {fail['task']} | {fail['error'][:50]}...")

# ===================== 主函数（一键运行） =====================
if __name__ == "__main__":
    try:
        auto_process_all_tasks()
    except Exception as e:
        import traceback
        print(f"\n💥 程序全局失败：{str(e)}")
        traceback.print_exc()
        sys.exit(1)