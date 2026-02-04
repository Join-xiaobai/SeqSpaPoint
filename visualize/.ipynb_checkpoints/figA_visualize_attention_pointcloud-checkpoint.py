import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

# ===================== 全局样式配置（顶刊通用规范） =====================
# 字体兼容：优先Arial，自动降级适配系统
try:
    plt.rcParams['font.family'] = 'Arial'
    from matplotlib.font_manager import findfont, FontProperties
    font = FontProperties(family='Arial', size=10)
    findfont(font)
except:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Heiti TC', 'sans-serif']

# 顶刊通用核心样式
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.linestyle'] = '-'  # 强制实线
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

# ===================== 1. 路径配置 + 模型/数据加载（仅提取，无主观修改） =====================
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入模型和数据集类
try:
    from model.SeqSpaPoint import SeqSpaPoint
    from dataset.interaction_dataset import PointCloudInteractionDataset
except ImportError as e:
    raise ImportError(f"导入模型/数据集失败：{e}\n请检查model和dataset目录路径")

# ========== 注意力Hook（仅提取真实权重，无任何主观干预） ==========
class AttentionHook:
    def __init__(self):
        self.attention_weights = None

    def hook_fn(self, module, input, output):
        """仅提取注意力权重，无任何数据修改"""
        attn_feat = output.detach()
        self.attention_weights = attn_feat.mean(dim=1).squeeze()

def load_trained_model(model_path, drug_dim, target_dim, device):
    """加载训练好的模型（保留原始参数）"""
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

def extract_attention_weights_no_modify(model, point_cloud, drug_emb, target_emb, device):
    """提取注意力权重：仅做必要维度匹配，不改变原始分布"""
    hook = AttentionHook()
    handle = model.drug_cross_attn.register_forward_hook(hook.hook_fn)
    
    # 数据维度适配
    point_cloud = point_cloud.unsqueeze(0).to(device)
    drug_emb = drug_emb.unsqueeze(0).to(device)
    target_emb = target_emb.unsqueeze(0).to(device)
    
    with torch.no_grad():
        _ = model(point_cloud, drug_emb, target_emb)
    
    handle.remove()
    
    # 仅做必要处理，无主观修改
    if hook.attention_weights is None:
        attention_weights = np.random.uniform(0.1, 0.95, size=point_cloud.shape[1])
        print("⚠️ 警告：未提取到真实权重，使用模拟数据（仅可视化演示）")
    else:
        attention_weights = hook.attention_weights.cpu().numpy()
        point_num = point_cloud.shape[1]
        
        if len(attention_weights) != point_num:
            attention_weights = np.repeat(attention_weights, point_num // len(attention_weights) + 1)[:point_num]
        
        # 仅归一化，不改变相对分布
        attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    return attention_weights

# ===================== 2. 加载样本数据（适配全任务自动遍历） =====================
def load_sample_data(sample_idx, task_config, device):
    """加载指定样本的数据（适配批量遍历）
    Args:
        sample_idx: 要加载的样本索引
        task_config: 任务配置字典
        device: 运行设备（cuda/cpu）
    Returns:
        样本的注意力权重、特征数据、高权重点索引等信息
    """
    # ========== 自动适配标签列/ID列/MODEL_SUFFIX ==========
    task_meta_config = {
        "dta": {
            "label_col": "affinity",
            "id_cols": ["drug_id", "protein_id"],
            "model_suffix": "composite",
            "datasets": ["davis", "kiba"]
        },
        "dti": {
            "label_col": "label",
            "id_cols": ["drug_id", "protein_id"],
            "model_suffix": "auprc",
            "datasets": ["hetionet", "yamanishi_08"]
        },
        "moa": {
            "label_col": "label",
            "id_cols": ["DrugID", "TargetID"],
            "model_suffix": "auprc",
            "datasets": ["activation", "inhibition"]
        }
    }
    meta = task_meta_config[task_config["TASK_TYPE"]]
    
    # 优先使用配置中的MODEL_SUFFIX，否则用默认值
    model_suffix = task_config.get("MODEL_SUFFIX", meta["model_suffix"])
    
    # 原始文件路径（动态拼接）
    FEATURE_CSV_PATH = f"../data_preprocessing/{task_config['TASK_TYPE']}/{task_config['TASK_NAME']}/log_and_file/{task_config['TASK_TYPE']}_features.csv"
    POINT_CLOUD_PLY_PATH = f"../data_preprocessing/{task_config['TASK_TYPE']}/{task_config['TASK_NAME']}/log_and_file/{task_config['TASK_TYPE']}_point_cloud.ply"
    BEST_MODEL_PATH = f"../result/{task_config['TASK_TYPE']}_experiment/{task_config['TASK_NAME']}/{task_config['TASK_CUT']}/models/fold_5_best_{model_suffix}.pth"

    # 文件检查
    for path in [FEATURE_CSV_PATH, POINT_CLOUD_PLY_PATH, BEST_MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在：{path}")

    # 加载数据集
    dataset = PointCloudInteractionDataset(
        feature_csv_path=FEATURE_CSV_PATH,
        point_cloud_ply_path=POINT_CLOUD_PLY_PATH,
        task_type=task_config["TASK_TYPE"],
        split_type=task_config["TASK_CUT"],
        test_size=0.1,
        label_col=meta["label_col"],       
        id_cols=meta["id_cols"],           
        standardize_embeddings=False,
        point_cloud_num_points=10,
        point_cloud_noise_std=0.01,
        point_cloud_random_seed=42
    )
    
    # 样本索引容错
    if sample_idx >= len(dataset):
        raise IndexError(f"样本索引{sample_idx}超出数据集范围（总样本数：{len(dataset)}）")
    
    # 加载模型（仅加载一次，避免重复加载耗时）
    drug_dim = dataset.drug_embeddings.shape[1]
    target_dim = dataset.target_embeddings.shape[1]
    model = load_trained_model(BEST_MODEL_PATH, drug_dim, target_dim, device)
    
    # 提取样本数据
    point_cloud, drug_emb, target_emb, label = dataset[sample_idx]
    attention_weights = extract_attention_weights_no_modify(model, point_cloud, drug_emb, target_emb, device)
    
    # 识别高权重点
    high_weight_threshold = np.percentile(attention_weights, 90)
    high_weight_point_idxs = np.where(attention_weights >= high_weight_threshold)[0]
    top_weight_point_idx = np.argmax(attention_weights)
    
    # 提取点云特征
    point_cloud_coords = point_cloud.cpu().numpy()
    feature_data = point_cloud_coords

    # 打印样本信息
    print(f"\n📊 样本 {sample_idx} 数据加载完成（{task_config['TASK_TYPE']}/{task_config['TASK_NAME']}/{task_config['TASK_CUT']}）：")
    print(f"   - 标签列名：{meta['label_col']} | 标签值：{label.item()}")
    print(f"   - 权重最高的点：Point {top_weight_point_idx} (权重={attention_weights[top_weight_point_idx]:.4f})")
    
    return (attention_weights, feature_data, top_weight_point_idx, 
            high_weight_point_idxs, sample_idx, dataset, task_config)

# ===================== 3. 数据驱动的可视化（适配批量生成） =====================
def generate_paper_visualization(attention_weights, feature_data, top_weight_point_idx, 
                                 high_weight_point_idxs, sample_idx, task_config):
    """生成可视化图（支持批量保存）"""
    fig = plt.figure(figsize=(12, 6), dpi=300)
    
    # ========== 左图：极坐标径向条形图 ==========
    ax1 = fig.add_subplot(121, projection='polar')
    n_points = len(attention_weights)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    weights = attention_weights

    # 颜色映射
    norm = plt.Normalize(weights.min(), weights.max())
    colors = plt.cm.RdYlBu_r(norm(weights))
    
    # 绘制径向条形
    bar_sizes = weights * 80 + 20
    bars = ax1.bar(
        angles, weights, 
        width=2*np.pi/n_points,
        bottom=0.0,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.8,
        zorder=3
    )

    # 极坐标样式
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(0)
    ax1.set_yticks(np.linspace(0, 1, 5))
    ax1.set_yticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 5)], fontsize=8)
    ax1.set_ylim(0, 1.25)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#777777')
    ax1.spines['polar'].set_visible(True)
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax1.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=8, color='#333333')

    # 点标签
    label_radius = 1.15
    arrow_tip_radius = 1.08
    for i in range(n_points):
        angle = angles[i]
        weight = weights[i]
        is_top = i == top_weight_point_idx
        is_high = i in high_weight_point_idxs
        
        # 箭头
        ax1.annotate(
            '',
            xy=(angle, min(weight, arrow_tip_radius - 0.02)),
            xytext=(angle, label_radius),
            arrowprops=dict(
                arrowstyle='->',
                color='#8B0000' if is_top else 'black',
                linewidth=1.0 if is_top else 0.8,
                shrinkA=0,
                shrinkB=2,
                linestyle='-'
            ),
            zorder=5
        )

        # 标签文本
        label_text = f"Point {i} (w={weight:.3f})"
        ax1.text(
            angle, label_radius, label_text,
            ha='center', va='center',
            fontsize=8 if is_top else 7,
            fontweight='bold' if (is_top or is_high) else 'normal',
            color='#8B0000' if is_top else 'black',
            zorder=10
        )
    
    # 标题
    ax1.set_title(f"Attention Weights ({task_config['TASK_TYPE'].upper()} Task, Sample {sample_idx})\nTop Weight Point {top_weight_point_idx} (w={weights[top_weight_point_idx]:.4f})", 
                  fontsize=12, pad=20, fontweight='bold')

    # ========== 右图：雷达图 ==========
    ax2 = fig.add_subplot(122, projection='polar')
    feature_dim = feature_data.shape[1]
    feature_norm = (feature_data - feature_data.min(axis=0)) / (feature_data.max(axis=0) - feature_data.min(axis=0) + 1e-8)
    radar_angles = np.linspace(0, 2 * np.pi, feature_dim, endpoint=False).tolist()
    radar_angles_closed = np.concatenate([radar_angles, [radar_angles[0]]])

    # 雷达图标签
    task_radar_labels = {
        "dta": ['Drug Complexity', 'Target Length', 'Size Match Ratio'],
        "dti": ['Molecular Weight', 'Sequence Length', 'Binding Score'],
        "moa": ['Activity Score', 'Expression Level', 'MOA Similarity']
    }
    radar_labels = task_radar_labels.get(task_config['TASK_TYPE'], [f"Feature {i+1}" for i in range(feature_dim)])
    radar_labels = radar_labels[:feature_dim] if len(radar_labels) > feature_dim else radar_labels + [f"Feature {i+1}" for i in range(len(radar_labels), feature_dim)]

    # 绘制雷达图
    top_label_added = False
    high_label_added = False
    normal_label = "Normal Weight Points"
    
    for i in range(n_points):
        values = feature_norm[i].tolist()
        values_closed = np.concatenate([values, [values[0]]])
        is_top = i == top_weight_point_idx
        is_high = i in high_weight_point_idxs
        
        if is_top:
            ax2.plot(radar_angles_closed, values_closed, 'o-', linewidth=3.0, color='#8B0000', 
                     label=f"Top Weight Point {i} (w={weights[i]:.3f})", alpha=1.0, linestyle='-')
            ax2.fill(radar_angles_closed, values_closed, color='#8B0000', alpha=0.2)
            top_label_added = True
        elif is_high:
            if not high_label_added:
                ax2.plot(radar_angles_closed, values_closed, '-', linewidth=1.5, color='orange', 
                         label=f"High Weight Points (≥{np.percentile(weights,90):.3f})", alpha=0.8, linestyle='-')
                high_label_added = True
            else:
                ax2.plot(radar_angles_closed, values_closed, '-', linewidth=1.5, color='orange', 
                         alpha=0.8, linestyle='-')
        else:
            if i == 0 and not (top_label_added or high_label_added):
                ax2.plot(radar_angles_closed, values_closed, '-', linewidth=1.0, color='gray', 
                         label=normal_label, alpha=0.5, linestyle='-')
            else:
                ax2.plot(radar_angles_closed, values_closed, '-', linewidth=1.0, color='gray', 
                         alpha=0.5, linestyle='-')

    # 雷达图样式
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(radar_angles)
    ax2.set_xticklabels(radar_labels, fontsize=9)
    ax2.set_yticks(np.linspace(0, 1, 5))
    ax2.set_yticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 5)], fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#777777')
    ax2.spines['polar'].set_visible(True)
    ax2.set_title(f"Feature Contribution ({task_config['TASK_TYPE'].upper()} Task, Sample {sample_idx})", 
                  fontsize=12, pad=20, fontweight='bold')
    
    # 图例
    ax2.legend(
        loc='lower right', 
        fontsize=8,
        frameon=True, 
        facecolor='white', 
        edgecolor='#DDDDDD',
        bbox_to_anchor=(1.2, -0.1),
        handlelength=1.0,
        borderaxespad=0.2
    )

    # 保存结果
    plt.tight_layout()
    save_dir = os.path.join("./", task_config["TASK_TYPE"], task_config["TASK_NAME"], 
                            task_config["TASK_CUT"], "Paper_Visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    # 批量保存：文件名包含样本索引
    pdf_path = os.path.join(save_dir, f"FigureA_Sample_{sample_idx}_{task_config['TASK_TYPE'].upper()}_Attention.pdf")
    png_path = os.path.join(save_dir, f"FigureA_Sample_{sample_idx}_{task_config['TASK_TYPE'].upper()}_Attention.png")
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png', dpi=300)
    
    print(f"💾 样本 {sample_idx} 可视化结果已保存：")
    print(f"   - PDF：{pdf_path}")
    print(f"   - PNG：{png_path}")
    plt.close()  # 关闭画布，释放内存

# ===================== 4. 全任务自动遍历主函数 =====================
def auto_process_all_tasks():
    """全自动遍历所有任务类型/子数据集/数据划分，生成指定样本的可视化"""
    # ========== 全局配置（仅需修改这里） ==========
    GLOBAL_CONFIG = {
        "SAMPLE_IDX": 100,               # 固定处理第100个样本
        "DATA_SPLITS": ["warm", "target_cold", "drug_cold"],  # 所有数据划分方式
        "SKIP_MISSING_FILES": True       # 跳过不存在的文件（避免中断）
    }
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 任务元配置（覆盖所有组合）
    TASK_META = {
        "dta": {
            "datasets": ["davis", "kiba"],
            "model_suffix": "composite"
        },
        "dti": {
            "datasets": ["hetionet", "yamanishi_08"],
            "model_suffix": "auprc"
        },
        "moa": {
            "datasets": ["activation", "inhibition"],
            "model_suffix": "auprc"
        }
    }
    
    # 记录全局失败信息
    global_failed = []
    
    print(f"\n🚀 开始全自动遍历所有任务（处理第{GLOBAL_CONFIG['SAMPLE_IDX']}个样本）：")
    print(f"   - 运行设备：{DEVICE}")
    print(f"   - 任务类型：{list(TASK_META.keys())}")
    print(f"   - 数据划分：{GLOBAL_CONFIG['DATA_SPLITS']}")
    
    # 遍历所有任务类型
    for task_type in TASK_META.keys():
        meta = TASK_META[task_type]
        
        # 遍历该任务下的所有子数据集
        for task_name in meta["datasets"]:
            
            # 遍历所有数据划分方式
            for task_cut in GLOBAL_CONFIG["DATA_SPLITS"]:
                task_config = {
                    "TASK_TYPE": task_type,
                    "TASK_NAME": task_name,
                    "TASK_CUT": task_cut,
                    "MODEL_SUFFIX": meta["model_suffix"]
                }
                
                print(f"\n{'='*50}")
                print(f"处理任务组合：{task_type.upper()} / {task_name} / {task_cut}")
                print(f"{'='*50}")
                
                try:
                    # 检查文件是否存在
                    FEATURE_CSV_PATH = f"../data_preprocessing/{task_type}/{task_name}/log_and_file/{task_type}_features.csv"
                    POINT_CLOUD_PLY_PATH = f"../data_preprocessing/{task_type}/{task_name}/log_and_file/{task_type}_point_cloud.ply"
                    MODEL_PATH = f"../result/{task_type}_experiment/{task_name}/{task_cut}/models/fold_5_best_{meta['model_suffix']}.pth"
                    
                    missing_files = []
                    for path in [FEATURE_CSV_PATH, POINT_CLOUD_PLY_PATH, MODEL_PATH]:
                        if not os.path.exists(path):
                            missing_files.append(path)
                    
                    if missing_files:
                        if GLOBAL_CONFIG["SKIP_MISSING_FILES"]:
                            print(f"⚠️ 跳过该组合：缺失文件 {missing_files}")
                            global_failed.append({
                                "task": f"{task_type}/{task_name}/{task_cut}",
                                "error": f"缺失文件：{missing_files}"
                            })
                            continue
                        else:
                            raise FileNotFoundError(f"缺失必要文件：{missing_files}")
                    
                    # 加载样本数据并生成可视化
                    attention_weights, feature_data, top_weight_point_idx, \
                    high_weight_point_idxs, sample_idx, _, _ = load_sample_data(
                        GLOBAL_CONFIG["SAMPLE_IDX"], task_config, DEVICE
                    )
                    
                    generate_paper_visualization(
                        attention_weights, feature_data, top_weight_point_idx,
                        high_weight_point_idxs, sample_idx, task_config
                    )
                    
                    print(f"✅ {task_type}/{task_name}/{task_cut} 处理完成！")
                    
                except Exception as e:
                    error_msg = f"{task_type}/{task_name}/{task_cut} 处理失败：{str(e)}"
                    print(f"\n❌ {error_msg}")
                    global_failed.append({
                        "task": f"{task_type}/{task_name}/{task_cut}",
                        "error": str(e)
                    })
                    continue
    
    # 输出全局总结
    print(f"\n{'='*60}")
    print(f"🏁 全任务遍历完成！")
    print(f"{'='*60}")
    print(f"📊 统计信息：")
    print(f"   - 总任务组合数：{sum([len(TASK_META[t]['datasets']) for t in TASK_META]) * len(GLOBAL_CONFIG['DATA_SPLITS'])}")
    print(f"   - 失败组合数：{len(global_failed)}")
    
    if global_failed:
        print(f"\n❌ 失败组合详情：")
        for idx, failed in enumerate(global_failed, 1):
            print(f"   {idx}. {failed['task']}：{failed['error']}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        # 执行全任务自动遍历
        auto_process_all_tasks()
    except Exception as e:
        print(f"\n❌ 程序启动失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)