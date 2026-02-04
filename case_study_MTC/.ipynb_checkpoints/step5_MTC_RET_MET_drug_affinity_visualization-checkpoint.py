import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ====================== 1. 基础设置 ======================
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ====================== 2. 读取数据并转换为结合强度 ======================
df = pd.read_csv('./mtc_prediction_results/mtc_dta_dti_predictions.csv')
# 转换亲和力为结合强度（避免除以0，加1e-6）
df['true_binding_strength'] = 1000 / (df['true_affinity'] + 1e-6)
df['pred_binding_strength'] = 1000 / (df['pred_affinity'] + 1e-6)

# ====================== 3. 核心可视化（结合强度版） ======================
fig, axes = plt.subplots(2, 2, figsize=(24, 18))
fig.suptitle('MTC RET/MET Dual-Target Drug Binding Strength Prediction Analysis', 
             fontsize=20, fontweight='bold', y=1.02)

# 配色 & 靶点列表
colors_ret = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
colors_met = ['#9B59B6', '#1ABC9C', '#E67E22']
targets_ret = ['RET', 'RET(M918T)', 'RET(V804L)', 'RET(V804M)']
targets_met = ['MET', 'MET(M1250T)', 'MET(Y1235D)']
all_targets = targets_ret + targets_met
all_colors = colors_ret + colors_met

# --------------------- 子图1：所有靶点结合强度分布（左上） ---------------------
ax1 = axes[0, 0]
box_data = [df[df['target_id'] == t]['pred_binding_strength'].values for t in all_targets]
bp1 = ax1.boxplot(box_data, tick_labels=all_targets, patch_artist=True, vert=False)
for patch, color in zip(bp1['boxes'], all_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1)
# 标注RET(M918T)均值
ax1.axvline(df[df['target_id'] == 'RET(M918T)']['pred_binding_strength'].mean(), 
            color='red', linestyle='--', linewidth=2, alpha=0.8, label='RET(M918T) Mean')
ax1.set_xlabel('Predicted Binding Strength', fontsize=14, fontweight='bold')
ax1.set_title('Predicted Binding Strength Distribution of All Targets', fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=12)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.tick_params(axis='y', labelsize=11)

# --------------------- 子图2：RET家族真实vs预测结合强度（右上） ---------------------
ax2 = axes[0, 1]
for target, color in zip(targets_ret, colors_ret):
    filtered = df[(df['target_id'] == target) & (df['true_binding_strength'] < 1000)]
    ax2.scatter(filtered['true_binding_strength'], filtered['pred_binding_strength'], 
                color=color, alpha=0.7, s=60, label=target, edgecolors='black', linewidth=0.5)
ax2.set_xlabel('True Binding Strength', fontsize=14, fontweight='bold')
ax2.set_ylabel('Predicted Binding Strength', fontsize=14, fontweight='bold')
ax2.set_title('RET Family: True vs Predicted Binding Strength', fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=12, loc='best')
ax2.grid(alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=11)

# --------------------- 子图3：MET家族真实vs预测结合强度（左下） ---------------------
ax3 = axes[1, 0]
for target, color in zip(targets_met, colors_met):
    filtered = df[(df['target_id'] == target) & (df['true_binding_strength'] < 1000)]
    ax3.scatter(filtered['true_binding_strength'], filtered['pred_binding_strength'], 
                color=color, alpha=0.7, s=60, label=target, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('True Binding Strength', fontsize=14, fontweight='bold')
ax3.set_ylabel('Predicted Binding Strength', fontsize=14, fontweight='bold')
ax3.set_title('MET Family: True vs Predicted Binding Strength (nM → Binding Strength)', 
              fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=12, loc='best')
ax3.grid(alpha=0.3, linestyle='--')
ax3.tick_params(labelsize=11)

# --------------------- 子图4：RET/MET独立TOP8药物（右下） ---------------------
ax4 = axes[1, 1]
# 筛选RET(M918T)的TOP8药物（按结合强度从高到低）
ret_top8 = df[df['target_id'] == 'RET(M918T)'].sort_values('true_binding_strength', ascending=False).head(8).reset_index(drop=True)
ret_cids = [f"CID:{id}" for id in ret_top8['drug_id']]
ret_vals = ret_top8['true_binding_strength'].values

# 筛选MET的TOP8药物（按结合强度从高到低）
met_top8 = df[df['target_id'] == 'MET'].sort_values('true_binding_strength', ascending=False).head(8).reset_index(drop=True)
met_cids = [f"CID:{id}" for id in met_top8['drug_id']]
met_vals = met_top8['true_binding_strength'].values

# 绘制左右分栏的独立柱状图
x = np.arange(8)
width = 0.35

# 左侧：RET(M918T) TOP8
bars_ret = ax4.bar(x - width/2, ret_vals, width, label='RET(M918T)', 
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.2)
# 右侧：MET TOP8
bars_met = ax4.bar(x + width/2, met_vals, width, label='MET', 
                   color='#9B59B6', alpha=0.8, edgecolor='black', linewidth=1.2)

# 调整标签和刻度
ax4.set_xticks(x)
ax4.set_xticklabels(ret_cids, rotation=45, ha='right', fontsize=10)
ax4.set_ylim(0, max(max(ret_vals), max(met_vals)) * 1.2)
ax4.set_xlabel('Drug CID (Complete)', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Binding Strength', fontsize=14, fontweight='bold')
ax4.set_title('Top 8 High-Binding-Strength Drugs: RET(M918T) & MET', fontsize=16, fontweight='bold', pad=20)
ax4.legend(fontsize=12)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.tick_params(labelsize=11)

# 标注数值
for bar in bars_ret:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars_met:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ====================== 4. 保存图表 ======================
plt.tight_layout()
plt.savefig('./mtc_result_analysis/MTC_Two-Target_Binding_Strength.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 输出结合强度对照表
drug_table = pd.DataFrame({
    'RET(M918T)-Rank': range(1, 9),
    'RET(M918T)-CID': ret_top8['drug_id'].values,
    'RET-Binding Strength': ret_top8['true_binding_strength'].values,
    'MET-Rank': range(1, 9),
    'MET-CID': met_top8['drug_id'].values,
    'MET-Binding Strength': met_top8['true_binding_strength'].values
})
drug_table.to_csv('./mtc_result_analysis/RET_MET_Binding_Strength_TOP8.csv', index=False, encoding='utf-8-sig')

print("✅ 结合强度版图表已生成：MTC_Two-Target_Binding_Strength.png")
print("✅ 结合强度对照表已保存：RET_MET_Binding_Strength_TOP8.csv")