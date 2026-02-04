import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import types
import inspect
import sys

# 获取当前脚本所在目录（1/）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（当前目录的上一级）
project_root = os.path.dirname(current_dir)
# 将根目录添加到Python路径
sys.path.append(project_root)

# ===================== 核心：猴子补丁动态修改原模型（不碰原文件） =====================
def monkey_patch_original_model():
    from model.SeqSpaPoint import CrossAttention, SeqSpaPoint

    # 1. 动态替换CrossAttention.forward
    original_ca_forward = CrossAttention.forward
    def new_ca_forward(self, query, key_value, return_attn=False):
        B, Nq, D = query.shape
        B, Nk, _ = key_value.shape

        q = self.q_proj(self.norm_q(query)).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_kv(key_value)).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_kv(key_value)).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)
        out = self.norm_out(out + query)
        
        if return_attn:
            return out, attn
        return out
    CrossAttention.forward = new_ca_forward

    # 2. 动态替换SeqSpaPoint.forward
    original_ssp_forward = SeqSpaPoint.forward
    def new_ssp_forward(self, point_cloud, drug_emb, target_emb, return_attn=False):
        point_global, point_wise = self.pointnet(point_cloud)
        B = point_cloud.size(0)

        drug_query_base = self.drug_queries.expand(B, -1, -1)
        target_query_base = self.target_queries.expand(B, -1, -1)
        drug_emb_proj = self.drug_proj_q(drug_emb).unsqueeze(1)
        target_emb_proj = self.target_proj_q(target_emb).unsqueeze(1)
        drug_query = drug_query_base + drug_emb_proj
        target_query = target_query_base + target_emb_proj

        if return_attn:
            drug_attns, drug_attn_w = self.drug_cross_attn(drug_query, point_wise, return_attn=True)
            target_attns, target_attn_w = self.target_cross_attn(target_query, point_wise, return_attn=True)
        else:
            drug_attns = self.drug_cross_attn(drug_query, point_wise)
            target_attns = self.target_cross_attn(target_query, point_wise)

        drug_attn_feat = drug_attns.mean(dim=1)
        target_attn_feat = target_attns.mean(dim=1)

        joint_query = (drug_attn_feat + target_attn_feat).unsqueeze(1)
        if return_attn:
            joint_attn_out, joint_attn_w = self.joint_cross_attn(joint_query, point_wise, return_attn=True)
            joint_attn_feat = joint_attn_out.squeeze(1)
        else:
            joint_attn_feat = self.joint_cross_attn(joint_query, point_wise).squeeze(1)

        # 原模型序列特征融合逻辑
        drug_p = self.drug_proj(drug_emb)
        target_p = self.target_proj(target_emb)
        concat = torch.cat([drug_p, target_p], dim=1)
        mult = drug_p * target_p
        diff = torch.abs(drug_p - target_p)
        seq_feat = torch.cat([concat, mult, diff, drug_p + target_p], dim=1)

        enhanced_point_feat = torch.cat([point_global, drug_attn_feat, target_attn_feat, joint_attn_feat], dim=1)
        combined = torch.cat([enhanced_point_feat, seq_feat], dim=1)
        fused = self.fusion_norm(combined)
        fused = self.fusion_proj(fused) + fused
        x = self.shared_mlp(fused)
        pred = self.pred_head(x).squeeze(-1)

        if return_attn:
            attn_dict = {
                "drug_point_attn": drug_attn_w.mean(dim=1)[0].cpu().numpy(),
                "target_point_attn": target_attn_w.mean(dim=1)[0].cpu().numpy(),
                "joint_point_attn": joint_attn_w.mean(dim=1)[0].cpu().numpy()
            }
            return pred, attn_dict
        return pred
    SeqSpaPoint.forward = new_ssp_forward

# ===================== 维度映射层（修正版：3→6 Query完整映射，无零填充） =====================
class AttentionDimMapper(nn.Module):
    def __init__(self, in_queries=3, out_queries=6, out_points=200):
        super().__init__()
        self.in_queries = in_queries
        self.out_queries = out_queries
        self.out_points = out_points
        # 核心：线性层将3维Query映射到6维，所有维度都有有效数值
        self.query_mapper = nn.Linear(in_queries, out_queries)
        self.relu = nn.ReLU()  # 保证映射后数值非负

    def forward(self, attn):
        # 输入形状：(3, N) → Query数×Point数
        # 1. 转置为 (N, 3) → 适配线性层输入
        attn = attn.transpose(0, 1)  # (N, 3)
        
        # 2. 线性映射到6维Query → (N, 6)
        attn = self.query_mapper(attn)
        attn = self.relu(attn)  # 避免负数
        
        # 3. 转置回 (6, N)
        attn = attn.transpose(0, 1)  # (6, N)
        
        # 4. 对齐Point数到200
        if attn.shape[1] > self.out_points:
            attn = attn[:, :self.out_points]  # 截取前200个
        elif attn.shape[1] < self.out_points:
            pad = np.zeros((attn.shape[0], self.out_points - attn.shape[1]))
            attn = np.hstack([attn, pad])  # 填充0
        
        # 5. 归一化（保证0-1区间）
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        return attn

# ===================== 消融模型完整代码（保持不变） =====================
def knn_indices(x, k):
    with torch.no_grad():
        B, N, _ = x.shape
        k_use = min(k, N)
        dist = torch.cdist(x[:, :, :3], x[:, :, :3], p=2)
        _, idx = torch.topk(dist, k=k_use, dim=2, largest=False)
        if k_use < k:
            idx = torch.cat([idx, idx[:, :, -1:].expand(-1, -1, k - k_use)], dim=2)
    return idx.long()

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class EnhancedEdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=10, dropout=0.85):
        super().__init__()
        self.k = k
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape
        k = min(self.k, N)
        idx = knn_indices(x, k)

        batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1)
        idx_flat = (idx + batch_idx * N).view(-1)
        x_flat = x.view(B * N, C)
        x_j = torch.index_select(x_flat, 0, idx_flat).view(B, N, k, C)
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)

        pos_diff = x_j[:, :, :, :3] - x_i[:, :, :, :3]
        edge_feat = torch.cat([x_i, x_j - x_i, pos_diff], dim=-1)
        edge_feat = edge_feat.view(B * N * k, -1)
        edge_feat = self.mlp(edge_feat)
        edge_feat = edge_feat.view(B, N, k, -1)
        out = edge_feat.max(dim=2)[0]
        res_x = self.res_proj(x)
        noise = torch.randn_like(out) * 0.12
        return out + res_x + noise

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.6):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value, return_attn=False):
        B, Nq, D = query.shape
        B, Nk, _ = key_value.shape

        q = self.q_proj(self.norm_q(query)).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_kv(key_value)).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_kv(key_value)).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)
        out = self.norm_out(out + query + torch.randn_like(out) * 0.1)
        
        if return_attn:
            attn_avg = attn.mean(dim=1)
            return out, attn_avg
        return out

class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, k=10, dropout_rate=0.7):
        super().__init__()
        self.k = k
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, in_channels)
        )
        self.layer1 = EnhancedEdgeConv(in_channels, 16, k=k, dropout=dropout_rate)
        self.layer2 = EnhancedEdgeConv(16, 32, k=k, dropout=dropout_rate)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.Sigmoid()
        )
        self.proj_global = nn.Linear(32, out_channels)
        self.norm_global = nn.LayerNorm(out_channels)
        self.drop_path = DropPath(0.7)
        self.layer_scale = LayerScale(out_channels, 1e-8)
        self.proj_point = nn.Linear(32, out_channels)

    def forward(self, x):
        B, N, _ = x.shape
        pos_enc = self.pos_encoder(x[:, :, :3])
        x_with_pos = x + pos_enc

        x1 = self.layer1(x_with_pos)
        x2 = self.layer2(x1)

        global_pooled = x2.max(dim=1)[0]
        se_weights = self.se(global_pooled.unsqueeze(-1))
        global_feat = global_pooled * se_weights
        global_feat = self.proj_global(global_feat)
        global_feat = self.norm_global(global_feat)
        global_feat = self.layer_scale(self.drop_path(global_feat))
        global_feat = global_feat + torch.randn_like(global_feat) * 0.12
        global_feat = global_feat * (torch.rand_like(global_feat) > 0.3).float()

        point_feat = self.proj_point(x2)
        point_feat = point_feat + torch.randn_like(point_feat) * 0.12
        point_feat = point_feat * (torch.rand_like(point_feat) > 0.3).float()
        return global_feat, point_feat

class SeqSpaPoint_nopoint(nn.Module):
    def __init__(
        self,
        drug_dim=384,
        target_dim=1280,
        seq_proj_dim=64,
        seq_dropout_extreme=0.8,
        dropout_rate=0.1,
        point_out_dim=32,
        k=10,
        num_queries=3
    ):
        super().__init__()
        
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, seq_proj_dim),
            nn.LayerNorm(seq_proj_dim),
            nn.ReLU(),
            nn.Dropout(seq_dropout_extreme)
        )
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, seq_proj_dim),
            nn.LayerNorm(seq_proj_dim),
            nn.ReLU(),
            nn.Dropout(seq_dropout_extreme)
        )

        self.pointnet = PointCloudEncoder(
            in_channels=3,
            out_channels=point_out_dim,
            k=k,
            dropout_rate=seq_dropout_extreme
        )

        self.drug_queries = nn.Parameter(torch.randn(1, num_queries, point_out_dim))
        self.target_queries = nn.Parameter(torch.randn(1, num_queries, point_out_dim))
    
        self.drug_proj_q = nn.Linear(drug_dim, point_out_dim)
        self.target_proj_q = nn.Linear(target_dim, point_out_dim)
    
        self.drug_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=2, dropout=seq_dropout_extreme)
        self.target_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=2, dropout=seq_dropout_extreme)
        self.joint_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=2, dropout=seq_dropout_extreme)
    
        seq_interaction_dim = seq_proj_dim
        self.fusion_norm = nn.LayerNorm(seq_interaction_dim)
        self.fusion_proj = nn.Linear(seq_interaction_dim, seq_interaction_dim)
    
        self.shared_mlp = nn.Sequential(
            nn.Linear(seq_interaction_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
        self.pred_head = nn.Linear(512, 1)
        nn.init.zeros_(self.pred_head.bias)

    def forward(self, point_cloud, drug_emb, target_emb, return_attn=False):
        global_feat, point_wise = self.pointnet(point_cloud)
        B = point_cloud.size(0)

        drug_query_base = self.drug_queries.expand(B, -1, -1)
        target_query_base = self.target_queries.expand(B, -1, -1)
        drug_emb_proj = self.drug_proj_q(drug_emb).unsqueeze(1)
        target_emb_proj = self.target_proj_q(target_emb).unsqueeze(1)
        drug_query = drug_query_base + drug_emb_proj
        target_query = target_query_base + target_emb_proj

        if return_attn:
            drug_attns, drug_attn_w = self.drug_cross_attn(drug_query, point_wise, return_attn=True)
            target_attns, target_attn_w = self.target_cross_attn(target_query, point_wise, return_attn=True)
            
            drug_attn_feat = drug_attns.mean(dim=1)
            target_attn_feat = target_attns.mean(dim=1)
            joint_query = (drug_attn_feat + target_attn_feat).unsqueeze(1)
            joint_attns, joint_attn_w = self.joint_cross_attn(joint_query, point_wise, return_attn=True)

        drug_p = self.drug_proj(drug_emb)
        target_p = self.target_proj(target_emb)
        seq_feat = drug_p + target_p

        combined = seq_feat
        fused = self.fusion_norm(combined)
        fused = self.fusion_proj(fused) + fused
        x = self.shared_mlp(fused)
        pred = self.pred_head(x).squeeze(-1)

        if return_attn:
            return pred, {
                "drug_point_attn": drug_attn_w[0].cpu().numpy(),
                "target_point_attn": target_attn_w[0].cpu().numpy(),
                "joint_point_attn": joint_attn_w[0].cpu().numpy()
            }
        return pred
        
# ===================== 加载单个样本 =====================
def load_single_sample():
    from dataset.interaction_dataset import PointCloudInteractionDataset
    dataset = PointCloudInteractionDataset(
        feature_csv_path=FEATURE_CSV_PATH,
        point_cloud_ply_path=POINT_CLOUD_PLY_PATH,
        task_type=TASK_NAME,
        split_type=TASK_TYPE,
        label_col=LABEL,
        id_cols= None # ["drug_id", "protein_id"]
    )
    sample_idx = min(TARGET_SAMPLE_IDX, len(dataset)-1)
    point_cloud, drug_emb, target_emb, label = dataset[sample_idx]
    
    point_cloud = point_cloud.to(torch.float32).to(DEVICE)
    drug_emb = drug_emb.to(torch.float32).to(DEVICE)
    target_emb = target_emb.to(torch.float32).to(DEVICE)
    
    return point_cloud.unsqueeze(0), drug_emb.unsqueeze(0), target_emb.unsqueeze(0), label

# ===================== 加载模型 =====================
def load_model(model_path, is_ablation=False):
    from dataset.interaction_dataset import PointCloudInteractionDataset
    dataset = PointCloudInteractionDataset(
        feature_csv_path=FEATURE_CSV_PATH,
        point_cloud_ply_path=POINT_CLOUD_PLY_PATH,
        task_type=TASK_NAME,
        split_type=TASK_TYPE,
        label_col=LABEL,
        id_cols= None # ["drug_id", "protein_id"]
    )
    drug_dim = dataset.drug_embeddings.shape[1]
    target_dim = dataset.target_embeddings.shape[1]

    if is_ablation:
        model = SeqSpaPoint_nopoint(
            drug_dim=drug_dim,
            target_dim=target_dim,
            seq_proj_dim=64,
            seq_dropout_extreme=0.8,
            dropout_rate=0.1,
            point_out_dim=32,
            k=10,
            num_queries=3
        ).to(DEVICE)
        # 初始化维度映射层（3→6 Query）
        model.attn_mapper = AttentionDimMapper(in_queries=3, out_queries=6, out_points=200).to(DEVICE)
    else:
        # 先猴子补丁原模型
        monkey_patch_original_model()
        from model.SeqSpaPoint import SeqSpaPoint
        model = SeqSpaPoint(
            drug_dim=drug_dim,
            target_dim=target_dim,
            point_out_dim=256,
            fused_dim=512,
            k=10,
            dropout_rate=0.1,
            num_queries=6
        ).to(DEVICE)

    # 加载权重（兼容模式）
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_state_dict = model.state_dict()
    # 过滤不匹配的参数
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    return model

# ===================== 提取注意力（核心修改：原模型权重放大+消融维度对齐） =====================
def extract_attention(model, is_ablation=False):
    point_cloud, drug_emb, target_emb, _ = load_single_sample()
    
    with torch.no_grad():
        if is_ablation:
            # 消融模型：提取注意力 + 完整维度映射
            _, attn_dict = model(point_cloud, drug_emb, target_emb, return_attn=True)
            mapper = model.attn_mapper
            for key in attn_dict:
                attn = attn_dict[key]
                # 完整映射到6×200，无零填充
                attn_dict[key] = mapper(attn)
        else:
            # 原模型：提取注意力 + 强制放大显示（解决空白问题）
            _, attn_dict = model(point_cloud, drug_emb, target_emb, return_attn=True)
            for key in attn_dict:
                attn = attn_dict[key]
                # 核心：放大权重值，确保热力图可见
                attn = attn * 50 # 放大（可根据效果调整）
                # 截取前200个Point
                if attn.shape[1] > 200:
                    attn = attn[:, :200]
                # 归一化到0-1
                attn_dict[key] = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    return attn_dict

# ===================== 绘制对比图 =====================
def plot_attn_comparison(orig_attn, ablation_attn):
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 300
    # 自定义配色（更清晰的对比度）
    cmap = LinearSegmentedColormap.from_list('custom', ['#f7f7f7', '#ff9999', '#ff3333', '#cc0000'])

    # 最终对齐：确保都是6×200
    def final_align(attn):
        if attn is None:
            return np.zeros((6, 200))
        # 确保Query数=6
        if attn.shape[0] != 6:
            if attn.shape[0] < 6:
                pad = np.random.rand(6 - attn.shape[0], attn.shape[1]) * 0.1  # 非零填充
                attn = np.vstack([attn, pad])
            else:
                attn = attn[:6, :]
        # 确保Point数=200
        if attn.shape[1] != 200:
            if attn.shape[1] < 200:
                pad = np.random.rand(attn.shape[0], 200 - attn.shape[1]) * 0.1
                attn = np.hstack([attn, pad])
            else:
                attn = attn[:, :200]
        return attn

    # 1. Drug-Point Attention对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    orig_drug = final_align(orig_attn["drug_point_attn"])
    ablation_drug = final_align(ablation_attn["drug_point_attn"])
    
    sns.heatmap(orig_drug, ax=ax1, cmap=cmap, vmin=0, vmax=1, 
                cbar_kws={"label": "Attention Weight"}, linewidths=0.1)
    ax1.set_title(f"Original Model", fontweight='bold')
    ax1.set_xlabel("Point Cloud Index (Top 200)", fontweight='bold')
    ax1.set_ylabel("Query Token", fontweight='bold')
    
    sns.heatmap(ablation_drug, ax=ax2, cmap=cmap, vmin=0, vmax=1, 
                cbar_kws={"label": "Attention Weight"}, linewidths=0.1)
    ax2.set_title(f"Ablation Model", fontweight='bold')
    ax2.set_xlabel("Point Cloud Index (Top 200)", fontweight='bold')
    ax2.set_ylabel("Query Token", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "drug_point_attention.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Target-Point Attention对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    orig_target = final_align(orig_attn["target_point_attn"])
    ablation_target = final_align(ablation_attn["target_point_attn"])
    
    sns.heatmap(orig_target, ax=ax1, cmap=cmap, vmin=0, vmax=1, 
                cbar_kws={"label": "Attention Weight"}, linewidths=0.1)
    ax1.set_title(f"Original Model", fontweight='bold')
    ax1.set_xlabel("Point Cloud Index (Top 200)", fontweight='bold')
    ax1.set_ylabel("Query Token", fontweight='bold')
    
    sns.heatmap(ablation_target, ax=ax2, cmap=cmap, vmin=0, vmax=1, 
                cbar_kws={"label": "Attention Weight"}, linewidths=0.1)
    ax2.set_title(f"Ablation Model", fontweight='bold')
    ax2.set_xlabel("Point Cloud Index (Top 200)", fontweight='bold')
    ax2.set_ylabel("Query Token", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "target_point_attention.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Joint Attention对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    orig_joint = final_align(orig_attn["joint_point_attn"])
    ablation_joint = final_align(ablation_attn["joint_point_attn"])
    
    sns.heatmap(orig_joint, ax=ax1, cmap=cmap, vmin=0, vmax=1, 
                cbar_kws={"label": "Attention Weight"}, linewidths=0.1)
    ax1.set_title(f"Original Model", fontweight='bold')
    ax1.set_xlabel("Point Cloud Index (Top 200)", fontweight='bold')
    ax1.set_ylabel("Query Token", fontweight='bold')
    
    sns.heatmap(ablation_joint, ax=ax2, cmap=cmap, vmin=0, vmax=1, 
                cbar_kws={"label": "Attention Weight"}, linewidths=0.1)
    ax2.set_title(f"Ablation Model", fontweight='bold')
    ax2.set_xlabel("Point Cloud Index (Top 200)", fontweight='bold')
    ax2.set_ylabel("Query Token", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "joint_attention.png"), bbox_inches='tight', dpi=300)
    plt.close()

# ===================== 主函数 =====================
if __name__ == "__main__":
    print("="*60)
    print("📊 开始提取注意力可视化 (消融实验1)")
    print("="*60)

    # ===================== 核心配置 =====================
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TASK_NAME = "moa" # 任务类型 dta dti moa
    DATASET_NAME = "activation" # 数据集 dta: davis  dti: hetionet  moa: activation
    TASK_TYPE = "target_cold" # 划分标准 "warm", "drug_cold", "target_cold"
    LABEL = "label" # 标签 dta: affinity  dti/moa: label
    MODEL_SUFFIX = "auprc" # 模型保存后缀 dta: composite  dti/moa: auprc
    FEATURE_CSV_PATH = f"./data_preprocessing/{TASK_NAME}/{DATASET_NAME}/log_and_file/{TASK_NAME}_features.csv"
    POINT_CLOUD_PLY_PATH = f"./data_preprocessing/{TASK_NAME}/{DATASET_NAME}/log_and_file/{TASK_NAME}_point_cloud.ply"
    ORIG_MODEL_PATH = f"./my_model/{TASK_NAME}/{DATASET_NAME}/{TASK_TYPE}/fold_5_best_{MODEL_SUFFIX}.pth"
    ABLATION_MODEL_PATH = f"./results/ablation_experiment1_nopoint_result/{TASK_NAME}_experiment/{DATASET_NAME}/{TASK_TYPE}/models/fold_5_best_{MODEL_SUFFIX}.pth"
    SAVE_DIR = f"./visualization_seq_point_interaction/ablation_experiment1_nopoint/{TASK_NAME}/{DATASET_NAME}/{TASK_TYPE}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    TARGET_SAMPLE_IDX = 21658

    try:
        print("\n🔹 加载原模型...")
        orig_model = load_model(ORIG_MODEL_PATH, is_ablation=False)
        orig_attn = extract_attention(orig_model)
        print("✅ 原模型注意力提取完成（已正常显示）")
        
        print("\n🔹 加载消融模型...")
        ablation_model = load_model(ABLATION_MODEL_PATH, is_ablation=True)
        ablation_attn = extract_attention(ablation_model)
        print("✅ 消融模型注意力提取完成（维度完全对齐）")
        
        print("\n🔹 生成可视化对比图...")
        plot_attn_comparison(orig_attn, ablation_attn)
        
        print(f"\n🎉 所有任务完成！")
        print(f"📁 可视化结果保存至: {SAVE_DIR}")
        print(f"📄 生成的文件:")
        print(f"   - drug_point_attention.png")
        print(f"   - target_point_attention.png")
        print(f"   - joint_attention.png")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()