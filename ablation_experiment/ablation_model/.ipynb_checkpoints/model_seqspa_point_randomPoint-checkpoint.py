import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class EnhancedEdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=10, dropout=0.6):
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
        return out + res_x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.6):
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

    def forward(self, query, key_value):
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
        return out


class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, k=10, dropout_rate=0.6):
        super().__init__()
        self.k = k
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels)
        )
        self.layer1 = EnhancedEdgeConv(in_channels, 64, k=k, dropout=dropout_rate)
        self.layer2 = EnhancedEdgeConv(64, 64, k=k, dropout=dropout_rate)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.Sigmoid()
        )
        self.proj_global = nn.Linear(64, out_channels)
        self.norm_global = nn.LayerNorm(out_channels)
        self.drop_path = DropPath(0.6)
        self.layer_scale = LayerScale(out_channels, 1e-4)
        self.proj_point = nn.Linear(64, out_channels)

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

        point_feat = self.proj_point(x2)
        return global_feat, point_feat


class SeqSpaPoint_randomPoint(nn.Module):
    def __init__(
        self,
        drug_dim=384,
        target_dim=1280,
        point_out_dim=64,
        fused_dim=512,
        k=10,
        dropout_rate=0.6,
        num_queries=6,
    ):
        super().__init__()
        self.num_queries = num_queries
    
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
        self.pointnet = PointCloudEncoder(
            in_channels=3,
            out_channels=point_out_dim,
            k=k,
            dropout_rate=dropout_rate
        )
    
        self.drug_queries = nn.Parameter(torch.randn(1, num_queries, point_out_dim))
        self.target_queries = nn.Parameter(torch.randn(1, num_queries, point_out_dim))
    
        self.drug_proj_q = nn.Linear(drug_dim, point_out_dim)
        self.target_proj_q = nn.Linear(target_dim, point_out_dim)
    
        self.drug_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=8, dropout=dropout_rate)
        self.target_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=8, dropout=dropout_rate)
        self.joint_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=8, dropout=dropout_rate)
    
        proj_dim = 64
        seq_interaction_dim = proj_dim
        fusion_input_dim = point_out_dim * 4 + seq_interaction_dim
    
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        self.fusion_proj = nn.Linear(fusion_input_dim, fusion_input_dim)
    
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
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

    def forward(self, point_cloud, drug_emb, target_emb):
        B, N, C = point_cloud.shape
        
        point_feat = point_cloud[:, :, 3:].clone() if C > 3 else torch.empty(B, N, 0, device=point_cloud.device)
        pure_noise_coord = torch.randn_like(point_cloud[:, :, :3])
        noise_point_cloud = torch.cat([pure_noise_coord, point_feat], dim=-1)

        point_global, point_wise = self.pointnet(noise_point_cloud)

        drug_query_base = self.drug_queries.expand(B, -1, -1)
        target_query_base = self.target_queries.expand(B, -1, -1)

        drug_emb_proj = self.drug_proj_q(drug_emb).unsqueeze(1)
        target_emb_proj = self.target_proj_q(target_emb).unsqueeze(1)

        drug_query = drug_query_base + drug_emb_proj
        target_query = target_query_base + target_emb_proj

        drug_attns = self.drug_cross_attn(drug_query, point_wise)
        target_attns = self.target_cross_attn(target_query, point_wise)

        drug_attn_feat = drug_attns.mean(dim=1)
        target_attn_feat = target_attns.mean(dim=1)

        joint_query = (drug_attn_feat + target_attn_feat).unsqueeze(1)
        joint_attn_feat = self.joint_cross_attn(joint_query, point_wise).squeeze(1)

        enhanced_point_feat = torch.cat([
            point_global,
            drug_attn_feat,
            target_attn_feat,
            joint_attn_feat
        ], dim=1)

        drug_p = self.drug_proj(drug_emb)
        target_p = self.target_proj(target_emb)
        seq_feat = drug_p + target_p

        combined = torch.cat([enhanced_point_feat, seq_feat], dim=1)
        fused = self.fusion_norm(combined)
        fused = self.fusion_proj(fused) + fused
        x = self.shared_mlp(fused)

        pred = self.pred_head(x).squeeze(-1)
        return pred