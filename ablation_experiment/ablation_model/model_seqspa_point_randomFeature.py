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
    def __init__(self, dim, init_values=1e-6):
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
            nn.Linear(out_dim, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, out_dim)
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
    def __init__(self, embed_dim, num_heads=1, dropout=0.85):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
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
        B, Nk, C = key_value.shape

        if C != self.embed_dim:
            key_value = torch.nn.functional.adaptive_avg_pool1d(key_value.transpose(1,2), self.embed_dim).transpose(1,2)
        
        q = self.q_proj(self.norm_q(query))
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_flat = key_value.reshape(B*Nk, -1)
        kv_norm = self.norm_kv(kv_flat).reshape(B, Nk, self.embed_dim)
        k = self.k_proj(kv_norm)
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        
        v = self.v_proj(kv_norm)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Nq, self.embed_dim)
        out = self.out_proj(out)
        out = self.norm_out(out + query)
        return out


class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, k=3, dropout_rate=0.8):
        super().__init__()
        self.k = k
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )
        self.layer1 = EnhancedEdgeConv(in_channels, 8, k=k, dropout=dropout_rate)
        self.layer2 = EnhancedEdgeConv(8, 16, k=k, dropout=dropout_rate)
        self.layer3 = EnhancedEdgeConv(16, out_channels, k=k, dropout=dropout_rate)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 4),
            nn.ReLU(),
            nn.Linear(4, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, out_channels),
            nn.Sigmoid()
        )
        self.proj_global = nn.Linear(out_channels, out_channels)
        self.norm_global = nn.LayerNorm(out_channels)
        self.drop_path = DropPath(0.7)
        self.layer_scale = LayerScale(out_channels, 1e-6)
        self.proj_point = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        B, N, _ = x.shape
        pos_enc = self.pos_encoder(x[:, :, :3])
        x_with_pos = x + pos_enc

        x1 = self.layer1(x_with_pos)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        global_pooled = x3.max(dim=1)[0]
        se_weights = self.se(global_pooled.unsqueeze(-1))
        global_feat = global_pooled * se_weights
        global_feat = self.proj_global(global_feat)
        global_feat = self.norm_global(global_feat)
        global_feat = self.layer_scale(self.drop_path(global_feat))

        point_feat = self.proj_point(x3)
        return global_feat, point_feat


class SeqSpaPoint_randomFeature(nn.Module):
    def __init__(
        self,
        drug_dim=384,
        target_dim=1280,
        point_out_dim=8,
        fused_dim=512,
        k=10,
        dropout_rate=0.7,
        num_queries=3,
    ):
        super().__init__()
        self.num_queries = num_queries
    
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim // 2, fused_dim // 4),
            nn.LayerNorm(fused_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim // 4, fused_dim // 2)
        )
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim // 2, fused_dim // 4),
            nn.LayerNorm(fused_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim // 4, fused_dim // 2)
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
    
        self.drug_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=1, dropout=dropout_rate)
        self.target_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=1, dropout=dropout_rate)
        self.joint_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=1, dropout=dropout_rate)
    
        proj_dim = fused_dim // 2
        seq_interaction_dim = proj_dim * 5
        fusion_input_dim = point_out_dim * 4 + seq_interaction_dim
    
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        self.fusion_proj = nn.Linear(fusion_input_dim, fusion_input_dim // 2)
        self.fusion_proj2 = nn.Linear(fusion_input_dim // 2, fusion_input_dim)  
    
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256 // 2),
            nn.LayerNorm(256 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256 // 2, 256 // 4),
            nn.LayerNorm(256 // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256 // 4, 128 // 2),
            nn.LayerNorm(128 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128 // 2, 128 // 4),
            nn.LayerNorm(128 // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128 // 4, 128)
        )
    
        self.pred_head = nn.Linear(128, 1)
        nn.init.zeros_(self.pred_head.bias)

    def forward(self, point_cloud, drug_emb, target_emb):
        B, N, C = point_cloud.shape
        
        point_cloud_input = point_cloud[:, :, :3].clone()
        
        point_global, point_wise = self.pointnet(point_cloud_input)

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
        concat = torch.cat([drug_p, target_p], dim=1)
        mult = drug_p * target_p
        diff = torch.abs(drug_p - target_p)
        seq_feat = torch.cat([concat, mult, diff, drug_p + target_p], dim=1)

        combined = torch.cat([enhanced_point_feat, seq_feat], dim=1)
        fused = self.fusion_norm(combined)
        fused = self.fusion_proj(fused)
        fused = self.fusion_proj2(fused)
        x = self.shared_mlp(fused)

        pred = self.pred_head(x).squeeze(-1)
        return pred