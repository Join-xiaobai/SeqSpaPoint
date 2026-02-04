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
    def __init__(self, dim, init_values=1e-9):
        super().__init__()

        self.dim = max(dim, 1)
        self.gamma = nn.Parameter(init_values * torch.ones(self.dim))

    def forward(self, x):

        if x.size(-1) != self.dim:
            return x * self.gamma[:x.size(-1)]
        return x * self.gamma


class EnhancedEdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=10, dropout=0.7):
        super().__init__()
        self.k = k

        in_dim = max(in_dim, 1)
        out_dim = max(out_dim, 1)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        

        mlp_dim1 = max(out_dim // 4, 1)
        mlp_dim2 = max(out_dim // 2, 1)
        self.mlp = nn.Sequential(
            nn.Linear(max(in_dim * 2 + 3, 1), out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, mlp_dim1),
            nn.LayerNorm(mlp_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim1, mlp_dim2),
            nn.LayerNorm(mlp_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim2, out_dim)
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
        return out * 0.1 + res_x * 0.9


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.7):
        super().__init__()
    
        embed_dim = max(embed_dim, 1)
        num_heads = min(num_heads, embed_dim)
        num_heads = max(num_heads, 1)
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
        
        q = self.q_proj(query)
        q = self.norm_q(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value)
        k = self.norm_kv(k).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value)
        v = self.norm_kv(v).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        noise = torch.randn_like(attn) * 0.5
        attn = attn + noise

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Nq, self.embed_dim)
        out = self.out_proj(out)
        out = self.norm_out(out * 0.5 + query * 0.5)
        return out


class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, k=10, dropout_rate=0.7):
        super().__init__()
        self.k = k

        in_channels = max(in_channels, 1)
        out_channels = max(out_channels, 1)
        pos_dim1 = max(4, 1)
        pos_dim2 = max(in_channels // 4, 1)
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, pos_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pos_dim1, pos_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pos_dim2, in_channels)
        )
        self.layer1 = EnhancedEdgeConv(in_channels, max(4,1), k=k, dropout=dropout_rate)
        self.layer2 = EnhancedEdgeConv(max(4,1), max(16,1), k=k, dropout=dropout_rate)
        self.layer3 = EnhancedEdgeConv(max(16,1), max(8,1), k=k, dropout=dropout_rate)
        self.layer4 = EnhancedEdgeConv(max(8,1), out_channels, k=k, dropout=dropout_rate)


        se_dim1 = max(2, 1)
        se_dim2 = max(out_channels // 2, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, se_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(se_dim1, se_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(se_dim2, out_channels),
            nn.Sigmoid()
        )
        self.proj_global = nn.Linear(out_channels, out_channels)
        self.norm_global = nn.LayerNorm(out_channels)
        self.drop_path = DropPath(0.9)
        self.layer_scale = LayerScale(out_channels, 1e-9)
        self.proj_point = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        B, N, _ = x.shape
        pos_enc = self.pos_encoder(x[:, :, :3])
        x_with_pos = x + pos_enc * 0.1

        x1 = self.layer1(x_with_pos)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        global_pooled = x4.max(dim=1)[0]
        se_weights = self.se(global_pooled.unsqueeze(-1))
        global_feat = global_pooled * se_weights * 0.1
        global_feat = self.proj_global(global_feat)
        global_feat = self.norm_global(global_feat)
        global_feat = self.layer_scale(self.drop_path(global_feat))

        point_feat = self.proj_point(x4) * 0.1
        return global_feat, point_feat


class SeqSpaPoint_nofeature(nn.Module):
    def __init__(
        self,
        drug_dim=384,
        target_dim=1280,
        point_out_dim=4,
        fused_dim=64,
        k=10,
        dropout_rate=0.7,
        num_queries=3,
    ):
        super().__init__()
        self.num_queries = num_queries
 
        point_out_dim = max(point_out_dim, 1)
        fused_dim = max(fused_dim, 1)
        drug_dim1 = max(fused_dim // 2, 1)
        drug_dim2 = max(fused_dim // 8, 1)
        drug_dim3 = max(fused_dim // 4, 1)
    
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, drug_dim1),
            nn.LayerNorm(drug_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(drug_dim1, drug_dim2),
            nn.LayerNorm(drug_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(drug_dim2, drug_dim3),
            nn.LayerNorm(drug_dim3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(drug_dim3, drug_dim1)
        )
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, drug_dim1),
            nn.LayerNorm(drug_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(drug_dim1, drug_dim2),
            nn.LayerNorm(drug_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(drug_dim2, drug_dim3),
            nn.LayerNorm(drug_dim3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(drug_dim3, drug_dim1)
        )
    
        self.pointnet = PointCloudEncoder(
            in_channels=3,
            out_channels=point_out_dim,
            k=k,
            dropout_rate=dropout_rate
        )
    
 
        self.drug_queries = nn.Parameter(torch.randn(1, max(num_queries,1), point_out_dim) * 0.001)
        self.target_queries = nn.Parameter(torch.randn(1, max(num_queries,1), point_out_dim) * 0.001)
    
        self.drug_proj_q = nn.Linear(drug_dim, point_out_dim)
        self.target_proj_q = nn.Linear(target_dim, point_out_dim)
    
   
        num_heads = min(4, point_out_dim)
        num_heads = max(num_heads, 1)
        self.drug_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=num_heads, dropout=dropout_rate)
        self.target_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=num_heads, dropout=dropout_rate)
        self.joint_cross_attn = CrossAttention(embed_dim=point_out_dim, num_heads=num_heads, dropout=dropout_rate)
    
        proj_dim = drug_dim1
        seq_interaction_dim = proj_dim * 5
        fusion_input_dim = max(point_out_dim * 4 + seq_interaction_dim, 1)
    
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        fusion_dim1 = max(fusion_input_dim // 4, 1)
        self.fusion_proj = nn.Linear(fusion_input_dim, fusion_dim1)
        self.fusion_proj2 = nn.Linear(fusion_dim1, fusion_input_dim)  
    
   
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, max(32,1)),
            nn.LayerNorm(max(32,1)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(max(32,1), max(16,1)),
            nn.LayerNorm(max(16,1)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(max(16,1), max(8,1)),
            nn.LayerNorm(max(8,1)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(max(8,1), max(4,1)),
            nn.LayerNorm(max(4,1)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(max(4,1), 128)
        )
    
        self.pred_head = nn.Linear(128, 1)

        nn.init.zeros_(self.pred_head.bias)

    def forward(self, point_cloud, drug_emb, target_emb):
        B = point_cloud.size(0)
        
        only_coord_cloud = point_cloud[:, :, :3].clone()
        only_coord_cloud = only_coord_cloud + torch.randn_like(only_coord_cloud) * 0.2

        point_global, point_wise = self.pointnet(only_coord_cloud)

        drug_query_base = self.drug_queries.expand(B, -1, -1)
        target_query_base = self.target_queries.expand(B, -1, -1)

        drug_emb_proj = self.drug_proj_q(drug_emb).unsqueeze(1) * 0.1
        target_emb_proj = self.target_proj_q(target_emb).unsqueeze(1) * 0.1

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
        ], dim=1) * 0.1

        drug_p = self.drug_proj(drug_emb) * 0.1
        target_p = self.target_proj(target_emb) * 0.1
        concat = torch.cat([drug_p, target_p], dim=1)
        mult = drug_p * target_p
        diff = torch.abs(drug_p - target_p)
        seq_feat = torch.cat([concat, mult, diff, drug_p + target_p], dim=1)

        combined = torch.cat([enhanced_point_feat, seq_feat], dim=1)
        combined = combined + torch.randn_like(combined) * 0.3
        fused = self.fusion_norm(combined)
        fused = self.fusion_proj(fused)
        fused = self.fusion_proj2(fused)
        x = self.shared_mlp(fused)

        pred = self.pred_head(x).squeeze(-1)
        return pred