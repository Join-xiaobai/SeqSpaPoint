import open3d as o3d
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import itertools  # 用于生成所有 drug-protein 组合

class PointCloudInteractionDataset(Dataset):
    def __init__(
        self,
        feature_csv_path,
        point_cloud_ply_path,
        task_type="dta",  # "dta", "dti", "moa"
        split_type="warm",
        test_size=0.1,
        label_col=None,  # 回归/分类目标列名
        id_cols=None,    # [drug_id_col, target_id_col]
        standardize_embeddings=True,  # 是否对预训练嵌入做标准化（通常可设为 False）
        # 新增点云配置参数（解决硬编码、保证可复现、增强兼容性）
        point_cloud_num_points=10,  # 每样本最终点云点数（替代硬编码10）
        point_cloud_noise_std=0.01, # 补充点噪声标准差（替代硬编码0.01）
        point_cloud_random_seed=42, # 噪声随机种子（保证实验可复现）
    ):
        """
        通用药物-靶点交互数据集，支持 DTA（回归）、DTI（二分类）、MOA（多分类/二分类）。
        关键改进：
        1. 分离 drug_emb 与 target_emb，支持交互建模。
        2. 点云处理消除硬编码，可配置、可复现、保留原始信息。
        3. DTA 任务中标签 **不再归一化**，直接使用原始亲和力值（如 KIBA score ∈ [0, ~17]）。
        4. 新增标签缺失值处理，避免 NaN 转换报错。
        
        参数说明：
            feature_csv_path: 特征CSV路径（必须含 emb_drug_* 和 emb_target_* 列）
            point_cloud_ply_path: PLY点云路径
            task_type: "dta" | "dti" | "moa"
            split_type: "warm" | "drug_cold" | "target_cold"
            test_size: 冷启动中新实体比例
            label_col: 标签列名（如 "affinity", "label", "moa_label"）
            id_cols: [drug_id列名, target_id列名]，如 ["drug_id", "protein_id"]
            standardize_embeddings: 是否对 drug/target 嵌入做 StandardScaler（建议 False）
            point_cloud_num_points: 每样本目标点云点数（默认10）
            point_cloud_noise_std: 补充点的高斯噪声标准差（默认0.01）
            point_cloud_random_seed: 噪声随机种子（默认42，保证复现）
        """
        # ========== 第一步：初始化基础参数（仅赋值，不依赖df） ==========
        self.task_type = task_type.lower()
        self.split_type = split_type
        self.label_col = label_col
        self.id_cols = id_cols or self._get_default_id_cols()  # 先赋值默认ID列，暂不过滤
        self.standardize_embeddings = standardize_embeddings
        # 保存点云配置参数
        self.point_cloud_num_points = point_cloud_num_points
        self.point_cloud_noise_std = point_cloud_noise_std
        self.point_cloud_random_seed = point_cloud_random_seed
    
        # ========== 第二步：加载CSV（核心！先有df，才能用df） ==========
        print(f"【步骤1/8】加载特征CSV: {feature_csv_path}")
        self.df = pd.read_csv(feature_csv_path)
        print(f"✅ 加载成功，总样本数: {len(self.df)}")
    
        # ========== 关键修复1：处理标签列 NaN 和类型错误 ==========
        if self.label_col is not None and self.label_col in self.df.columns:
            # 1. 统计原始缺失值
            label_nan_count = self.df[self.label_col].isna().sum()
            print(f"【标签预处理】原始标签缺失值数量: {label_nan_count}")
            
            # 2. 删除标签缺失的行（最稳妥，避免填充引入偏差）
            self.df = self.df.dropna(subset=[self.label_col])
            print(f"✅ 删除标签缺失行后，剩余样本数: {len(self.df)}")
            
            # 3. 转换标签为数值类型（失败则设为 NaN，再删除）
            self.df[self.label_col] = pd.to_numeric(self.df[self.label_col], errors='coerce')
            post_convert_nan = self.df[self.label_col].isna().sum()
            if post_convert_nan > 0:
                self.df = self.df.dropna(subset=[self.label_col])
                print(f"✅ 删除标签转换失败行后，剩余样本数: {len(self.df)}")
            
            # 4. 按任务类型处理标签
            if self.task_type == "dta":
                self.labels_raw = self.df[self.label_col].values.astype(np.float32)
                print(f"【DTA任务】标签范围: [{self.labels_raw.min():.4f}, {self.labels_raw.max():.4f}]")
            elif self.task_type == "dti":
                self.labels_raw = self.df[self.label_col].values.astype(np.int64)
                print(f"【DTI任务】标签分布: {np.bincount(self.labels_raw)}")
            elif self.task_type == "moa":
                # 可选：若MOA是多分类，注释掉clip；若为二分类，保留clip
                # self.labels_raw = self.df[self.label_col].values.astype(np.int64)  # 多分类
                self.labels_raw = np.clip(self.df[self.label_col].values.astype(np.int64), 0, 1)  # 二分类
                print(f"【MOA任务】标签分布: {np.bincount(self.labels_raw)}")
        else:
            raise ValueError(f"❌ 标签列 {self.label_col} 不存在！")
    
        # ========== 第三步：过滤ID列（现在df已加载，可安全访问） ==========
        # 容错：只保留存在的列，避免列不存在报错
        self.id_cols = [col for col in self.id_cols if col in self.df.columns]
        if len(self.id_cols) < 2:
            raise ValueError(f"❌ 至少需要2个ID列（drug/target），但仅找到：{self.id_cols}")
    
        # ========== 步骤2：提取ID、分离嵌入、几何特征 ==========
        print("【步骤2/8】提取ID、分离 drug/target 嵌入、几何特征...")
        self.drug_ids = self.df[self.id_cols[0]].values
        self.target_ids = self.df[self.id_cols[1]].values
    
        # 分离 drug 和 target 嵌入
        drug_emb_cols = sorted([col for col in self.df.columns if col.startswith("emb_drug_")])
        target_emb_cols = sorted([col for col in self.df.columns if col.startswith("emb_target_")])

        if not drug_emb_cols or not target_emb_cols:
            raise ValueError(
                "❌ CSV 中缺少 emb_drug_* 或 emb_target_* 列！\n"
                "请确保预处理脚本分别保存了 ChemBERTa 和 ESM-2 的嵌入。"
            )

        self.drug_embeddings_raw = self.df[drug_emb_cols].values.astype(np.float32)      # (N, D_d)
        self.target_embeddings_raw = self.df[target_emb_cols].values.astype(np.float32)  # (N, D_t)
        # 嵌入维度非空校验
        if self.drug_embeddings_raw.shape[1] == 0 or self.target_embeddings_raw.shape[1] == 0:
            raise ValueError(
                f"❌ 嵌入维度为0！请检查特征文件是否包含 emb_drug_* / emb_target_* 列\n"
                f"Drug嵌入列数: {len(drug_emb_cols)}, Target嵌入列数: {len(target_emb_cols)}"
            )

        # ========== 关键修复2：删除几何特征列重复赋值 ==========
        # 几何特征（非ID、非标签、非嵌入、非gene_id）
        geo_cols = [
            col for col in self.df.columns
            if col not in self.id_cols + [self.label_col, "gene_id"]  # 排除 gene_id
               and not col.startswith("emb_drug_")
               and not col.startswith("emb_target_")
        ]
        self.geometric_features_raw = self.df[geo_cols].values.astype(np.float32) if geo_cols else np.zeros((len(self.df), 0), dtype=np.float32)

        print(f"  - Drug 嵌入维度: {self.drug_embeddings_raw.shape[1]}")
        print(f"  - Target 嵌入维度: {self.target_embeddings_raw.shape[1]}")
        print(f"  - 几何特征维度: {self.geometric_features_raw.shape[1]}")

        # === 步骤3：加载点云（修复硬编码、保证可复现、保留原始信息）===
        print("【步骤3/8】加载PLY点云并处理（可配置、可复现）...")
        pcd = o3d.io.read_point_cloud(point_cloud_ply_path)
        self.point_clouds = np.asarray(pcd.points).astype(np.float32)

        # 第一步：统一点云形状为 (N, num_points_per_sample, 3)
        if len(self.point_clouds.shape) == 2:
            # 原始形状 (N, 3) → 转换为单一点每样本 (N, 1, 3)
            self.point_clouds = self.point_clouds[:, np.newaxis, :]

        # 第二步：获取原始点数，处理目标点数（保留原始信息，不强行覆盖）
        num_points_per_sample_original = self.point_clouds.shape[1]
        target_num_points = self.point_cloud_num_points

        if num_points_per_sample_original < target_num_points:
            # 原始点数不足，补充带噪声的复制点（保留原始点在前）
            need_add_points = target_num_points - num_points_per_sample_original
            num_samples = len(self.point_clouds)

            # 固定随机种子，保证噪声可复现
            rng = np.random.RandomState(self.point_cloud_random_seed)

            # 生成噪声（形状：(N, need_add_points, 3)）
            noise = rng.normal(
                0,
                self.point_cloud_noise_std,
                (num_samples, need_add_points, 3)
            ).astype(np.float32)

            # 补充「原始最后一个点 + 噪声」，避免破坏原始点分布
            add_points = self.point_clouds[:, -1:, :].repeat(need_add_points, axis=1) + noise

            # 拼接原始点和补充点，形成最终点云
            self.point_clouds = np.concatenate([self.point_clouds, add_points], axis=1)
        elif num_points_per_sample_original > target_num_points:
            # 原始点数超过目标，截断到目标点数（保留前N个原始点）
            self.point_clouds = self.point_clouds[:, :target_num_points, :]

        # 第三步：点云与样本数对齐（兼容样本数不匹配场景）
        self.num_samples = len(self.df)  # 修复：用处理后的df长度作为样本数
        if len(self.point_clouds) != self.num_samples:
            print(f"⚠️ 点云数量({len(self.point_clouds)}) ≠ 样本数({self.num_samples})，自动对齐")
            if len(self.point_clouds) > self.num_samples:
                self.point_clouds = self.point_clouds[:self.num_samples]
            else:
                pad_shape = (self.num_samples - len(self.point_clouds), self.point_clouds.shape[1], 3)
                pad = np.zeros(pad_shape, dtype=np.float32)
                self.point_clouds = np.concatenate([self.point_clouds, pad], axis=0)

        print(f"✅ 点云形状: {self.point_clouds.shape}（每样本 {self.point_clouds.shape[1]} 个点，原始点 {num_points_per_sample_original} 个）")

        # === 步骤4：划分训练/测试索引 ===
        print(f"【步骤4/8】按 {split_type} 场景划分索引...")
        self.train_indices, self.test_indices = self._split_by_scenario(split_type, test_size)
        print(f"  - 训练集: {len(self.train_indices)} 样本")
        print(f"  - 测试集: {len(self.test_indices)} 样本")

        # === 步骤5：标准化（仅用训练集）===
        print("【步骤5/8】基于训练集拟合标准化器...")

        # Drug 嵌入标准化（可选）
        if self.standardize_embeddings:
            self.scaler_drug = StandardScaler()
            self.scaler_drug.fit(self.drug_embeddings_raw[self.train_indices])
            self.drug_embeddings = self.scaler_drug.transform(self.drug_embeddings_raw).astype(np.float32)
        else:
            self.drug_embeddings = self.drug_embeddings_raw.copy()
            self.scaler_drug = None

        # Target 嵌入标准化（可选）
        if self.standardize_embeddings:
            self.scaler_target = StandardScaler()
            self.scaler_target.fit(self.target_embeddings_raw[self.train_indices])
            self.target_embeddings = self.scaler_target.transform(self.target_embeddings_raw).astype(np.float32)
        else:
            self.target_embeddings = self.target_embeddings_raw.copy()
            self.scaler_target = None

        # 几何特征标准化（如有）
        if self.geometric_features_raw.shape[1] > 0:
            self.scaler_geo = StandardScaler()
            self.scaler_geo.fit(self.geometric_features_raw[self.train_indices])
            self.geometric_features = self.scaler_geo.transform(self.geometric_features_raw).astype(np.float32)
        else:
            self.geometric_features = self.geometric_features_raw.copy()
            self.scaler_geo = None

        # === 关键修改：DTA 任务不再归一化标签！===
        # 直接使用原始标签
        self.labels = self.labels_raw.copy()
        self.label_scaler = None  # 不再使用 MinMaxScaler

        # 打印原始标签范围（仅用于参考）
        if self.task_type == "dta":
            print(f"  - 亲和力原始范围: [{self.labels.min():.4f}, {self.labels.max():.4f}]")
        else:
            print(f"  - 标签类别数: {len(np.unique(self.labels))}")

        print("【步骤6/8】数据集初始化完成！\n")

        # ========== 新增：添加 ID 映射属性 ==========
        # 构建 drug_id/protein_id 与样本索引的映射表
        self.id_map = pd.DataFrame({
            "drug_id": self.drug_ids,       # 你原有代码已经定义的 self.drug_ids
            "protein_id": self.target_ids   # 你原有代码已经定义的 self.target_ids
        })

    # ========== 新增：通过样本索引查 drug_id/protein_id 的方法 ==========
    def get_ids_by_index(self, idx):
        """
        根据样本索引获取对应的药物ID和蛋白ID
        参数：idx - 样本索引（int）
        返回：(drug_id, target_id[, gene_id]) 元组
        """
        row = self.id_map.iloc[idx]
        
        # 兼容 DTI/DTA/MOA 列名
        drug_id = row["drug_id"] if "drug_id" in row else row.get("DrugID", None)
        target_id = row["protein_id"] if "protein_id" in row else row.get("TargetID", None)
        gene_id = row.get("gene_id", None)
        
        if gene_id is not None:
            return drug_id, target_id, gene_id
        return drug_id, target_id

    def _get_default_id_cols(self):
        """获取默认ID列名，适配不同任务类型"""
        if self.task_type == "moa":
            # 修复：MOA 只返回2列核心ID，避免后续过滤后触发错误
            return ["DrugID", "TargetID"]
        else: # dti,dta
            return ["drug_id", "protein_id"]

    def _split_by_scenario(self, split_type, test_size):
        """按场景划分训练/测试集，支持暖启动和多种冷启动"""
        all_indices = np.arange(len(self.drug_ids))

        if split_type == "warm":
            return train_test_split(all_indices, test_size=test_size, random_state=42)

        elif split_type == "drug_cold":
            unique_entities = np.unique(self.drug_ids)
            n_test = max(1, int(len(unique_entities) * test_size))
            np.random.seed(42)
            test_entities = np.random.choice(unique_entities, n_test, replace=False)
            test_mask = np.isin(self.drug_ids, test_entities)
            test_idx = all_indices[test_mask]
            train_idx = all_indices[~test_mask]

        elif split_type == "target_cold":
            unique_entities = np.unique(self.target_ids)
            n_test = max(1, int(len(unique_entities) * test_size))
            np.random.seed(42)
            test_entities = np.random.choice(unique_entities, n_test, replace=False)
            test_mask = np.isin(self.target_ids, test_entities)
            test_idx = all_indices[test_mask]
            train_idx = all_indices[~test_mask]

        else:
            raise ValueError(f"不支持的 split_type: {split_type}")

        if len(train_idx) == 0 or len(test_idx) == 0:
            print("⚠️ 冷启动划分失败，回退到随机划分")
            return train_test_split(all_indices, test_size=test_size, random_state=42)

        return train_idx.tolist(), test_idx.tolist()

    def __len__(self):
        """返回总样本数"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        数据获取接口，返回四项数据（支持模型交互建模）：
            point_cloud: (num_points, 3) —— 处理后的点云数据
            drug_emb: (drug_dim,)       —— ChemBERTa 提取的药物嵌入
            target_emb: (target_dim,)   —— ESM-2 提取的靶点嵌入
            label: scalar               —— 任务标签（DTA为float，其他为long）
        """
        point_cloud = torch.tensor(self.point_clouds[idx], dtype=torch.float32)
        drug_emb = torch.tensor(self.drug_embeddings[idx], dtype=torch.float32)
        target_emb = torch.tensor(self.target_embeddings[idx], dtype=torch.float32)
        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32 if self.task_type == "dta" else torch.long
        )
        return point_cloud, drug_emb, target_emb, label