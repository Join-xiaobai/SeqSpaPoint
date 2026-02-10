# PointCloud-DTA/DTI/MOA 预测项目
SeqSpaPoint: A multitask framework based on sequence-derived point clouds and cross-modal attention

## 项目简介
本项目是一款基于深度学习的生物信息学多任务预测框架，核心支持**药物-靶点亲和力预测（DTA）**、**药物-靶点相互作用预测（DTI）** 和**药物作用机制预测（MOA）** 三大核心任务。创新融合点云几何特征与序列嵌入特征，适配暖启动（warm）、药物冷启动（drug_cold）、靶点冷启动（target_cold）三种实际应用场景，通过单一入口脚本与集中式配置，实现“零代码修改”切换任务与数据集，兼顾易用性与泛化能力。

## 核心特性
- **多任务统一框架**：单脚本支持三大任务，共享预处理与训练流程，降低开发维护成本
- **点云特征创新**：构建药物/靶点3D点云表示，捕捉空间结构信息，提升预测精度
- **全场景适配**：支持暖启动、药物冷启动、靶点冷启动，覆盖新药/新靶点预测需求
- **高鲁棒性设计**：集成Focal Loss、动态λ调度、数值稳定等机制，缓解类别不平衡与训练震荡
- **可复现性保障**：固定随机种子、统一配置参数、完整日志记录，实验结果可精准复现

## 项目结构
```
point_cloud_model/
├── ablation_experiment/     # 消融实验模块
│   ├── ablation_model/      # 4组消融模型（移除点云/特征等变体）
│   │   ├── model_seqspa_point_nopoint.py      # 移除点云特征
│   │   ├── model_seqspa_point_nofeature.py   # 移除法线向量特征
│   │   ├── model_seqspa_point_randomFeature.py # 随机特征向量
│   │   └── model_seqspa_point_randomPoint.py  # 随机噪声点云
│   ├── results/             # 消融实验结果日志
│   └── visualization_of_ablation_experiment*.py # 注意力可视化脚本（4组对应实验）
├── case_study/          # MTC临床案例分析模块
│   ├── chembl29/                # 原始数据集
│   ├── MTC/                # 案例数据集（DAVIS子集）
│   ├── pdbbind/                # 原始数据集
│   ├── UniProt/                # 原始数据集
│   ├── step1_build_mtc_dta_benchmark.py # 基准集构建
│   ├── step2.1_tuning_training_MTC.py # 模型微调训练
│   ├── step2.2_mtc_finetuned_predict.py # 模型预测
│   └── step3_analysis_results.py # 结果分析与可视化
├── data/                    # 原始数据集目录（需自行存放）
│   ├── dta/                 # DTA数据集（davis、kiba）
│   ├── dti/                 # DTI数据集（hetionet、yamanishi_08）
│   └── moa/                 # MOA数据集（inhibition、activation）
├── data_preprocessing/      # 数据预处理模块
│   └── point_cloud_coordinate_construction.py  # 点云构建+嵌入生成
│   └── model/               # 序列生成预训练模型
├── dataset/                   # 工具函数模块（新增详细说明）
│   └── interaction_dataset.py      # 数据处理工具：特征归一化、点云采样、嵌入向量加载
├── evalMetrics/                   # 数据验证评估模块
│   ├── classification_metrics.py      # 评估指标工具：DTI/MOA（AUPRC/AUROC/F1）计算
│   ├── eval_utils.py       # 模型评估：评估测试
│   └── visualization_utils.py # 评估指标工具：DTA（CI/Pearson/RMSE）计算
├── model/                   # 核心模型
│   └── SeqSpaPoint.py      # SeqSpaPoint模型定义模块
├── result/                   # 自动生成的训练输出目录
│   ├── dta_experiment.py         # DTA任务结果（模型、日志、指标）
│   ├── dti_experiment.py         # DTI任务结果
│   └── moa_experiment.py         # MOA任务结果
├── train/                   # 任务训练模块
│   ├── train_dta.py         # DTA任务训练逻辑（回归）
│   ├── train_dti.py         # DTI任务训练逻辑（分类）
│   └── train_moa.py         # MOA任务训练逻辑（分类）
├── utils/                   # 工具函数模块
│   └── common_utils.py      # 处理工具
├── visualization/           # 结果可视化脚本（生成FigureA-G）
├── main.py                  # 统一主入口脚本（任务切换核心）
└── requirements.txt         # 依赖包清单
```

## 环境配置
### 依赖安装
推荐Python 3.8-3.10版本，通过conda创建虚拟环境并安装依赖（详细信息请参考`requirements.txt`）：
```bash
conda create -n pointcloud-bio python=3.9
conda activate pointcloud-bio
pip install -r requirements.txt
```

### 核心依赖说明
| 类别 | 关键依赖 | 版本要求 |
|------|----------|----------|
| 深度学习框架 | PyTorch | ≥1.12.0 |
| 3D点云处理 | Open3D、plyfile | Open3D≥0.15.0 |
| 数据处理 | NumPy、Pandas、SciPy | NumPy≥1.24.3 |
| 模型评估 | scikit-learn | ≥1.2.2 |
| 序列嵌入 | Transformers | ≥4.30.2 |
| 工具类 | tqdm、json5 | - |

## 快速使用
### 1. 数据准备
（1）在`data/`目录下按任务创建数据集文件夹，示例结构：
```
data/
└── dta/
    └── davis/  # 数据集名称需与配置一致
        └── 相关数据集  # 原始数据需包含ID列、序列列、标签列
```
（2）下载ESM-2 t33_650M_UR50D（靶点嵌入模型）、ChemBERTa（药物嵌入模型）文件，保存到`data_preprocessing/models/`文件夹中（下载地址和相关说明可查看`data_preprocessing/models/models_readme.md`，以下将简要介绍核心必要部分）：

#### 一、ESM-2 t33_650M_UR50D（靶点嵌入模型）
##### 1. 存放路径
`data_preprocessing/models/esm2_t33_650M_UR50D/`

##### 2. 需存放的全部文件及功能（下载链接：https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main）
| 文件名                | 核心功能                     |
|-----------------------|------------------------------|
| config.json           | 定义模型结构（层数、维度等） |
| model.safetensors     | 模型权重的安全存储格式文件   |
| pytorch_model.bin     | PyTorch格式的模型权重文件（生成靶点嵌入） |
| README.md             | 模型说明文档                 |
| special_tokens_map.json | 特殊符号映射配置文件         |
| tf_model.h5           | TensorFlow格式的模型权重文件 |
| tokenizer_config.json | 蛋白序列编码规则配置         |
| vocab.txt             | 氨基酸符号词汇表             |

#### 二、ChemBERTa（药物嵌入模型）
##### 1. 存放路径
`data_preprocessing/models/chemberta/`

##### 2. 需存放的全部文件及功能（下载链接：https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main）
| 文件名                | 核心功能                     |
|-----------------------|------------------------------|
| added_tokens.json     | 新增符号配置文件             |
| config.json           | 定义模型结构（层数、维度等） |
| merges.txt            | Tokenizer的子词合并规则文件  |
| pytorch_model.bin     | PyTorch格式的模型权重文件（生成药物嵌入） |
| special_tokens_map.json | 特殊符号映射配置文件         |
| tokenizer_config.json | SMILES序列编码规则配置       |
| tokenizer.json        | Tokenizer的完整配置文件      |
| training_args.bin     | 训练参数存储文件             |
| vocab.json            | 化学符号词汇表               |

原始数据要求：
- DTA：需含`drug_id`、`protein_id`、`affinity`（亲和力值）
- DTI：需含`drug_id`、`protein_id`、`label`（0/1）
- MOA：需含`DrugID`、`TargetID`、`label`（0/1）

### 2. 任务配置
修改`main.py`中的核心配置（无需改动其他代码）：
```python
# 步骤1：选择任务（dta/dti/moa）
TASK_NAME = "dta"  # 切换任务仅需修改此处

# 步骤2：配置数据集与超参数（已预设最优值）
CONFIG = {
    "dta": {
        "dataset_name": "davis",  # 可选davis/kiba
        "label_col": "affinity",
        "id_cols": ["drug_id", "protein_id"],
        "training_args": {
            "split_types": ["warm", "drug_cold", "target_cold"],  # 需启用的场景
            "num_folds": 5,  # 交叉验证折数
            "num_epochs": 300,
            "batch_size": 128,
            "lr": 1e-4,
            # 其他超参数已按数据集优化，无需修改
        }
    }
    # DTI/MOA配置同理，已预设最优参数
}
```

### 3. 运行项目
直接执行主脚本，自动完成预处理与训练：
```bash
python main.py
```
脚本自动流程：
1. 检测并生成药物/靶点嵌入（缺失时自动运行）
2. 构建特征CSV与归一化点云PLY文件
3. 按配置启动交叉验证训练
4. 结果自动保存至`./result/`目录

### 4. 结果查看
训练输出目录结构（以DTA-davis为例）：
```
result/dta_experiment/davis/warm/
├── models/          # 最优模型权重（按折保存）
├── train_log.txt    # 完整训练日志（含每轮指标）
└── metrics.csv      # 最终评估指标汇总
```
核心指标说明：
- DTA：CI、Pearson、RMSE、MAE、R²
- DTI/MOA：AUPRC、AUROC、F1-score、Accuracy

## 消融实验说明
提供4种消融模型，验证核心模块有效性：
- `model_seqspa_point_nopoint.py`：移除点云特征，仅保留序列特征
- `model_seqspa_point_nofeature.py`：移除额外特征，仅保留点云坐标
- `model_seqspa_point_randomFeature.py`：使用随机生成的特征向量
- `model_seqspa_point_randomPoint.py`：使用随机噪声点云坐标

实验结果保存于`ablation_experiment/results/`，支持直接对比模块贡献度。

## 注意事项
1. 数据路径严格遵循`data/{task}/{dataset}/`格式，否则预处理失败
2. 首次运行会生成点云与嵌入文件（耗时较长），后续运行自动跳过
3. 复现实验需保持`seed=42`（已固定），避免随机因素影响结果
4. 支持CPU/GPU自动适配，GPU环境需安装对应版本PyTorch（CUDA≥11.3）

## 常见问题
- 预处理失败：检查原始数据格式是否符合要求，ID列与标签列名称是否匹配CONFIG配置
- 训练震荡：降低学习率至5e-5，或启用`use_numerical_stability=True`
- 指标不佳：确认是否选择正确场景（冷启动需适配专属超参数）
- 内存溢出：减小`batch_size`至64，或降低点云采样点数（预处理模块中调整）