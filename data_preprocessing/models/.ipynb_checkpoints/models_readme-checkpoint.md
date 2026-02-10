# 模型文件配置说明（由于模型文件比较大无法直接上传预训练模型）
本项目的数据预处理流程依赖两款领域内主流的预训练模型生成药物/靶点嵌入特征，是后续点云特征融合与模型训练的基础。需严格按指定路径存放所有模型文件，确保预处理脚本可正常加载权重与配置，避免因文件缺失或路径错误导致流程中断。

## 一、ESM-2 t33_650M_UR50D（靶点嵌入模型）
### 1. 模型背景与选用原因
ESM-2（Evolutionary Scale Modeling）是Meta AI推出的蛋白序列预训练模型，基于Transformer架构，在海量非冗余蛋白序列（UR50D）上训练，具备极强的蛋白序列特征提取能力。
本项目选用`t33_650M_UR50D`版本的核心原因：
- **规模适配**：6.5亿参数的中型模型，兼顾特征表达能力与计算效率，适配生物医药场景的算力需求；
- **任务匹配**：专为蛋白序列设计，能捕捉氨基酸序列的进化信息与结构关联，远超传统手工特征（如AAC、PseAAC）；
- **开源成熟**：支持PyTorch无缝集成，输出的高维嵌入向量可直接与点云特征融合，无需额外转换。

### 2. 存放路径
```
data_preprocessing/models/esm2_t33_650M_UR50D/
```

### 3. 需存放的全部文件及功能（下载链接：https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main）
| 文件名                | 核心功能                     | 必要性 |
|-----------------------|------------------------------|--------|
| config.json           | 定义模型结构（层数、维度、注意力头数等） | 必需 |
| model.safetensors     | 模型权重的安全存储格式文件（兼容PyTorch） | 可选（与pytorch_model.bin二选一） |
| pytorch_model.bin     | PyTorch格式的核心权重文件（生成靶点嵌入的核心） | 必需 |
| README.md             | 模型官方说明文档（含使用示例） | 可选 |
| special_tokens_map.json | 特殊符号（如<cls>、<pad>）映射配置 | 必需 |
| tf_model.h5           | TensorFlow格式的模型权重文件 | 可选（本项目仅用PyTorch） |
| tokenizer_config.json | 蛋白序列编码规则（氨基酸映射、最大长度等） | 必需 |
| vocab.txt             | 氨基酸符号词汇表（20种标准氨基酸+特殊符号） | 必需 |

## 二、ChemBERTa（药物嵌入模型）
### 1. 模型背景与选用原因
ChemBERTa是基于BERT架构的小分子药物预训练模型，专为SMILES（简化分子线性输入规范）序列设计，在ZINC海量小分子数据集上训练，是药物表征的主流选择。
本项目选用`ChemBERTa-zinc-base-v1`版本的核心原因：
- **领域适配**：针对SMILES序列优化Tokenizer，能准确解析药物分子的结构特征（如官能团、化学键）；
- **泛化能力**：训练数据覆盖千万级小分子，适配不同靶点类型的药物表征需求；
- **融合友好**：输出的嵌入维度与ESM-2一致，可直接与靶点嵌入、点云特征拼接，保证特征维度统一。

### 2. 存放路径
```
data_preprocessing/models/chemberta/
```

### 3. 需存放的全部文件及功能（下载链接：https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main）
| 文件名                | 核心功能                     | 必要性 |
|-----------------------|------------------------------|--------|
| added_tokens.json     | 新增化学符号（如环结构、取代基）配置 | 必需 |
| config.json           | 定义模型结构（层数、隐藏层维度等） | 必需 |
| merges.txt            | Tokenizer的子词合并规则（解析SMILES长序列） | 必需 |
| pytorch_model.bin     | PyTorch格式的核心权重文件（生成药物嵌入的核心） | 必需 |
| special_tokens_map.json | 特殊符号（如<s>、</s>）映射配置 | 必需 |
| tokenizer_config.json | SMILES序列编码规则（最大长度、截断策略等） | 必需 |
| tokenizer.json        | Tokenizer完整配置（含词汇表与编码逻辑） | 必需 |
| training_args.bin     | 模型训练参数记录文件 | 可选 |
| vocab.json            | 化学符号词汇表（SMILES字符映射） | 必需 |

## 三、配置验证与常见问题
### 1. 验证配置成功
将所有**必需**文件放入对应路径后，运行预处理脚本：
```bash
python data_preprocessing/point_cloud_coordinate_construction.py
```
若脚本无`FileNotFoundError`（文件未找到）、`ConfigError`（配置解析失败）等报错，且能正常生成`drug_embedding.npy`、`target_embedding.npy`文件，即说明模型配置完成。

### 2. 常见问题解决
- **权重文件缺失**：优先下载`pytorch_model.bin`（本项目核心依赖），`safetensors`/`tf_model.h5`可忽略；
- **Tokenizer报错**：检查`vocab.txt`/`vocab.json`是否完整，确保无字符缺失；
- **路径错误**：严格按`data_preprocessing/models/[模型名]/`路径存放，避免多级目录或拼写错误（如`esm2`写成`esm-2`）。

### 3. 算力优化建议
若本地算力有限，可：
- 下载模型后删除`tf_model.h5`等非必需文件，减少磁盘占用；
- 预处理时启用`torch.no_grad()`（脚本已内置），避免显存溢出；
- 对大规模数据集，分批生成嵌入向量，避免一次性加载全部序列。