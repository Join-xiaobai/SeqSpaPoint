# 模型文件配置说明（由于模型文件比较大无法直接上传预训练模型）
本项目预处理脚本依赖以下两个预训练模型，需严格按指定路径存放所有文件，确保脚本正常加载。

## 一、ESM-2 t33_650M_UR50D（靶点嵌入模型）
### 1. 存放路径
data_preprocessing/models/esm2_t33_650M_UR50D/
### 2. 需存放的全部文件及功能（下载链接： https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main）
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

## 二、ChemBERTa（药物嵌入模型）
### 1. 存放路径
data_preprocessing/models/chemberta/
### 2. 需存放的全部文件及功能（下载链接： https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main）
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

## 三、验证配置成功
将所有文件放入对应路径后，运行预处理脚本point_cloud_coordinate_construction.py，无`FileNotFoundError`报错即配置完成。