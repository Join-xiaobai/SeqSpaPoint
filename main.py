"""
统一主入口脚本：支持 DTA（Drug-Target Affinity）、DTI（Drug-Target Interaction）、MOA（Mechanism of Action）三大任务。

设计目标：
  - 单一入口文件，避免维护多个几乎相同的 main_xxx.py
  - 所有实验配置集中管理，修改实验只需调整常量或配置字典
  - 自动执行数据预处理（点云 + 嵌入生成），再启动训练
  - 无外部依赖（如命令行参数），便于在服务器/IDE 中一键运行
  - 符合 Python PEP 8 规范，具备良好可读性和可维护性

使用流程：
  1. 设置 TASK_NAME 为 "dta" / "dti" / "moa", 同时设置CONFIG的相关参数.
  2. 确保原始数据已放在 ./data/{task}/{dataset}/ 目录下
  3. 运行 `python main.py`
  4. 脚本将自动：
        a) 生成药物/靶点嵌入（若缺失）
        b) 构建特征 CSV 和归一化点云 PLY
        c) 启动对应任务的交叉验证训练
  5. 结果保存至 ./result/...

注意事项：
  - 本脚本不接受命令行参数（简化使用）
  - 若需复现实验，请固定 TASK_NAME 和随机种子（已在配置中设定）
  - 数据路径基于项目根目录，请确保相对路径有效
  - 预处理输出路径与训练输入路径严格对齐，避免路径错位
"""

import os
import sys

# 确保能正确导入 data_preprocessing 模块（即使从项目根目录运行）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# 🔧 第一步：选择当前要运行的任务类型
# ------------------------------------------------------------------------------
# 支持三种任务：
#   - "dta": 预测药物与靶点之间的亲和力（回归任务，标签为连续值）
#   - "dti": 预测药物与靶点是否存在相互作用（二分类任务，标签为 0/1）
#   - "moa": 预测药物作用机制（二分类任务，标签为 0/1）
#
# 修改此变量即可切换整个实验流程（包括预处理 + 训练），无需改动其他代码。
# ==============================================================================
TASK_NAME = "moa"  # ←←← 在此处修改任务名称（可选值: "dta", "dti", "moa"）


# ==============================================================================
# 📦 第二步：根据任务动态导入对应的训练函数
# ------------------------------------------------------------------------------
# 每个任务使用不同的训练逻辑（如损失函数、评估指标、冷启动策略等），
# 因此 train_dta.py、train_dti.py、train_moa.py 是独立实现的。
# 这里通过条件导入，将正确的函数别名为 `train_func` 供后续调用。
# 注意：所有训练函数必须命名为 `train_with_cv`
# ==============================================================================
if TASK_TYPE := TASK_NAME:
    if TASK_TYPE == "dta":
        from train.train_dta import train_with_cv as train_func
    elif TASK_TYPE == "dti":
        from train.train_dti import train_with_cv as train_func
    elif TASK_TYPE == "moa":
        from train.train_moa import train_with_cv as train_func  # 确保train_moa.py存在且函数名正确
    else:
        raise ValueError(
            f"❌ 不支持的任务类型: '{TASK_TYPE}'.\n"
            "✅ 请设置 TASK_NAME 为以下之一: 'dta', 'dti', 'moa'"
        )
else:
    raise RuntimeError("TASK_NAME 不能为空")


# ==============================================================================
# ⚙️ 第三步：定义各任务的完整配置（数据集 + 超参数 + 列名 + 其他）
# ------------------------------------------------------------------------------
# 注意：
#   - dataset_name 必须与 ./data/{task}/{dataset_name}/ 下的实际目录名一致
#   - label_col 是 CSV 中标签列的名称
#   - id_cols 是 [drug_id列, target_id列]，MOA 与其他任务不同
#   - MOA 为二分类任务，无需num_classes（删除多余配置）
# ==============================================================================
CONFIG = {
    "dta": {
        "dataset_name": "kiba",  # 可选: "davis", "kiba"
        "label_col": "affinity",
        "id_cols": ["drug_id", "protein_id"],
        "training_args": {
            "split_types": ["warm", "drug_cold", "target_cold"],
            "num_folds": 5,
            "num_epochs": 300,
            "batch_size": 128,
            "lr": 1e-4, 
            "warmup_epochs": 15, 
            "weight_decay": 1e-5,
            "patience": 50,
            "seed": 42,
            "dropout_rate": 0.2, # kiba 用 0.1 davis 用 0.2
            # KIBA数据集全部设置 False 关闭, Davis数据集全部设置 True 打开
            "use_dynamic_lambda_warmup": False,
            "use_dataset_lambda_clip": False,
            "use_batch_threshold": False,
            "use_numerical_stability": False,
        },
    },
    "dti": {
        "dataset_name": "hetionet",  # 可选: "hetionet", "yamanishi_08"
        "label_col": "label",
        "id_cols": ["drug_id", "protein_id"],
        "training_args": {
            "split_types": ["warm", "drug_cold", "target_cold"],
            "num_folds": 10,
            "num_epochs": 300,
            "batch_size": 128,
            "lr": 1e-4,
            "warmup_epochs": 15,
            "weight_decay": 1e-5,
            "patience": 50,
            "seed": 42,
            # ========== 补充 DTI 专属参数 ==========
            "dropout_rate": 0.2,          # DTI 模型 dropout 率
            "use_focal_loss": True,        # 是否使用 Focal Loss
            "use_class_weight": True,      # 是否加权采样
            "use_numerical_stability": True, # 数值稳定
            "focal_gamma": 2.0,            # Focal Loss 参数
            "focal_alpha": 0.25            # Focal Loss 参数
        },
    },
    "moa": {
        "dataset_name": "inhibition",  # 可选: "inhibition", "activation"
        "label_col": "label",  # MOA二分类标签列统一为"label"（和数据集类对齐）
        "id_cols": ["DrugID", "TargetID"],  # MOA 专属ID列名
        "training_args": {
            # 仅保留MOA核心场景（warm + drug_cold + target_cold）
            "split_types": ["warm", "drug_cold", "target_cold"],
            "num_folds": 5,
            "num_epochs": 300,  # MOA训练轮数适配
            "batch_size": 128,   # MOA批次大小适配
            "lr": 1e-4,
            "warmup_epochs": 15, # MOA热身轮数
            "weight_decay": 1e-5,
            "patience": 50,     # MOA早停耐心值
            "seed": 42,
            "dropout_rate": 0.15, # MOA专属dropout率
            # ========== MOA二分类专属参数（必需） ==========
            "use_focal_loss": True,        # MOA必须使用Focal Loss解决类别不平衡
            "use_class_weight": True,      # MOA必须加权采样
            "use_numerical_stability": True, # MOA数值稳定保护
            "focal_gamma": 2.0,            # MOA Focal Loss参数
            "focal_alpha": 0.25            # MOA Focal Loss参数
        },
    },
}


# ==============================================================================
# 🛠️ 第四步：定义路径构建辅助函数
# ------------------------------------------------------------------------------
# 该函数返回训练所需的数据文件路径，这些文件由预处理模块生成。
# 路径格式必须与 point_cloud_coordinate_construction.py 中的 BASE_OUTPUT_DIR 一致。
# ==============================================================================
def build_data_paths(task_name: str, dataset_name: str) -> dict:
    """
    动态构建训练所需的数据文件路径。
    
    路径基于预处理模块的输出目录：
        ./data_preprocessing/{task}/{dataset}/log_and_file/
    
    返回:
        dict: 包含 feature_csv_path 和 point_cloud_ply_path 的字典
    """
    base_output_dir = f"./data_preprocessing/{task_name}/{dataset_name}/log_and_file"
    # MOA的文件名和其他任务保持一致（避免路径错误）
    feature_file = f"{task_name}_features.csv"
    point_cloud_file = f"{task_name}_point_cloud.ply"

    return {
        "feature_csv_path": os.path.join(base_output_dir, feature_file),
        "point_cloud_ply_path": os.path.join(base_output_dir, point_cloud_file),
    }


# ==============================================================================
# 🧠 第五步：主执行逻辑（智能判断是否需预处理，再训练）
# ==============================================================================
def main():
    """主程序逻辑：智能判断是否需要运行预处理，再启动训练流程。"""
    task_config = CONFIG[TASK_NAME]
    dataset_name = task_config["dataset_name"]

    print("=" * 70)
    print(f"🚀 启动 {TASK_NAME.upper()} 任务全流程（数据集: {dataset_name}）")
    print("=" * 70)

    # ────────────────────────────────────────────────────────
    # 🔍 步骤 1: 检查预处理输出是否已存在
    # ────────────────────────────────────────────────────────
    data_paths = build_data_paths(TASK_NAME, dataset_name)
    feature_csv_path = data_paths["feature_csv_path"]
    point_cloud_ply_path = data_paths["point_cloud_ply_path"]

    feature_exists = os.path.exists(feature_csv_path)
    point_cloud_exists = os.path.exists(point_cloud_ply_path)

    if feature_exists and point_cloud_exists:
        print("✅ 检测到预处理输出已存在，跳过数据预处理阶段。")
        print(f"   - 特征文件: {feature_csv_path}")
        print(f"   - 点云文件: {point_cloud_ply_path}")
    else:
        print("\n🔧 预处理输出不完整，正在运行数据预处理（嵌入生成 + 点云构建）...")
        try:
            from data_preprocessing.point_cloud_coordinate_construction import run_preprocessing
            run_preprocessing(TASK_NAME, dataset_name)
            print("✅ 预处理完成！\n")
        except Exception as e:
            print(f"\n❌ 预处理失败，终止训练流程：{e}")
            return

    # ────────────────────────────────────────────────────────
    # 📊 步骤 2: 验证文件存在性（双重保险）
    # ────────────────────────────────────────────────────────
    if not (os.path.exists(feature_csv_path) and os.path.exists(point_cloud_ply_path)):
        raise FileNotFoundError(
            "❌ 预处理后仍未找到所需文件！请检查 point_cloud_coordinate_construction.py 是否正常运行。"
        )

    print("📁 特征文件路径     :", feature_csv_path)
    print("☁️  点云文件路径     :", point_cloud_ply_path)
    print("🏷️  标签列名         :", task_config["label_col"])
    print("🧬 ID 列名           :", task_config["id_cols"])
    print("-" * 70)

    # ────────────────────────────────────────────────────────
    # 🚀 步骤 3: 启动训练（传入统一参数）
    # ────────────────────────────────────────────────────────
    base_save_dir = os.path.join(
        "./result",
        f"{TASK_NAME}_experiment",
        dataset_name,
    )
    os.makedirs(base_save_dir, exist_ok=True)

    print(f"\n🎯 启动训练（划分策略: {task_config['training_args']['split_types']}）")
    print(f"💾 基础保存目录: {base_save_dir}")

    # 构建传递给 train_func 的完整参数（统一接口）
    train_kwargs = {
        "FEATURE_CSV_PATH": feature_csv_path,
        "POINT_CLOUD_PLY_PATH": point_cloud_ply_path,
        "BASE_SAVE_DIR": base_save_dir,
        "task_type": TASK_NAME,
        "label_col": task_config["label_col"],
        "id_cols": task_config["id_cols"],
        "dataset_name": dataset_name,
    }

    # 合并通用训练超参数
    train_kwargs.update(task_config["training_args"])

    # 调用训练函数
    train_func(**train_kwargs)

    # ────────────────────────────────────────────────────────
    # ✅ 完成提示
    # ────────────────────────────────────────────────────────
    overall_result_dir = os.path.join("./result", f"{TASK_NAME}_experiment", dataset_name)
    print("\n" + "=" * 70)
    print(f"🎉 {TASK_NAME.upper()} 任务（数据集: {dataset_name}）全流程执行完毕！")
    print(f"📊 最终结果目录: {overall_result_dir}")
    print("=" * 70)


# ==============================================================================
# 🏁 第六步：脚本入口保护
# ==============================================================================
if __name__ == "__main__":
    main()