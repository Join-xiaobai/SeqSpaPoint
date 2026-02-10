import os
import sys
import traceback
import importlib.util

# ===================== 核心配置（仅需配置相对路径，无绝对路径）=====================
# 1. 任务配置（固定不变）
TASK_MODE = "dta"                # DTA模式（药物-靶点亲和力）
DATASET_NAME = "mtc"             # 数据集标识

# 2. 相对路径配置（移植时只需调整这几个相对路径，无需改绝对路径）
# 预处理模块相对于当前脚本的路径（例：当前脚本在case study，模块在../data_preprocessing/）
PREPROCESS_MODULE_REL_PATH = "../data_preprocessing/point_cloud_coordinate_construction.py"
# 采样后的MTC数据相对于当前脚本的路径
MTC_SAMPLED_REL_PATH = "./MTC/MTC_sampled_5p"
# 预处理输出目录相对于当前脚本的路径
OUTPUT_REL_PATH = "./data_preprocessing/dta/mtc/log_and_file"

# ===================== 工具函数（动态推导路径，移植友好）=====================
def get_abs_path(rel_path: str) -> str:
    """
    动态推导绝对路径（基于当前脚本位置）
    :param rel_path: 相对于当前脚本的路径
    :return: 绝对路径
    """
    # 获取当前脚本的绝对目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接并规范化路径（自动处理../ ./等相对路径）
    abs_path = os.path.abspath(os.path.join(current_script_dir, rel_path))
    return abs_path

def import_preprocessing_module(rel_module_path: str):
    """
    基于相对路径导入预处理模块（完全移植友好）
    """
    # 动态推导模块绝对路径
    module_abs_path = get_abs_path(rel_module_path)
    
    # 1. 检查模块文件是否存在
    if not os.path.exists(module_abs_path):
        raise FileNotFoundError(
            f"❌ 预处理模块文件不存在\n"
            f"  相对路径：{rel_module_path}\n"
            f"  推导绝对路径：{module_abs_path}\n"
            f"请检查相对路径是否正确"
        )
    
    # 2. 动态加载模块（不依赖Python环境变量）
    try:
        spec = importlib.util.spec_from_file_location("preprocess_module", module_abs_path)
        preprocess_module = importlib.util.module_from_spec(spec)
        sys.modules["preprocess_module"] = preprocess_module
        spec.loader.exec_module(preprocess_module)
        print(f"✅ 成功加载预处理模块\n"
              f"  相对路径：{rel_module_path}\n"
              f"  绝对路径：{module_abs_path}")
    except Exception as e:
        raise ImportError(
            f"❌ 模块加载失败：{str(e)}\n"
            f"请确认模块文件无语法错误，且包含 run_preprocessing 函数"
        ) from e
    
    # 3. 检查核心函数是否存在
    if not hasattr(preprocess_module, "run_preprocessing"):
        raise AttributeError(
            f"❌ 预处理模块中未找到 run_preprocessing 函数\n"
            f"模块路径：{module_abs_path}"
        )
    
    return preprocess_module.run_preprocessing

def validate_input_files(rel_input_path: str):
    """校验采样文件（基于相对路径）"""
    input_abs_path = get_abs_path(rel_input_path)
    required_files = [
        ("ligands_can.txt", "药物SMILES文件"),
        ("proteins.txt", "靶点序列文件"),
        ("Y.npy", "亲和力矩阵文件")
    ]
    
    missing_or_empty = []
    for file_name, desc in required_files:
        file_abs_path = os.path.join(input_abs_path, file_name)
        if not os.path.exists(file_abs_path):
            missing_or_empty.append(f"{desc}（缺失）→ {file_abs_path}")
        elif os.path.getsize(file_abs_path) == 0:
            missing_or_empty.append(f"{desc}（空文件）→ {file_abs_path}")
    
    if missing_or_empty:
        raise ValueError(
            f"❌ 输入文件校验失败：\n" + "\n  ".join(missing_or_empty)
        )
    
    print(f"✅ 输入文件校验通过\n"
          f"  相对路径：{rel_input_path}\n"
          f"  绝对路径：{input_abs_path}")

def ensure_dirs():
    """确保所有目录存在（基于相对路径）"""
    # 推导所有目录的绝对路径
    input_abs_path = get_abs_path(MTC_SAMPLED_REL_PATH)
    output_abs_path = get_abs_path(OUTPUT_REL_PATH)
    
    # 创建目录（不存在则创建）
    os.makedirs(input_abs_path, exist_ok=True)
    os.makedirs(output_abs_path, exist_ok=True)
    
    print(f"✅ 目录准备完成\n"
          f"  采样数据目录（相对）：{MTC_SAMPLED_REL_PATH}\n"
          f"  采样数据目录（绝对）：{input_abs_path}\n"
          f"  输出目录（相对）：{OUTPUT_REL_PATH}\n"
          f"  输出目录（绝对）：{output_abs_path}")
    
    return input_abs_path, output_abs_path

# ===================== 主函数（核心执行逻辑）=====================
def main():
    print("=" * 88)
    print(f"          MTC采样文件预处理（DTA模式 | 完全移植友好）")
    print("=" * 88)
    
    try:
        # 步骤1：准备目录（动态推导路径）
        input_abs_path, output_abs_path = ensure_dirs()
        
        # 步骤2：导入预处理模块（基于相对路径）
        run_preprocessing = import_preprocessing_module(PREPROCESS_MODULE_REL_PATH)
        
        # 步骤3：校验输入文件
        validate_input_files(MTC_SAMPLED_REL_PATH)
        
        # 步骤4：执行预处理（无随机操作，结果可复现）
        print("\n🚀 开始执行预处理流程（纯确定性计算）...")
        run_preprocessing(
            mode=TASK_MODE,
            dataset_name=DATASET_NAME,
            input_dir=input_abs_path,
            output_dir=output_abs_path
        )
        
        # 步骤5：输出结果校验
        generated_files = [
            "drug_embeddings.pkl",
            "target_embeddings.pkl",
            "dta_point_cloud.ply",
            "dta_features.csv"
        ]
        
        print("\n📁 生成文件清单（输出目录：{}）".format(output_abs_path))
        for idx, file_name in enumerate(generated_files, 1):
            file_abs_path = os.path.join(output_abs_path, file_name)
            if os.path.exists(file_abs_path) and os.path.getsize(file_abs_path) > 0:
                size_mb = round(os.path.getsize(file_abs_path) / 1024 / 1024, 2)
                print(f"   {idx}. ✅ {file_name} | 大小：{size_mb}MB")
            else:
                print(f"   {idx}. ❌ {file_name} | 缺失或为空")
        
        print("\n🎉 MTC采样文件预处理全流程完成！")
        print(f"📌 移植友好特性：\n"
              f"  1. 无任何硬编码绝对路径，仅配置相对路径\n"
              f"  2. 路径基于当前脚本位置动态推导\n"
              f"  3. 可无缝复制到任意环境运行\n"
              f"  4. 结果100%可复现（无随机逻辑）")
    
    except Exception as e:
        print(f"\n❌ 预处理执行失败：{str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()