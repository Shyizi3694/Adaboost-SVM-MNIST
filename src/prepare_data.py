# src/prepare_data.py
import sys
from pathlib import Path
import time # 用于模拟耗时操作或简单计时（可选）

# --- 动态路径调整 (确保能找到 src.config 等模块) ---
# 这使得脚本可以直接从 src 目录外运行 (例如从项目根目录 python src/prepare_data.py)
# 或者通过 python -m src.prepare_data 运行
try:
    # 如果脚本是以 python -m src.prepare_data 方式运行，下面的导入应该能工作
    # 如果是直接 python src/prepare_data.py，可能需要调整路径
    # 为了健壮性，我们先尝试添加项目根目录到sys.path
    current_file_path = Path(__file__).resolve()
    # PROJECT_ROOT is parent of src, and __file__ is src/prepare_data.py
    # So, parent.parent gives PROJECT_ROOT
    PROJECT_ROOT_GUESS = current_file_path.parent.parent 
    if str(PROJECT_ROOT_GUESS) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_GUESS))
        print(f"信息: 已将项目根目录 '{PROJECT_ROOT_GUESS}' 添加到 sys.path。")
except Exception as e:
    print(f"警告: 尝试动态调整 sys.path 失败: {e}")


# --- 导入项目模块 ---
try:
    from src import config
    from src.data_preprocess.loader import (
        download_mnist_dataset, 
        load_raw_mnist_data,
        load_processed_train_data, # 用于加载未标准化的训练数据以拟合Scaler
        load_processed_test_data   # 用于加载未标准化的测试数据以传递给standardscalar_processor
    )
    from src.data_preprocess.processor import (
        split_and_save_data,
        standardscalar_processor
    )
except ImportError as e:
    print(f"CRITICAL: 导入必要的项目模块失败: {e}")
    print("请确保您在项目的根目录下，并且 src 目录结构正确，或者 PYTHONPATH 已设置。")
    print("尝试运行: python -m src.prepare_data")
    sys.exit(1)

def main():
    """
    执行所有一次性的数据准备步骤：
    1. 下载原始数据。
    2. 加载原始数据。
    3. 划分数据并保存（未标准化的）。
    4. 加载未标准化的训练/测试数据，拟合StandardScaler，并保存Scaler。
    """
    print("="*60)
    print("开始执行数据准备脚本 (prepare_data.py)...")
    print("="*60)

    # --- 0. 定义路径并确保目录存在 ---
    print("\n--- 步骤 0: 初始化路径和目录 ---")
    try:
        raw_data_dir = Path(config.RAW_DATA_DIR)
        processed_dir = Path(config.PROCESSED_DATA_DIR)
        utils_dir = Path(config.UTILS_DIR)
        
        # 确保这些目录存在，脚本后面会用到它们
        raw_data_dir.mkdir(parents=True, exist_ok=True) # download_mnist_dataset 内部也会创建
        processed_dir.mkdir(parents=True, exist_ok=True)
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_filename = "mnist_standard_scaler.pkl" # 标准的scaler文件名
        scaler_save_path = utils_dir / scaler_filename

        print(f"  原始数据目录: {raw_data_dir}")
        print(f"  预处理数据目录 (保存划分后的数据): {processed_dir}")
        print(f"  工具/辅助文件目录 (保存Scaler): {utils_dir}")
        print(f"  Scaler将保存为: {scaler_save_path}")
    except AttributeError as e:
        print(f"错误: config.py 中缺少必要的路径定义 (例如 RAW_DATA_DIR, PROCESSED_DATA_DIR, UTILS_DIR)。错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 初始化路径或创建目录时失败: {e}")
        sys.exit(1)


    # --- 1. 下载原始数据 ---
    print("\n--- 步骤 1: 检查并下载原始MNIST数据集 ---")
    try:
        download_mnist_dataset() # 此函数内部有打印信息
        print("步骤 1 完成: 原始数据已确保可用。")
    except Exception as e:
        print(f"错误: 下载原始MNIST数据集失败: {e}")
        # traceback.print_exc() # 取消注释以查看详细堆栈
        sys.exit(1) # 下载失败则无法继续

    # --- 2. 加载原始数据 ---
    print("\n--- 步骤 2: 加载完整的原始MNIST数据 ---")
    try:
        X_all, y_all = load_raw_mnist_data() # 此函数内部有打印信息
        print(f"  加载的完整数据形状: X_all: {X_all.shape}, y_all: {y_all.shape}")
        print("步骤 2 完成: 完整原始数据加载完毕。")
    except Exception as e:
        print(f"错误: 加载原始MNIST数据失败: {e}")
        sys.exit(1)

    # --- 3. 划分数据并保存（未标准化的） ---
    print("\n--- 步骤 3: 划分数据集为训练集和测试集，并保存（未标准化的） ---")
    # MNIST 标准划分：60000训练，10000测试
    num_train_samples = 60000 
    try:
        split_and_save_data(X_all, y_all, 
                            train_samples=num_train_samples, 
                            processed_data_dir=processed_dir) # 此函数内部有打印信息
        print(f"步骤 3 完成: 数据已划分为训练集和测试集，并保存在 '{processed_dir}'。")
    except Exception as e:
        print(f"错误: 划分并保存数据失败: {e}")
        sys.exit(1)

# --- 步骤 3.1: 验证已保存的划分后数据的形状 ---
    print("\n--- 步骤 3.1: 验证已保存的划分后数据的形状 ---")
    try:
        print(f"  正在从 '{processed_dir}' 加载已划分的训练数据进行验证...")
        X_train_check, y_train_check = load_processed_train_data(processed_data_dir=processed_dir)
        print(f"    加载的训练数据形状: X_train_check: {X_train_check.shape}, y_train_check: {y_train_check.shape}")

        print(f"  正在从 '{processed_dir}' 加载已划分的测试数据进行验证...")
        X_test_check, y_test_check = load_processed_test_data(processed_data_dir=processed_dir)
        print(f"    加载的测试数据形状: X_test_check: {X_test_check.shape}, y_test_check: {y_test_check.shape}")
        
        # 简单校验样本总数是否与原始数据一致 (如果 X_all 和 y_all 仍然在作用域内)
        if 'X_all' in locals() and 'y_all' in locals(): # 检查变量是否存在
            if (X_train_check.shape[0] + X_test_check.shape[0]) == X_all.shape[0]:
                print("  验证通过：划分后的训练集和测试集样本总数与原始数据一致。")
            else:
                print(f"  警告：划分后的样本总数 ({X_train_check.shape[0] + X_test_check.shape[0]}) "
                      f"与原始数据总数 ({X_all.shape[0]}) 不一致。")
        print("步骤 3.1 完成: 已划分数据的形状验证完毕。")
    except FileNotFoundError as e_fnf:
        print(f"错误 (步骤 3.1): 验证时未能找到预处理后的数据文件: {e_fnf}")
        print(f"      请确保步骤3中的 'split_and_save_data' 函数已成功执行并保存了 .npy 文件。")
        # 根据情况，您可能希望在这里也 sys.exit(1)
    except Exception as e:
        print(f"错误 (步骤 3.1): 验证已划分数据时发生错误: {e}")
        # 根据情况，您可能希望在这里也 sys.exit(1)

    # --- 4. 拟合StandardScaler并保存 ---
    print("\n--- 步骤 4: 加载未标准化的训练/测试数据，拟合StandardScaler并保存Scaler对象 ---")
    try:
        print("  正在加载刚保存的未标准化的训练数据 (用于拟合Scaler)...")
        X_train_unscaled, _ = load_processed_train_data(processed_data_dir=processed_dir)
        # standardscalar_processor 需要 X_test 来进行 transform (虽然我们这里不直接用它的输出)
        # 或者，我们可以修改 standardscalar_processor 只接收 X_train 来拟合和保存
        # 但保持其现有接口，我们加载 X_test_unscaled
        print("  正在加载刚保存的未标准化的测试数据 (用于传递给 standardscalar_processor)...")
        X_test_unscaled, _ = load_processed_test_data(processed_data_dir=processed_dir)

        print(f"  将使用 X_train_unscaled (shape: {X_train_unscaled.shape}) 来拟合Scaler。")
        
        # standardscalar_processor 会在内部拟合、转换并保存scaler
        # 它返回的 X_train_scaled, X_test_scaled 在此脚本中不是主要目的，主要目的是保存scaler
        _, _ = standardscalar_processor(
            X_train_unscaled, 
            X_test_unscaled, 
            scaler_save_path=scaler_save_path
        ) # 此函数内部有打印信息
        print(f"步骤 4 完成: StandardScaler已在训练数据上拟合，并保存至 '{scaler_save_path}'。")
    except FileNotFoundError as e_fnf:
        print(f"错误: 加载预处理（但未标准化）的数据文件失败，这不应该发生在此阶段: {e_fnf}")
        print(f"      请检查 '{processed_dir}' 目录中的 .npy 文件是否已由步骤3正确生成。")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 拟合或保存Scaler失败: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("数据准备脚本 (prepare_data.py) 执行完毕！")
    print("现在您的项目中应该具备：")
    print(f"  1. 原始数据缓存于: '{raw_data_dir}'")
    print(f"  2. 已划分但未标准化的数据保存于: '{processed_dir}' (X_train.npy, y_train.npy, 等)")
    print(f"  3. 已拟合的StandardScaler对象保存于: '{scaler_save_path}'")
    print("="*60)

if __name__ == "__main__":
    main()