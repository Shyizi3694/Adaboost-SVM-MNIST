# src/data_preprocess/processor.py
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib # 用于保存和加载 sklearn 模型
from pathlib import Path


def standardscalar_processor(X_train: np.ndarray, X_test: np.ndarray, scaler_save_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    使用 StandardScaler 标准化训练集和测试集特征。
    Scaler 在 X_train 上进行拟合，然后用于转换 X_train 和 X_test。
    拟合后的 scaler 对象将被保存到 scaler_save_path。

    输入:
        X_train (np.ndarray): 训练集特征数据。
        X_test (np.ndarray): 测试集特征数据。
        scaler_save_path (pathlib.Path): StandardScalar 模型 (.pkl) 的完整保存路径。

    输出:
        tuple[np.ndarray, np.ndarray]: (X_train_scaled, X_test_scaled)
                                       经过标准化的训练集和测试集。
    """
    scaler = StandardScaler()

    # 在训练数据上拟合 scaler 并转换训练数据
    X_train_scaled = scaler.fit_transform(X_train)

    # 使用同一个 scaler 转换测试数据
    X_test_scaled = scaler.transform(X_test)

    # 保存拟合后的 scaler
    # 确保 scaler_save_path 的父目录存在 (通常由主调用程序负责创建)
    scaler_save_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
    joblib.dump(scaler, scaler_save_path)
    print(f"StandardScaler 已拟合训练数据并保存至: {scaler_save_path}")

    return X_train_scaled, X_test_scaled


def split_and_save_data(X_all: np.ndarray, y_all: np.ndarray, 
                        train_samples: int, processed_data_dir: Path):
    """
    将完整数据集划分为训练集和测试集，并分别保存到指定目录。
    文件将以 .npy 格式保存 (例如 X_train.npy, y_train.npy 等)。

    输入:
        X_all (np.ndarray): 完整的特征数据集。
        y_all (np.ndarray): 完整的标签数据集。
        train_samples (int): 用于训练集的样本数量。
        processed_data_dir (pathlib.Path): 保存预处理后数据的目录路径。
    """
    if X_all.shape[0] != y_all.shape[0]:
        raise ValueError("X_all 和 y_all 的样本数量必须一致。")
    if X_all.shape[0] < train_samples:
        raise ValueError(f"总样本数 {X_all.shape[0]} 少于请求的训练样本数 {train_samples}。")

    # 确保输出目录存在
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    X_train = X_all[:train_samples]
    y_train = y_all[:train_samples]
    X_test = X_all[train_samples:]
    y_test = y_all[train_samples:]

    # 定义保存的文件名
    path_X_train = processed_data_dir / "X_train.npy"
    path_y_train = processed_data_dir / "y_train.npy"
    path_X_test = processed_data_dir / "X_test.npy"
    path_y_test = processed_data_dir / "y_test.npy"

    # 保存数据
    np.save(path_X_train, X_train)
    np.save(path_y_train, y_train)
    np.save(path_X_test, X_test)
    np.save(path_y_test, y_test)

    print(f"数据已划分为训练集 ({X_train.shape[0]} 条) 和测试集 ({X_test.shape[0]} 条)。")
    print(f"并已保存至目录: {processed_data_dir}")



# ==========================
# 测试程序 (Test Harness)
# ==========================
if __name__ == "__main__":
    print("="*60)
    print("开始在 processor.py 中直接测试数据处理流程...")
    print("注意：此测试块尝试动态调整 sys.path 以便进行相对导入。")
    print("如果遇到导入错误，更推荐的测试方式是从项目根目录运行：")
    print("  python -m src.data_preprocess.processor")
    print("="*60 + "\n")

    import sys
    import traceback # 用于打印详细的错误堆栈

    # --- 动态路径调整 ---
    current_file_path = Path(__file__).resolve()
    # processor.py 路径: PROJECT_ROOT/src/data_preprocess/processor.py
    # project_root_for_test 是 PROJECT_ROOT
    project_root_for_test = current_file_path.parent.parent.parent 
    if str(project_root_for_test) not in sys.path:
        sys.path.insert(0, str(project_root_for_test))
        print(f"信息: 已将项目根目录 '{project_root_for_test}' 添加到 sys.path 进行测试。\n")

    # --- 延迟导入，确保 sys.path 已设置 ---
    try:
        from src.data_preprocess.loader import (
            download_mnist_dataset, 
            load_raw_mnist_data,
            load_processed_train_data,
            load_processed_test_data
        )
        from src import config # 导入配置以获取路径
    except ImportError as e:
        print(f"CRITICAL: 测试所需的模块导入失败: {e}")
        print("请确保 src/config.py 和 src/data_preprocess/loader.py 文件存在且路径正确。")
        print("也请检查 PYTHONPATH 或从项目根目录运行此脚本。")
        sys.exit(1) # 导入失败则无法继续测试

    # --- 定义常量和路径 ---
    # 从 config 中获取路径，并转换为 Path 对象
    try:
        PROCESSED_DIR = Path(config.PROCESSED_DATA_DIR)
        UTILS_DIR = Path(config.UTILS_DIR) # 确保在 config.py 中定义了 UTILS_DIR
    except AttributeError as e:
        print(f"CRITICAL: config.py 中缺少必要的路径定义 (PROCESSED_DATA_DIR 或 UTILS_DIR): {e}")
        print("请确保在 src/config.py 中定义了 PROCESSED_DATA_DIR 和 UTILS_DIR。")
        sys.exit(1)

    SCALER_FILENAME = "mnist_standard_scaler_test.pkl" # 测试用的scaler文件名
    SCALER_SAVE_PATH = UTILS_DIR / SCALER_FILENAME
    TRAIN_SAMPLE_COUNT = 60000 # MNIST 标准训练集大小

    # --- 主测试逻辑 ---
    try:
        print("--- 步骤 0: 确保必要的目录存在 ---")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        UTILS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"预处理数据目录: {PROCESSED_DIR}")
        print(f"工具/模型目录: {UTILS_DIR}\n")

        print("--- 步骤 1: 调用 download_mnist_dataset() 确保原始数据已下载 ---")
        download_mnist_dataset()
        print("--- download_mnist_dataset() 调用完成 ---\n")

        print("--- 步骤 2: 调用 load_raw_mnist_data() 加载完整原始数据 ---")
        X_all, y_all = load_raw_mnist_data()
        print(f"原始数据加载完成: X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
        print("--- load_raw_mnist_data() 调用完成 ---\n")

        print("--- 步骤 3: 调用 split_and_save_data() 分割并保存数据 ---")
        split_and_save_data(X_all, y_all, 
                            train_samples=TRAIN_SAMPLE_COUNT, 
                            processed_data_dir=PROCESSED_DIR)
        print("--- split_and_save_data() 调用完成 ---\n")

        print("--- 步骤 4: 调用 load_processed_train_data() 加载预处理后的训练数据 ---")
        X_train_loaded, y_train_loaded = load_processed_train_data(processed_data_dir=PROCESSED_DIR)
        print(f"预处理训练数据加载完成: X_train_loaded shape: {X_train_loaded.shape}, y_train_loaded shape: {y_train_loaded.shape}")
        print("--- load_processed_train_data() 调用完成 ---\n")

        print("--- 步骤 5: 调用 load_processed_test_data() 加载预处理后的测试数据 ---")
        X_test_loaded, y_test_loaded = load_processed_test_data(processed_data_dir=PROCESSED_DIR)
        print(f"预处理测试数据加载完成: X_test_loaded shape: {X_test_loaded.shape}, y_test_loaded shape: {y_test_loaded.shape}")
        print("--- load_processed_test_data() 调用完成 ---\n")

        print("--- 步骤 6: 调用 standardscalar_processor() 标准化数据并保存 scaler ---")
        X_train_scaled, X_test_scaled = standardscalar_processor(
            X_train_loaded, X_test_loaded, scaler_save_path=SCALER_SAVE_PATH
        )
        print(f"数据标准化完成: X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")
        print(f"Scaler 应已保存至: {SCALER_SAVE_PATH}")
        print("--- standardscalar_processor() 调用完成 ---\n")

        print("--- 步骤 7: 验证保存的 scaler ---")
        if SCALER_SAVE_PATH.exists():
            loaded_scaler = joblib.load(SCALER_SAVE_PATH)
            if isinstance(loaded_scaler, StandardScaler):
                print(f"Scaler 成功从 {SCALER_SAVE_PATH} 加载，且为 StandardScaler 实例。")
                # 简单打印 scaler 的一些属性，例如均值（如果存在且特征数不多）
                if hasattr(loaded_scaler, 'mean_') and loaded_scaler.mean_ is not None:
                     print(f"  Scaler 均值 (前5个特征): {loaded_scaler.mean_[:5]}")
                else:
                     print(f"  Scaler 均值未记录或不可用 (可能因为 fit 还未在实际数据上完成或数据特殊)。")

                # 尝试使用加载的 scaler 转换一小部分数据 (可选验证)
                # X_test_verify_transform = loaded_scaler.transform(X_test_loaded[:5]) 
                # print(f"  使用加载的 scaler 转换测试集前5条的结果 (前3特征): \n{X_test_verify_transform[:, :3]}")
            else:
                print(f"错误: 从 {SCALER_SAVE_PATH} 加载的对象不是 StandardScaler 实例。类型为: {type(loaded_scaler)}")
        else:
            print(f"错误: 未找到已保存的 scaler 文件于 {SCALER_SAVE_PATH}")
        print("--- Scaler 验证完成 ---\n")

        print("="*60)
        print("processor.py 测试流程成功结束!")
        print("="*60)

    except FileNotFoundError as e_fnf:
        print("\n" + "="*20 + " 测试因文件未找到而失败 " + "="*20)
        print(f"错误信息: {e_fnf}")
        print("请检查相应的数据文件或目录是否存在，以及路径配置是否正确。")
        print("特别是 PROCESSED_DATA_DIR 和 UTILS_DIR 是否在 config.py 中正确设置并可访问。")
        print("\n详细错误追溯信息:")
        traceback.print_exc()
        print("="*60)
    except Exception as e:
        print("\n" + "="*20 + " 测试失败 " + "="*20)
        print(f"错误类型: {type(e).__name__}")
        print(f"实际错误信息: {e}")
        print("\n详细错误追溯信息:")
        traceback.print_exc()
        print("="*60)