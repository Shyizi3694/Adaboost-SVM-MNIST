# src/data_preprocess/loader.py

# thirdparty libraries
import urllib.request
import urllib.error
import os
import shutil
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
import sklearn.datasets._base as sklearn_base
import joblib  # 用于加载 joblib 保存的对象
from typing import Any  # 用于 load_model 的返回类型提示
from sklearn.preprocessing import StandardScaler

# local libraries
from .. import config


# --- 下载确保函数 ---
def download_mnist_dataset():
    """
    确保 MNIST 数据集 (mnist_784) 已下载并缓存到本地。
    如果数据集不在缓存中，则触发下载并显示进度条（如果补丁成功）。
    此函数不返回数据集本身，其主要目的是确保数据在本地可用。
    """
    raw_data_path = str(config.RAW_DATA_DIR)
    print(f"检查 MNIST 数据集状态于: {raw_data_path}")

    original_request_fn = getattr(sklearn_base, '_request_url_and_write_file', None)
    patch_applied = False

    if original_request_fn:
        def patched_request_url_and_write_file(remote, file_path, *, chunk_size=8192):
            # (tqdm 补丁函数的具体实现 - 和之前一样)
            req = urllib.request.Request(remote.url, headers={'User-agent': 'scikit-learn'})
            try:
                with urllib.request.urlopen(req) as C_response, open(file_path, "wb") as F_dest:
                    content_length = C_response.headers.get('Content-Length')
                    total_size = None
                    if content_length:
                        try:
                            total_size = int(content_length.strip())
                        except ValueError:
                            print(f"警告: 无法解析 Content-Length 用于进度条: {content_length}")
                            total_size = None
                    
                    desc = os.path.basename(file_path)
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc, leave=False) as pbar:
                        wrapped_file = CallbackIOWrapper(pbar.update, F_dest, "write")
                        shutil.copyfileobj(C_response, wrapped_file, length=chunk_size)
            except urllib.error.URLError:
                raise

        sklearn_base._request_url_and_write_file = patched_request_url_and_write_file
        patch_applied = True
        # print("信息: 已应用 tqdm 进度条补丁。")
    else:
        print("警告: 未能找到 scikit-learn 的内部下载函数用于进度条。\n"
              "      如果发生下载，过程将不显示详细进度条。")

    try:
        if patch_applied:
            print(f"信息: 尝试确保 MNIST 数据集 (mnist_784) 可用 (带进度条补丁)...")
        else:
            print(f"信息: 尝试确保 MNIST 数据集 (mnist_784) 可用...")

        # 调用 fetch_openml 主要是为了其下载和缓存的副作用
        # 我们不需要它返回的数据
        fetch_openml(
            'mnist_784',
            version=1,
            as_frame=False, # 仍然需要指定，即使我们不直接使用返回的 Bunch
            data_home=raw_data_path,
            parser='liac-arff'
        )
        print(f"MNIST 数据集已在 '{raw_data_path}' 中可用或已成功下载。")

    except Exception as e:
        print(f"在确保 MNIST 数据集可用性时发生错误: {type(e).__name__} - {e}")
        raise
    finally:
        if patch_applied and original_request_fn:
            sklearn_base._request_url_and_write_file = original_request_fn
            # print("信息: 已恢复 scikit-learn 原始下载函数。")


# --- 数据加载函数 ---
def load_raw_mnist_data():
    """
    从本地缓存加载完整的原始 MNIST 数据集 (mnist_784) 为 NumPy 数组。
    假定数据集已通过 download_mnist_dataset() 确保在本地存在。
    如果缓存不存在，fetch_openml 仍会尝试下载。

    返回:
        tuple: (X, y)
               X (numpy.ndarray): 图像数据 (70000, 784), float32
               y (numpy.ndarray): 标签数据 (70000,), uint8
    """
    raw_data_path = str(config.RAW_DATA_DIR)
    print(f"从本地缓存加载 MNIST 数据集: {raw_data_path}...")
    try:
        mnist_bunch = fetch_openml(
            'mnist_784',
            version=1,
            as_frame=False,
            data_home=raw_data_path, # fetch_openml 会从此路径加载（或下载后加载）
            parser='liac-arff'
        )
        X = mnist_bunch.data.astype(np.float32)
        y = mnist_bunch.target.astype(np.uint8)
        print("原始 MNIST 数据集已成功从缓存加载。")
        return X, y
    except Exception as e:
        print(f"从缓存加载 MNIST 数据集时发生错误: {type(e).__name__} - {e}")
        print("请确保首先调用 download_mnist_dataset() 成功下载了数据集。")
        raise


# --- 训练集加载函数 ---
def load_processed_train_data(processed_data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    从指定目录加载预处理后的训练数据 (X_train.npy, y_train.npy)。
    """
    path_X_train = processed_data_dir / "X_train.npy"
    path_y_train = processed_data_dir / "y_train.npy"

    if not path_X_train.exists() or not path_y_train.exists():
        raise FileNotFoundError(f"在 {processed_data_dir} 中未找到预处理的训练数据文件。请先运行数据分割和保存步骤。")

    X_train = np.load(path_X_train)
    y_train = np.load(path_y_train)
    print(f"已从 {processed_data_dir} 加载预处理的训练数据。")
    return X_train, y_train


# --- 测试集加载函数 -
def load_processed_test_data(processed_data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    从指定目录加载预处理后的测试数据 (X_test.npy, y_test.npy)。
    """
    path_X_test = processed_data_dir / "X_test.npy"
    path_y_test = processed_data_dir / "y_test.npy"

    if not path_X_test.exists() or not path_y_test.exists():
        raise FileNotFoundError(f"在 {processed_data_dir} 中未找到预处理的测试数据文件。请先运行数据分割和保存步骤。")

    X_test = np.load(path_X_test)
    y_test = np.load(path_y_test)
    print(f"已从 {processed_data_dir} 加载预处理的测试数据。")
    return X_test, y_test


# --- 模型加载函数 ---
def load_model(model_path: Path) -> Any:
    """
    从指定的 .pkl (joblib) 文件加载已训练的模型。

    输入:
        model_path (pathlib.Path): 模型文件的完整路径。

    输出:
        Any: 加载的模型对象。返回类型为 Any，因为可以是任何类型的模型 (例如 SVC, 自定义的 AdaBoost 等)。
    
    异常:
        FileNotFoundError: 如果模型文件不存在。
        Exception: 如果在加载过程中发生其他错误 (例如文件损坏)。
    """
    print(f"尝试从路径加载模型: {model_path}")
    if not model_path.exists():
        print(f"错误: 模型文件 {model_path} 未找到。")
        raise FileNotFoundError(f"无法加载模型：文件 {model_path} 不存在。")
    
    try:
        model = joblib.load(model_path)
        print(f"模型已成功从 {model_path} 加载。")
        return model
    except Exception as e:
        print(f"错误: 从 {model_path} 加载模型失败。详细信息: {e}")
        # 根据需要，可以选择是否将原始异常再次抛出，或者抛出自定义异常
        raise 


# --- scalar 加载函数 ---
def load_scaler(scaler_path: Path) -> StandardScaler:
    """
    从指定的 .pkl (joblib) 文件加载已保存的 scaler 对象。
    当前设计假定加载的是一个 StandardScaler 对象。

    输入:
        scaler_path (pathlib.Path): scaler 文件的完整路径。

    输出:
        StandardScaler: 加载的 scaler 对象。
    
    异常:
        FileNotFoundError: 如果 scaler 文件不存在。
        Exception: 如果在加载过程中发生其他错误 (例如文件损坏或类型不匹配)。
    """
    print(f"尝试从路径加载 scaler: {scaler_path}")
    if not scaler_path.exists():
        print(f"错误: Scaler 文件 {scaler_path} 未找到。")
        raise FileNotFoundError(f"无法加载 scaler：文件 {scaler_path} 不存在。")
        
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler 已成功从 {scaler_path} 加载。")
        
        # (可选) 进行类型检查，确保加载的是预期的对象类型
        if not isinstance(scaler, StandardScaler):
            print(f"警告: 从 {scaler_path} 加载的对象类型为 {type(scaler).__name__}, 而预期为 StandardScaler。")
            # 根据严格程度，您可以在这里抛出 TypeError
            # raise TypeError(f"加载的 scaler 类型不正确，预期 StandardScaler，得到 {type(scaler).__name__}")
            
        return scaler
    except Exception as e:
        print(f"错误: 从 {scaler_path} 加载 scaler 失败。详细信息: {e}")
        raise


# # ==========================
# # 测试程序 ( Test for download_mnist_dataset() & load_raw_mnist_data() & load_processed_train_data() & load_processed_test_data() )
# # ==========================
# if __name__ == "__main__":
#     print("="*50)
#     print("开始在 loader.py 中测试数据处理函数...")
#     # (sys.path 修改逻辑 - 和之前一样)
#     import sys
#     from pathlib import Path
#     import traceback

#     current_file_path = Path(__file__).resolve()
#     project_root_for_test = current_file_path.parent.parent.parent
#     if str(project_root_for_test) not in sys.path:
#         sys.path.insert(0, str(project_root_for_test))
#         print(f"信息: 已将项目根目录 '{project_root_for_test}' 添加到 sys.path 进行测试。\n")
#     # --- 测试流程 ---
#     try:
#         # 步骤 1: 确保数据集已下载
#         print("\n--- 步骤 1: 调用 download_mnist_dataset() ---")
#         download_mnist_dataset()
#         print("--- download_mnist_dataset() 调用完成 ---\n")

#         # 步骤 2: 从缓存加载原始数据
#         print("\n--- 步骤 2: 调用 load_raw_mnist_data() ---")
#         X_data, y_data = load_raw_mnist_data()
#         print("--- load_raw_mnist_data() 调用完成 ---\n")
        
#         # 打印加载的数据信息
#         print("\n" + "="*20 + " 加载数据测试结果 " + "="*20)
#         print(f"返回的 X 数据形状: {X_data.shape}, 数据类型: {X_data.dtype}")
#         print(f"返回的 y 数据形状: {y_data.shape}, 数据类型: {y_data.dtype}")
        
#         if X_data.size > 0 and y_data.size > 0:
#             print(f"\nX 数据样本 (前3行, 每行前10个像素):")
#             print(X_data[:3, :10])
#             print(f"\ny 数据样本 (前10个标签):")
#             print(y_data[:10])
#             print(f"\nX 数据统计: 最小值={X_data.min():.2f}, 最大值={X_data.max():.2f}, 平均值={X_data.mean():.2f}")
#             print(f"y 标签中的唯一值: {np.unique(y_data)}")
#         else:
#             print("\n警告: 加载的数据 X 或 y 为空或大小不正确。")
#         print("="*62)

#     except Exception as e:
#         print("\n" + "="*20 + " 测试失败 " + "="*20)
#         print(f"错误类型: {type(e).__name__}")
#         actual_error_message = str(e)
#         print(f"实际错误信息: {actual_error_message}")
#         # (更详细的 ImportError 提示 - 和之前一样)
#         if isinstance(e, ImportError):
#             if "pandas" in actual_error_message:
#                 print("提示: 此导入错误与 pandas 库有关...")
#             elif "No module named 'src.config'" in actual_error_message or \
#                  "attempted relative import beyond top-level package" in actual_error_message or \
#                  "Parent module '' not loaded, cannot perform relative import" in actual_error_message:
#                 print("提示: 此导入错误可能与找不到 'src.config' 模块有关...")
#         print("\n详细错误追溯信息:")
#         traceback.print_exc()
#         print("="*52)





# # =====================================
# # 测试 load_model 和 load_scaler 函数
# # =====================================
# if __name__ == '__main__':
#     import sys
#     import traceback
#     # 为了测试，我们需要创建一些虚拟对象并保存它们
#     # 这些导入仅用于 __main__ 测试块
#     from sklearn.preprocessing import StandardScaler 
#     from sklearn.svm import SVC
#     import numpy as np # 用于创建少量虚拟数据（可选）

#     print("="*60)
#     print("开始在 loader.py 中测试 load_model 和 load_scaler 函数...")
#     print("注意：此测试块尝试动态调整 sys.path 以便进行相对导入。")
#     print("如果遇到与 config 相关的导入错误，更推荐从项目根目录运行：")
#     print("  python -m src.data_preprocess.loader")
#     print("="*60 + "\n")

#     # --- 动态路径调整 ---
#     current_file_path = Path(__file__).resolve()
#     project_root_for_test = current_file_path.parent.parent.parent 
#     if str(project_root_for_test) not in sys.path:
#         sys.path.insert(0, str(project_root_for_test))
#         print(f"信息: 已将项目根目录 '{project_root_for_test}' 添加到 sys.path 进行测试。\n")

#     # --- 延迟导入 config，确保 sys.path 已设置 ---
#     try:
#         from src import config # 导入配置以获取路径
#     except ImportError as e:
#         print(f"CRITICAL: 测试所需的 'src.config' 模块导入失败: {e}")
#         print("请确保 src/config.py 文件存在且 PYTHONPATH 或执行路径正确。")
#         sys.exit(1) # 导入失败则无法继续测试

#     # --- 定义测试文件路径 ---
#     TEST_ARTIFACTS_SUBDIR_NAME = "loader_internal_test_artifacts" # 临时子目录名
#     dummy_scaler_path = None
#     dummy_model_path = None
#     temp_dirs_to_clean = []

#     try:
#         # 确保 UTILS_DIR 和 TRAINED_MODEL_DIR 在 config 中定义
#         if not hasattr(config, 'UTILS_DIR') or not hasattr(config, 'TRAINED_MODEL_DIR'):
#             raise AttributeError("config.py 中必须定义 UTILS_DIR 和 TRAINED_MODEL_DIR")

#         DUMMY_SCALER_PARENT_DIR = Path(config.UTILS_DIR)
#         DUMMY_MODEL_PARENT_DIR = Path(config.TRAINED_MODEL_DIR)

#         DUMMY_SCALER_DIR = DUMMY_SCALER_PARENT_DIR / TEST_ARTIFACTS_SUBDIR_NAME
#         DUMMY_MODEL_DIR = DUMMY_MODEL_PARENT_DIR / TEST_ARTIFACTS_SUBDIR_NAME
        
#         DUMMY_SCALER_DIR.mkdir(parents=True, exist_ok=True)
#         DUMMY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
#         temp_dirs_to_clean.extend([DUMMY_SCALER_DIR, DUMMY_MODEL_DIR]) # 记录以便清理

#         dummy_scaler_path = DUMMY_SCALER_DIR / "dummy_scaler_for_loader_test.pkl"
#         dummy_model_path = DUMMY_MODEL_DIR / "dummy_model_for_loader_test.pkl"
#         non_existent_path = DUMMY_SCALER_DIR / "this_file_does_not_exist.pkl"

#     except AttributeError as e:
#         print(f"CRITICAL: config.py 中缺少必要的路径定义: {e}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"CRITICAL: 创建测试目录或定义路径时出错: {e}")
#         sys.exit(1)

#     # --- 准备虚拟对象 ---
#     scaler_to_save = StandardScaler()
#     # 可选：在一些虚拟数据上拟合 scaler，使其不完全是“空”的
#     # scaler_to_save.fit(np.array([[10.0],[20.0],[30.0]]))
    
#     model_to_save = SVC(kernel='linear', C=0.001, random_state=42) 
#     # 可选：在一些虚拟数据上拟合模型
#     # model_to_save.fit(np.array([[1,1],[2,2],[3,3]]), np.array([0,0,1]))

#     test_passed_count = 0
#     test_failed_count = 0
#     total_tests = 0

#     try:
#         print(f"\n--- 开始针对 loader.py 的 load_scaler 和 load_model 进行专项测试 ---")

#         # 1. 保存虚拟 scaler 和 model
#         print(f"\n步骤 1: 保存虚拟对象...")
#         total_tests += 2
#         try:
#             joblib.dump(scaler_to_save, dummy_scaler_path)
#             print(f"  (+) 虚拟 scaler 已保存至: {dummy_scaler_path}")
#             test_passed_count += 1
#         except Exception as e_save_scaler:
#             print(f"  (-) 错误: 保存虚拟 scaler 失败: {e_save_scaler}")
#             test_failed_count += 1
        
#         try:
#             joblib.dump(model_to_save, dummy_model_path)
#             print(f"  (+) 虚拟 model 已保存至: {dummy_model_path}")
#             test_passed_count += 1
#         except Exception as e_save_model:
#             print(f"  (-) 错误: 保存虚拟 model 失败: {e_save_model}")
#             test_failed_count += 1
        
#         if test_failed_count > 0 and (not dummy_scaler_path.exists() or not dummy_model_path.exists()):
#              print("由于保存步骤失败，后续加载测试可能无法正确执行。")
#              # 可以选择在此处停止测试，或者继续尝试（load会失败）


#         # 2. 测试 load_scaler (成功案例)
#         print(f"\n步骤 2: 测试 load_scaler (成功案例)...")
#         total_tests += 1
#         if dummy_scaler_path.exists(): # 仅当保存成功时尝试加载
#             try:
#                 loaded_scaler = load_scaler(dummy_scaler_path) # 调用本文件中的函数
#                 if isinstance(loaded_scaler, StandardScaler):
#                     print(f"  (+) 成功: load_scaler 返回了 StandardScaler 实例。")
#                     test_passed_count += 1
#                 else:
#                     print(f"  (-) 失败: load_scaler 返回类型为 {type(loaded_scaler)}, 预期 StandardScaler。")
#                     test_failed_count += 1
#             except Exception as e_ls:
#                 print(f"  (-) 失败: load_scaler 引发异常: {e_ls}")
#                 test_failed_count += 1
#         else:
#             print(f"  (!) 跳过: 虚拟 scaler 文件不存在，无法测试加载。")
#             test_failed_count +=1 # 因为保存步骤应该成功


#         # 3. 测试 load_model (成功案例)
#         print(f"\n步骤 3: 测试 load_model (成功案例)...")
#         total_tests += 1
#         if dummy_model_path.exists(): # 仅当保存成功时尝试加载
#             try:
#                 loaded_model = load_model(dummy_model_path) # 调用本文件中的函数
#                 if isinstance(loaded_model, SVC): 
#                     print(f"  (+) 成功: load_model 返回了 SVC 实例。")
#                     test_passed_count += 1
#                 else:
#                     print(f"  (-) 失败: load_model 返回类型为 {type(loaded_model)}, 预期 SVC。")
#                     test_failed_count += 1
#             except Exception as e_lm:
#                 print(f"  (-) 失败: load_model 引发异常: {e_lm}")
#                 test_failed_count += 1
#         else:
#             print(f"  (!) 跳过: 虚拟 model 文件不存在，无法测试加载。")
#             test_failed_count += 1 # 因为保存步骤应该成功

#         # 4. 测试 load_scaler (文件不存在)
#         print(f"\n步骤 4: 测试 load_scaler (文件不存在)...")
#         total_tests += 1
#         try:
#             load_scaler(non_existent_path)
#             print(f"  (-) 失败: load_scaler 未对不存在的文件引发 FileNotFoundError。")
#             test_failed_count += 1
#         except FileNotFoundError:
#             print(f"  (+) 成功: load_scaler 对不存在的文件正确引发了 FileNotFoundError。")
#             test_passed_count += 1
#         except Exception as e_ls_fnf:
#             print(f"  (-) 失败: load_scaler 对不存在的文件引发了意外的异常: {e_ls_fnf}")
#             test_failed_count += 1
            
#         # 5. 测试 load_model (文件不存在)
#         print(f"\n步骤 5: 测试 load_model (文件不存在)...")
#         total_tests += 1
#         try:
#             load_model(non_existent_path)
#             print(f"  (-) 失败: load_model 未对不存在的文件引发 FileNotFoundError。")
#             test_failed_count += 1
#         except FileNotFoundError:
#             print(f"  (+) 成功: load_model 对不存在的文件正确引发了 FileNotFoundError。")
#             test_passed_count += 1
#         except Exception as e_lm_fnf:
#             print(f"  (-) 失败: load_model 对不存在的文件引发了意外的异常: {e_lm_fnf}")
#             test_failed_count += 1

#     except Exception as e_global:
#         print(f"\n测试过程中发生严重错误: {type(e_global).__name__} - {e_global}")
#         traceback.print_exc()
#         # 假设所有未完成的测试都失败
#         test_failed_count = total_tests - test_passed_count 
#     finally:
#         print(f"\n--- 测试清理 ---")
#         # 清理虚拟文件
#         if dummy_scaler_path and dummy_scaler_path.exists():
#             try:
#                 dummy_scaler_path.unlink()
#                 print(f"  已删除虚拟 scaler 文件: {dummy_scaler_path}")
#             except Exception as e_del_s:
#                 print(f"  错误: 删除虚拟 scaler 文件失败: {e_del_s}")
#         if dummy_model_path and dummy_model_path.exists():
#             try:
#                 dummy_model_path.unlink()
#                 print(f"  已删除虚拟 model 文件: {dummy_model_path}")
#             except Exception as e_del_m:
#                 print(f"  错误: 删除虚拟 model 文件失败: {e_del_m}")
        
#         # 清理临时子目录 (如果它们是空的)
#         for temp_dir_path in reversed(temp_dirs_to_clean): # 反向确保子目录先处理
#             try:
#                 if temp_dir_path.exists() and not any(temp_dir_path.iterdir()):
#                     temp_dir_path.rmdir()
#                     print(f"  已删除空的测试目录: {temp_dir_path}")
#             except Exception as e_rmdir:
#                 print(f"  清理测试目录 {temp_dir_path} 时发生轻微错误: {e_rmdir}")


#     print(f"\n--- 测试总结 ---")
#     print(f"总计测试点: {total_tests}")
#     print(f"通过的测试点: {test_passed_count}")
#     print(f"失败的测试点: {test_failed_count}")
#     if test_failed_count == 0 and test_passed_count == total_tests :
#         print("loader.py 的 load_model/load_scaler 相关测试均通过！")
#     else:
#         print("loader.py 的 load_model/load_scaler 相关测试存在失败或未执行项。")
#     print("="*60)