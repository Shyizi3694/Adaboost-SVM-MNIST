# src/models/rbf_svm_trainer.py
from sklearn.svm import SVC
import joblib
from pathlib import Path
import time
import numpy as np
from src import config

def train_rbf_svm(X_train: np.ndarray, y_train: np.ndarray, 
                  model_save_path: Path, 
                  C_param: float = 1.0, gamma_param: str | float = 'scale') -> tuple[SVC, float]:
    """
    训练RBF核SVM模型并保存。

    输入:
        X_train (np.ndarray): 训练集特征。
        y_train (np.ndarray): 训练集标签。
        model_save_path (pathlib.Path): 训练好的模型的完整保存路径。
        C_param (float): SVM的正则化参数C。
        gamma_param (str | float): RBF核的gamma参数。可以是 'scale', 'auto' 或一个浮点数值。

    输出:
        tuple[SVC, float]: (trained_model, training_time)
                           训练好的SVM模型对象和训练时长（秒）。
    """
    print(f"\n--- 开始训练RBF核SVM ---")
    print(f"参数: C={C_param}, gamma={gamma_param}")
    print(f"训练数据规模: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 初始化RBF核SVM模型
    # probability=True 可以获取概率估计，但会增加训练时间。
    # random_state 用于结果可复现。
    model = SVC(kernel='rbf', C=C_param, gamma=gamma_param, probability=True, random_state=config.RANDOM_STATE)
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 计算训练时长
    training_time = time.time() - start_time
    
    print(f"RBF核SVM训练完成，耗时: {training_time:.4f} 秒。")

    # 保存模型
    try:
        model_save_path.parent.mkdir(parents=True, exist_ok=True) # 确保保存目录存在
        joblib.dump(model, model_save_path)
        print(f"RBF核SVM模型已成功保存至: {model_save_path}")
    except Exception as e:
        print(f"错误：保存RBF核SVM模型失败: {e}")
        # raise
            
    return model, training_time

if __name__ == '__main__':
    # --- 这是一个简单的测试代码块，实际使用时会在 run_task1_svm.py 中调用 ---
    print(">>> 测试 train_rbf_svm 函数 (这是一个占位测试，请在主脚本中集成) <<<")
    
    print("正在生成模拟数据进行测试...")
    X_train_sample = np.random.rand(100, 10)
    y_train_sample = np.random.randint(0, 2, 100)
    
    from src import config # 假设可以这样导入
    
    if hasattr(config, 'TRAINED_MODEL_DIR'):
        test_model_dir = Path(config.TRAINED_MODEL_DIR) / "test_outputs"
    else:
        print("警告: config.py 中未找到 TRAINED_MODEL_DIR，测试模型将保存在本地 'temp_models' 目录。")
        test_model_dir = Path("./temp_models")
        
    test_model_dir.mkdir(parents=True, exist_ok=True)
    sample_model_path = test_model_dir / "test_rbf_svm.pkl"

    print(f"模拟训练数据 X shape: {X_train_sample.shape}")
    print(f"模拟训练标签 y shape: {y_train_sample.shape}")
    print(f"测试模型将保存到: {sample_model_path}")

    try:
        # 测试使用默认的 'scale' 作为 gamma 值
        trained_model, time_taken = train_rbf_svm(X_train_sample, y_train_sample, 
                                                  model_save_path=sample_model_path, 
                                                  C_param=1.0, gamma_param='scale')
        
        print("\n测试函数调用成功 (gamma='scale')！")
        print(f"返回的模型对象类型: {type(trained_model)}")
        print(f"记录的训练时间: {time_taken:.4f} 秒")
        
        if sample_model_path.exists():
            print(f"模型文件 {sample_model_path} 已成功创建。")
            loaded_model = joblib.load(sample_model_path)
            print(f"尝试重新加载模型成功，类型: {type(loaded_model)}")
        else:
            print(f"错误: 模型文件 {sample_model_path} 未找到！")
            
        # 测试使用具体的 gamma 值
        sample_model_path_custom_gamma = test_model_dir / "test_rbf_svm_custom_gamma.pkl"
        print(f"\n测试自定义 gamma (例如 0.1)，模型将保存到: {sample_model_path_custom_gamma}")
        trained_model_custom, time_taken_custom = train_rbf_svm(X_train_sample, y_train_sample, 
                                                                model_save_path=sample_model_path_custom_gamma, 
                                                                C_param=1.0, gamma_param=0.1)
        print("\n测试函数调用成功 (gamma=0.1)！")
        print(f"返回的模型对象类型: {type(trained_model_custom)}")
        print(f"记录的训练时间: {time_taken_custom:.4f} 秒")
        if sample_model_path_custom_gamma.exists():
            print(f"模型文件 {sample_model_path_custom_gamma} 已成功创建。")
        else:
            print(f"错误: 模型文件 {sample_model_path_custom_gamma} 未找到！")

    except ImportError:
        print("错误: 无法导入 'src.config'。请确保您在项目根目录下运行，或者 PYTHONPATH 设置正确。")
        print("      或者，确保 src/config.py 文件存在。")
    except AttributeError as e:
        if 'TRAINED_MODEL_DIR' in str(e):
             print(f"错误: 请确保 TRAINED_MODEL_DIR 在 src/config.py 中定义。")
        else:
            print(f"发生属性错误: {e}")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()