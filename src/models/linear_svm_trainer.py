# src/models/linear_svm_trainer.py
from sklearn.svm import SVC
import joblib
from pathlib import Path
import time
import numpy as np
from src import config

def train_linear_svm(X_train: np.ndarray, y_train: np.ndarray, 
                     model_save_path: Path, C_param: float = 1.0) -> tuple[SVC, float]:
    """
    训练线性核SVM模型并保存。

    输入:
        X_train (np.ndarray): 训练集特征。
        y_train (np.ndarray): 训练集标签。
        model_save_path (pathlib.Path): 训练好的模型的完整保存路径。
        C_param (float): SVM的正则化参数C。

    输出:
        tuple[SVC, float]: (trained_model, training_time)
                           训练好的SVM模型对象和训练时长（秒）。
    """
    print(f"\n--- 开始训练线性SVM ---")
    print(f"参数: C={C_param}")
    print(f"训练数据规模: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 初始化线性SVM模型
    # probability=True 可以获取概率估计，但会增加训练时间。如果不需要，可以设为False。
    # random_state 用于结果可复现。
    model = SVC(kernel='linear', C=C_param, probability=True, random_state=config.RANDOM_STATE) 
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 计算训练时长
    training_time = time.time() - start_time
    
    print(f"线性SVM训练完成，耗时: {training_time:.4f} 秒。")

    # 保存模型
    try:
        model_save_path.parent.mkdir(parents=True, exist_ok=True) # 确保保存目录存在
        joblib.dump(model, model_save_path)
        print(f"线性SVM模型已成功保存至: {model_save_path}")
    except Exception as e:
        print(f"错误：保存线性SVM模型失败: {e}")
        # 根据需要，可以选择是否在此处重新抛出异常
        # raise
    
    return model, training_time

if __name__ == '__main__':
    # --- 这是一个简单的测试代码块，实际使用时会在 run_task1_svm.py 中调用 ---
    print(">>> 测试 train_linear_svm 函数 (这是一个占位测试，请在主脚本中集成) <<<")
    
    # 模拟一些数据
    # 注意：在实际项目中，数据加载和预处理由外部完成
    print("正在生成模拟数据进行测试...")
    X_train_sample = np.random.rand(100, 10) # 100个样本，10个特征
    y_train_sample = np.random.randint(0, 2, 100) # 二分类标签
    
    # 定义模型保存路径 (实际路径应从 config.py 获取或在主脚本中定义)
    # 为了独立测试，这里临时定义一个
    from src import config # 假设可以这样导入，或者调整路径
    
    # 确保 TRAINED_MODEL_DIR 在 config 中定义
    if hasattr(config, 'TRAINED_MODEL_DIR'):
        test_model_dir = Path(config.TRAINED_MODEL_DIR) / "test_outputs"
    else: # 回退到本地目录
        print("警告: config.py 中未找到 TRAINED_MODEL_DIR，测试模型将保存在本地 'temp_models' 目录。")
        test_model_dir = Path("./temp_models") # 临时目录
        
    test_model_dir.mkdir(parents=True, exist_ok=True)
    sample_model_path = test_model_dir / "test_linear_svm.pkl"
    
    print(f"模拟训练数据 X shape: {X_train_sample.shape}")
    print(f"模拟训练标签 y shape: {y_train_sample.shape}")
    print(f"测试模型将保存到: {sample_model_path}")

    try:
        trained_model, time_taken = train_linear_svm(X_train_sample, y_train_sample, 
                                                     model_save_path=sample_model_path, 
                                                     C_param=0.1)
        
        print("\n测试函数调用成功！")
        print(f"返回的模型对象类型: {type(trained_model)}")
        print(f"记录的训练时间: {time_taken:.4f} 秒")
        
        # 检查模型是否已保存
        if sample_model_path.exists():
            print(f"模型文件 {sample_model_path} 已成功创建。")
            # (可选) 尝试加载模型
            loaded_model = joblib.load(sample_model_path)
            print(f"尝试重新加载模型成功，类型: {type(loaded_model)}")
        else:
            print(f"错误: 模型文件 {sample_model_path} 未找到！")
            
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