# src/models/adaboost_trainer.py

import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 假设 CustomOvRAdaboostClassifier 定义在同目录的 custom_adaboost.py 或可以被正确导入
# 如果 CustomOvRAdaboostClassifier 和 CustomBinaryAdaBoostClassifier 在同一个文件:
from .custom_adaboost import CustomOvRAdaboostClassifier
# 如果 CustomOvRAdaboostClassifier 在一个单独的文件，例如 custom_ovr_adaboost.py，则相应修改导入路径
# from .custom_ovr_adaboost import CustomOvRAdaboostClassifier


def train_custom_adaboost_with_stumps(
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        model_save_path: Path,
        n_estimators: int = 100, 
        learning_rate: float = 1.0, 
        random_state: int = None) -> tuple[CustomOvRAdaboostClassifier, float]:
    """
    训练以决策树桩为基学习器的自定义AdaBoost模型 (OvR策略) 并保存。

    输入:
        X_train (np.ndarray): 训练集特征。
        y_train (np.ndarray): 训练集标签。
        model_save_path (pathlib.Path): 训练好的模型的完整保存路径。
        n_estimators (int): AdaBoost中每个二元分类器的基学习器数量。
        learning_rate (float): AdaBoost的学习率。
        random_state (int, optional): 用于复现的随机状态。

    输出:
        tuple[CustomOvRAdaboostClassifier, float]: (trained_model, training_time)
                                                   训练好的AdaBoost模型对象和训练时长（秒）。
    """
    print(f"\n--- 开始训练 AdaBoost (基学习器: 决策树桩) ---")
    print(f"参数: n_estimators={n_estimators}, learning_rate={learning_rate}, random_state={random_state}")
    print(f"训练数据规模: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 1. 创建决策树桩原型
    stump_prototype = DecisionTreeClassifier(max_depth=1, random_state=random_state)

    # 2. 实例化自定义的OvR AdaBoost分类器
    adaboost_model = CustomOvRAdaboostClassifier(
        base_estimator_prototype_for_binary_ada=stump_prototype,
        n_estimators_per_binary_classifier=n_estimators,
        learning_rate_per_binary_classifier=learning_rate,
        random_state_per_classifier=random_state # 传递给内部的二元AdaBoost
    )

    # 3. 记录开始时间并训练模型
    start_time = time.time()
    adaboost_model.fit(X_train, y_train) # fit 方法内部应有详细的打印输出
    training_time = time.time() - start_time
    
    print(f"AdaBoost (决策树桩) 训练完成，总耗时: {training_time:.4f} 秒。")

    # 4. 保存模型
    try:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(adaboost_model, model_save_path)
        print(f"AdaBoost (决策树桩) 模型已成功保存至: {model_save_path}")
    except Exception as e:
        print(f"错误：保存 AdaBoost (决策树桩) 模型失败: {e}")
        # raise # 根据需要选择是否重新抛出异常
            
    return adaboost_model, training_time


def train_custom_adaboost_with_linear_svm(
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        model_save_path: Path,
        n_estimators: int = 10, 
        learning_rate: float = 1.0,
        base_svm_C: float = 0.01,
        base_svm_max_iter: int = 1000, # 为弱SVM添加迭代限制
        random_state: int = None) -> tuple[CustomOvRAdaboostClassifier, float]:
    """
    训练以弱线性SVM为基学习器的自定义AdaBoost模型 (OvR策略) 并保存。

    输入:
        X_train (np.ndarray): 训练集特征。
        y_train (np.ndarray): 训练集标签。
        model_save_path (pathlib.Path): 训练好的模型的完整保存路径。
        n_estimators (int): AdaBoost中每个二元分类器的基学习器数量。
        learning_rate (float): AdaBoost的学习率。
        base_svm_C (float): 线性SVM基学习器的正则化参数C。
        base_svm_max_iter (int): 线性SVM基学习器的最大迭代次数。
        random_state (int, optional): 用于复现的随机状态。

    输出:
        tuple[CustomOvRAdaboostClassifier, float]: (trained_model, training_time)
                                                   训练好的AdaBoost模型对象和训练时长（秒）。
    """
    print(f"\n--- 开始训练 AdaBoost (基学习器: 弱线性SVM) ---")
    print(f"参数: n_estimators={n_estimators}, learning_rate={learning_rate}, "
          f"base_svm_C={base_svm_C}, base_svm_max_iter={base_svm_max_iter}, random_state={random_state}")
    print(f"训练数据规模: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 1. 创建弱线性SVM原型
    svm_prototype = SVC(
        kernel='linear', 
        C=base_svm_C, 
        probability=False, # 通常设为False以加速，除非内部AdaBoost严格需要概率
        max_iter=base_svm_max_iter, 
        random_state=random_state
    )

    # 2. 实例化自定义的OvR AdaBoost分类器
    adaboost_model = CustomOvRAdaboostClassifier(
        base_estimator_prototype_for_binary_ada=svm_prototype,
        n_estimators_per_binary_classifier=n_estimators,
        learning_rate_per_binary_classifier=learning_rate,
        random_state_per_classifier=random_state # 传递给内部的二元AdaBoost
    )

    # 3. 记录开始时间并训练模型
    start_time = time.time()
    adaboost_model.fit(X_train, y_train) # fit 方法内部应有详细的打印输出
    training_time = time.time() - start_time
    
    print(f"AdaBoost (线性SVM) 训练完成，总耗时: {training_time:.4f} 秒。")

    # 4. 保存模型
    try:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(adaboost_model, model_save_path)
        print(f"AdaBoost (线性SVM) 模型已成功保存至: {model_save_path}")
    except Exception as e:
        print(f"错误：保存 AdaBoost (线性SVM) 模型失败: {e}")
        # raise
            
    return adaboost_model, training_time


if __name__ == '__main__':
    # --- adaboost_trainer.py 的简单测试代码块 ---
    print("\n" + "="*70)
    print(">>> 开始测试 adaboost_trainer.py 中的训练函数 <<<")
    print("="*70)

    import sys
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # --- 动态路径调整 (确保能找到 src.config) ---
    try:
        current_file_path = Path(__file__).resolve()
        PROJECT_ROOT_GUESS = current_file_path.parent.parent.parent 
        if str(PROJECT_ROOT_GUESS) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT_GUESS))
            print(f"信息: 已将项目根目录 '{PROJECT_ROOT_GUESS}' 添加到 sys.path 进行测试。\n")
    except Exception as e_path:
        print(f"警告: 尝试动态调整 sys.path 失败: {e_path}")

    # --- 导入 config ---
    try:
        from src import config
        TEST_MODEL_SAVE_DIR = Path(config.TRAINED_MODEL_DIR) / "adaboost_trainer_tests"
    except (ImportError, AttributeError) as e_config:
        print(f"警告: 无法从 src.config 导入 TRAINED_MODEL_DIR (错误: {e_config})。"
              f"测试模型将保存在本地 './temp_trained_models/adaboost_trainer_tests' 目录。")
        TEST_MODEL_SAVE_DIR = Path("./temp_trained_models/adaboost_trainer_tests")
    
    TEST_MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"测试中训练好的模型将保存到: {TEST_MODEL_SAVE_DIR}\n")


    # 1. 准备一个多分类数据集 (例如3个类别)
    X_multi_sample, y_multi_sample = make_classification(
        n_samples=250, n_features=15, 
        n_informative=8, n_redundant=2, 
        random_state=123, n_classes=3, n_clusters_per_class=1
    )
    X_train_m_s, X_test_m_s, y_train_m_s, y_test_m_s = train_test_split(
        X_multi_sample, y_multi_sample, test_size=0.3, random_state=321
    )
    print(f"测试用多分类数据形状: X_train: {X_train_m_s.shape}, y_train: {y_train_m_s.shape}")
    print(f"训练集标签类别: {np.unique(y_train_m_s)}\n")

    # --- 测试 train_custom_adaboost_with_stumps ---
    stump_model_path = TEST_MODEL_SAVE_DIR / "test_adaboost_stumps.pkl"
    try:
        print("--- 测试场景1: train_custom_adaboost_with_stumps ---")
        stump_ada_model, stump_train_time = train_custom_adaboost_with_stumps(
            X_train_m_s, y_train_m_s, stump_model_path,
            n_estimators=10, learning_rate=1.0, random_state=42 # 使用较少迭代次数以加速测试
        )
        print(f"  返回的模型类型: {type(stump_ada_model)}, 训练时间: {stump_train_time:.4f}s")
        if stump_model_path.exists():
            print(f"  模型已成功保存到: {stump_model_path}")
            # (可选) 加载并验证
            loaded_model = joblib.load(stump_model_path)
            y_pred_test = loaded_model.predict(X_test_m_s)
            print(f"  加载模型后在测试集上的准确率: {accuracy_score(y_test_m_s, y_pred_test):.4f}")
        else:
            print(f"  错误: 模型文件 {stump_model_path} 未找到!")
    except Exception as e1:
        print(f"  测试 train_custom_adaboost_with_stumps 时发生错误: {e1}")
        import traceback
        traceback.print_exc()


    # --- 测试 train_custom_adaboost_with_linear_svm ---
    svm_model_path = TEST_MODEL_SAVE_DIR / "test_adaboost_svm.pkl"
    try:
        print("\n--- 测试场景2: train_custom_adaboost_with_linear_svm ---")
        svm_ada_model, svm_train_time = train_custom_adaboost_with_linear_svm(
            X_train_m_s, y_train_m_s, svm_model_path,
            n_estimators=3, learning_rate=0.5, # 极少迭代次数以加速SVM测试
            base_svm_C=0.01, base_svm_max_iter=200, random_state=42
        )
        print(f"  返回的模型类型: {type(svm_ada_model)}, 训练时间: {svm_train_time:.4f}s")
        if svm_model_path.exists():
            print(f"  模型已成功保存到: {svm_model_path}")
            loaded_model_svm = joblib.load(svm_model_path)
            y_pred_test_svm = loaded_model_svm.predict(X_test_m_s)
            print(f"  加载模型后在测试集上的准确率: {accuracy_score(y_test_m_s, y_pred_test_svm):.4f}")
        else:
            print(f"  错误: 模型文件 {svm_model_path} 未找到!")
    except Exception as e2:
        print(f"  测试 train_custom_adaboost_with_linear_svm 时发生错误: {e2}")
        import traceback
        traceback.print_exc()

    print("\n>>> adaboost_trainer.py 函数测试结束 <<<")
    print(f"请检查 '{TEST_MODEL_SAVE_DIR}' 目录下的输出模型文件。")
    print("="*70)