# src/common/prediction.py
import numpy as np
from typing import Any # 用于通用模型类型提示
# 如果您的所有模型都继承自某个 scikit-learn 基类，可以使用更具体的类型提示，
# 例如: from sklearn.base import BaseEstimator

def predict_with_model(model: Any, X_data: np.ndarray) -> np.ndarray:
    """
    使用训练好的模型对给定的数据进行预测。

    输入:
        model (Any): 已经训练好的分类模型对象。
                     该对象应具备一个 .predict() 方法 (例如 scikit-learn 中的模型)。
        X_data (np.ndarray): 需要进行预测的特征数据集 (例如 X_test_scaled)。

    输出:
        np.ndarray: 模型预测的标签数组。
    
    可能抛出的异常:
        AttributeError: 如果提供的模型对象没有 'predict' 方法。
        Exception: 如果在预测过程中发生其他类型的错误。
    """
    print(f"\n正在使用模型 '{type(model).__name__}' 对数据进行预测...")
    print(f"输入数据规模: X_data shape: {X_data.shape}")

    if not hasattr(model, 'predict'):
        error_msg = f"错误：提供的模型对象 (类型: {type(model).__name__}) 没有 'predict' 方法。"
        print(error_msg)
        raise AttributeError(error_msg)

    try:
        y_pred = model.predict(X_data)
        print("预测完成。")
        if hasattr(y_pred, 'shape'):
            print(f"预测结果 y_pred shape: {y_pred.shape}")
        else:
            print(f"预测结果 y_pred 类型: {type(y_pred)}") # 应对 predict 返回非 ndarray 的罕见情况
        return y_pred
    except Exception as e:
        print(f"错误：模型预测过程中发生错误: {e}")
        # 在实际调试中，您可能希望在此处打印完整的堆栈跟踪信息
        # import traceback
        # traceback.print_exc()
        raise # 将异常重新抛出，以便上层调用者可以处理