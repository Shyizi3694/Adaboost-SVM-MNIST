# src/common/evaluator.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score # , precision_score, recall_score, confusion_matrix

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "模型") -> dict[str, float]:
    """
    计算给定真实标签和预测标签的各项分类评估指标。

    输入:
        y_true (np.ndarray): 真实标签数组。
        y_pred (np.ndarray): 模型预测的标签数组。
        model_name (str): （可选）模型的名称，用于打印输出。

    输出:
        dict[str, float]: 一个包含以下键值对的字典：
            'accuracy': 准确率 (float)
            'f1_micro': F1分数 (micro-average) (float)
            'f1_macro': F1分数 (macro-average) (float)
            'f1_weighted': F1分数 (weighted-average) (float)
            (未来可以轻松扩展以包含 precision, recall 等)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"真实标签和预测标签的形状必须一致！y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    accuracy = accuracy_score(y_true, y_pred)
    
    # zero_division 参数用于处理当某个类别没有预测样本或真实样本时的情况，避免警告并按指定值处理。
    # 对于F1分数，0表示如果分母为0（precision+recall=0），则F1为0。
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n--- “{model_name}” 的性能评估指标 ---")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  F1 分数 (Micro):   {f1_micro:.4f}")
    print(f"  F1 分数 (Macro):   {f1_macro:.4f}") # 对所有类别同等重要
    print(f"  F1 分数 (Weighted):{f1_weighted:.4f}") # 考虑了类别不平衡

    metrics_dict = {
        'model_name': model_name, # 将模型名称也加入字典，方便后续记录
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }
    
    return metrics_dict






# ==========================
# 测试程序
# ==========================

if __name__ == '__main__':
    # --- 简单的测试代码块 ---
    print(">>> 测试 calculate_classification_metrics 函数 <<<")
    
    # 模拟真实标签和预测标签
    y_true_sample = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
    y_pred_sample1 = np.array([0, 1, 2, 0, 1, 1, 2, 0, 1, 2]) # 一些错误
    y_pred_sample2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # 更多错误
    y_all_correct = y_true_sample.copy()

    print("\n测试案例1 (一些错误):")
    metrics1 = calculate_classification_metrics(y_true_sample, y_pred_sample1, model_name="测试模型1")
    print("返回的字典:", metrics1)

    print("\n测试案例2 (较多错误):")
    metrics2 = calculate_classification_metrics(y_true_sample, y_pred_sample2, model_name="测试模型2")
    print("返回的字典:", metrics2)

    print("\n测试案例3 (全部正确):")
    metrics3 = calculate_classification_metrics(y_true_sample, y_all_correct, model_name="完美模型")
    print("返回的字典:", metrics3)

    # 测试标签不匹配的情况 (预期会抛出ValueError)
    print("\n测试案例4 (标签形状不匹配 - 预期错误):")
    try:
        y_pred_wrong_shape = np.array([0,1,2])
        calculate_classification_metrics(y_true_sample, y_pred_wrong_shape, model_name="形状错误测试")
    except ValueError as e:
        print(f"成功捕获到预期的错误: {e}")
    except Exception as e:
        print(f"捕获到非预期的错误: {e}")