# src/common/result_utils.py
import csv
from pathlib import Path
from typing import Dict, Any, List

def append_metrics_to_csv(filepath: Path, 
                          record_data: Dict[str, Any], 
                          field_order: List[str]):
    """
    将一条记录（包含实验参数和性能指标的字典）追加到指定的CSV文件。
    如果文件不存在，则会创建文件并首先写入表头。

    输入:
        filepath (pathlib.Path): CSV文件的完整路径。
        record_data (Dict[str, Any]): 要写入的一行数据，键应与 field_order 中的元素对应。
        field_order (List[str]): CSV文件的列顺序列表（即表头顺序）。
                                 所有 record_data 中的相关键都应包含在此列表中。
    """
    if not field_order:
        print("错误: 必须提供 field_order (列顺序列表)。")
        raise ValueError("field_order 参数不能为空列表。")

    try:
        # 确保父目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 检查文件是否存在以决定是否写入表头
        file_exists = filepath.is_file() # 使用 is_file() 更准确

        with open(filepath, mode='a', newline='', encoding='utf-8') as csvfile:
            # 使用 field_order 作为 DictWriter 的列名，确保写入顺序
            writer = csv.DictWriter(csvfile, fieldnames=field_order, extrasaction='ignore')

            if not file_exists:
                writer.writeheader()
                print(f"CSV文件不存在，已在 '{filepath}' 创建并写入表头。")
            
            # 准备要写入的行，只包含在 field_order 中定义的字段
            # 对于 record_data 中可能没有的字段（但在 field_order 中），写入空字符串
            row_to_write = {field: record_data.get(field, '') for field in field_order}
            writer.writerow(row_to_write)
            
        print(f"记录已成功追加到CSV文件: '{filepath}'")

    except IOError as e:
        print(f"错误: 写入CSV文件 '{filepath}' 失败: {e}")
        # 考虑是否重新抛出异常，或返回False表示失败
        raise 
    except Exception as e:
        print(f"错误: 处理CSV文件 '{filepath}' 时发生未知错误: {e}")
        raise



# ==========================
# 测试程序
# ==========================

if __name__ == '__main__':
    print(">>> 测试 append_metrics_to_csv 函数 <<<")
    
    # --- 模拟设置 ---
    try:
        from src import config # 尝试导入 config
        TEST_METRICS_DIR = Path(config.METRICS_DIR) / "test_results_utils"
    except (ImportError, AttributeError):
        print("警告: 无法从 src.config 导入 METRICS_DIR 或 config 本身。测试结果将保存在本地 ./temp_test_metrics 目录。")
        current_file_dir = Path(__file__).parent
        TEST_METRICS_DIR = current_file_dir / "temp_test_metrics" 

    TEST_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    test_csv_path = TEST_METRICS_DIR / "test_experiment_metrics.csv"

    FIELD_ORDER = [
        'model_name', 'kernel_type', 'C_param', 'gamma_param', 'n_estimators', 
        'training_time_seconds', 'accuracy', 'f1_macro', 'f1_weighted', 'notes'
    ]

    if test_csv_path.exists():
        print(f"清理旧的测试文件: {test_csv_path}")
        test_csv_path.unlink()

    print(f"\n测试案例1: 写入第一条记录 (应创建文件并写入表头)")
    record1_params = {'model_name': 'Linear SVM', 'kernel_type': 'linear', 'C_param': 1.0, 'training_time_seconds': 10.5}
    record1_metrics = {'accuracy': 0.92, 'f1_macro': 0.91, 'f1_weighted': 0.915, 'notes': '第一次运行'}
    record1_full = {**record1_params, **record1_metrics}
    try:
        append_metrics_to_csv(test_csv_path, record1_full, FIELD_ORDER)
        if test_csv_path.exists():
            # ***********************************************
            # 修改点: 为 open() 添加 encoding='utf-8'
            with open(test_csv_path, 'r', encoding='utf-8') as f:
            # ***********************************************
                print(f"'{test_csv_path}' 内容:\n{f.read()}")
        else:
            print(f"错误: 测试文件 {test_csv_path} 未被创建。")
    except Exception as e:
        # 注意：这里的捕获可能掩盖了 append_metrics_to_csv 内部的原始错误（如果它重新抛出）
        # 但根据您的输出，append_metrics_to_csv 看起来是成功写入了
        print(f"测试案例1打印验证时发生错误: {e}") # 明确指出是打印验证时的错误

    print(f"\n测试案例2: 写入第二条记录 (应追加到现有文件)")
    record2_params = {'model_name': 'RBF SVM', 'kernel_type': 'rbf', 'C_param': 10.0, 'gamma_param': 0.1, 'training_time_seconds': 120.7}
    record2_metrics = {'accuracy': 0.95, 'f1_macro': 0.948, 'f1_weighted': 0.95, 'notes': '使用调优参数'}
    record2_full = {**record2_params, **record2_metrics}
    try:
        append_metrics_to_csv(test_csv_path, record2_full, FIELD_ORDER)
        if test_csv_path.exists():
            # ***********************************************
            # 修改点: 为 open() 添加 encoding='utf-8'
            with open(test_csv_path, 'r', encoding='utf-8') as f:
            # ***********************************************
                print(f"'{test_csv_path}' 更新后内容:\n{f.read()}")
    except Exception as e:
        print(f"测试案例2打印验证时发生错误: {e}")

    print(f"\n测试案例3: 写入记录，但缺少 'gamma_param' (应在该列留空)")
    record3_params = {'model_name': 'Linear SVM', 'kernel_type': 'linear', 'C_param': 0.5, 'training_time_seconds': 8.2}
    record3_metrics = {'accuracy': 0.91, 'f1_macro': 0.90, 'f1_weighted': 0.905}
    record3_full = {**record3_params, **record3_metrics}
    try:
        append_metrics_to_csv(test_csv_path, record3_full, FIELD_ORDER)
        if test_csv_path.exists():
            # ***********************************************
            # 修改点: 为 open() 添加 encoding='utf-8'
            with open(test_csv_path, 'r', encoding='utf-8') as f:
            # ***********************************************
                print(f"'{test_csv_path}' 更新后内容:\n{f.read()}")
    except Exception as e:
        print(f"测试案例3打印验证时发生错误: {e}")

    print("\n>>> append_metrics_to_csv 函数测试结束 <<<")
    # 注意：测试后可以手动删除 test_experiment_metrics.csv 文件或 temp_test_metrics 目录