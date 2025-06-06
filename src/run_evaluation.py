# src/run_evaluation.py

import sys
import time # 虽然不训练，但可以保留用于未来可能的计时
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix # 用于生成混淆矩阵数据

# --- 动态路径调整 (确保能找到 src.config 等模块) ---
try:
    current_file_path = Path(__file__).resolve()
    PROJECT_ROOT_GUESS = current_file_path.parent.parent 
    if str(PROJECT_ROOT_GUESS) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_GUESS))
        print(f"信息: 已将项目根目录 '{PROJECT_ROOT_GUESS}' 添加到 sys.path。")
except Exception as e_path:
    print(f"警告: 尝试动态调整 sys.path 失败: {e_path}")

# --- 导入项目模块 ---
try:
    from src import config
    from src.data_preprocess.loader import (
        load_processed_test_data,
        load_scaler,
        load_model # 新增的加载模型的函数
    )
    from src.common.predictor import predict_with_model
    from src.common.evaluator import calculate_classification_metrics
    from src.common.plotting import (
        plot_confusion_matrix_heatmap,
        plot_bar_comparison 
        # 根据需要导入其他绘图函数
    )
except ImportError as e_import:
    print(f"CRITICAL: 导入必要的项目模块失败: {e_import}")
    print("请确保所有依赖的 .py 文件都在正确的位置，并且 src 目录结构正确。")
    print("尝试从项目根目录运行: python -m src.run_evaluation")
    sys.exit(1)

def ensure_evaluation_directories_exist():
    """确保评估脚本所需的输出目录存在"""
    print("  确保评估结果输出目录存在...")
    # EVAL_PLOTS_DIR 会在 main 函数中根据 config.PLOTS_DIR 和子目录名定义
    # 这里主要是为了演示，实际的 EVAL_PLOTS_DIR 的创建在 main 中进行
    # 但可以先检查 config 中定义的基础 PLOTS_DIR 是否存在
    paths_to_check = [
        config.PROCESSED_DATA_DIR, # 需要从中加载数据
        config.UTILS_DIR,          # 需要从中加载scaler
        config.TRAINED_MODEL_DIR,   # 需要从中加载模型
        config.PLOTS_DIR           # 绘图的基础目录
    ]
    for path_str in paths_to_check:
        Path(path_str).mkdir(parents=True, exist_ok=True)
    print("  基础输出/输入目录检查完毕。")


def display_and_evaluate_single_model(
        model_file_path: Path, 
        model_display_name: str,
        X_test_scaled: np.ndarray, 
        y_test: np.ndarray,
        plots_output_dir: Path, 
        show_plots_flag: bool,
        class_names: list) -> dict | None:
    """
    辅助函数：加载指定路径的模型，进行预测，评估性能，并显示/保存其混淆矩阵。

    返回:
        metrics (dict | None): 包含评估指标的字典，如果加载或评估失败则返回None。
    """
    print(f"\n" + "-"*20 + f" 正在评估模型: {model_display_name} " + "-"*20)
    
    # 1. 加载模型
    print(f"  从路径加载模型: {model_file_path}")
    if not model_file_path.exists():
        print(f"  错误: 模型文件 {model_file_path} 未找到！请确保路径和文件名正确，并且模型已训练保存。")
        return None
    try:
        model = load_model(model_file_path) # 使用 loader.py 中的函数
    except Exception as e_load:
        print(f"  错误: 加载模型 {model_file_path} 失败: {e_load}")
        return None

    # 2. 在测试集上进行预测
    print(f"  在测试集上进行预测...")
    try:
        y_pred = predict_with_model(model, X_test_scaled) # 使用 prediction.py 中的函数
    except Exception as e_pred:
        print(f"  错误: 模型 '{model_display_name}' 预测失败: {e_pred}")
        return None

    # 3. 计算性能指标
    print(f"  计算性能指标...")
    # calculate_classification_metrics 函数内部会打印指标
    metrics = calculate_classification_metrics(y_test, y_pred, model_name=model_display_name) 
    
    # 4. 生成并显示/保存混淆矩阵图
    print(f"  生成混淆矩阵图...")
    cm = confusion_matrix(y_test, y_pred, labels=class_names) # 使用传入的class_names作为labels参数确保顺序
    
    # 清理模型名称以用作文件名 (移除特殊字符和空格)
    safe_model_name_for_file = "".join(c if c.isalnum() else "_" for c in model_display_name)
    cm_plot_filename = f"eval_cm_{safe_model_name_for_file}.png"
    cm_plot_path = plots_output_dir / cm_plot_filename
    
    plot_confusion_matrix_heatmap( # 使用 plotting.py 中的函数
        cm, 
        class_names=class_names, 
        title=f"{model_display_name} 混淆矩阵 (测试集)",
        save_path=cm_plot_path,    # 总是尝试保存
        show_plot=show_plots_flag  # 根据开关决定是否 plt.show()
    )
    return metrics


def main():
    """
    主函数：执行已训练模型的快速评估流程。
    """
    print("="*70)
    print("开始执行已训练模型的快速评估演示脚本 (run_evaluation.py)...")
    print("="*70)

    # --- 0. 配置与路径定义 ---
    print("\n--- 步骤 0: 初始化配置和路径 ---")
    ensure_evaluation_directories_exist() # 确保基础目录存在

    PROCESSED_DIR = Path(config.PROCESSED_DATA_DIR)
    UTILS_DIR = Path(config.UTILS_DIR)
    SAVED_MODELS_BASE_DIR = Path(config.TRAINED_MODEL_DIR) # 从config获取模型保存的根目录
    
    # 为本次评估脚本生成的图表创建一个特定的子目录
    EVAL_PLOTS_DIR = Path(config.PLOTS_DIR) / "evaluation_summary_plots"
    EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  评估报告图表将保存到: {EVAL_PLOTS_DIR}")

    # 控制是否在运行时交互式显示图表 (老师验收时设为True可能更好)
    SHOW_PLOTS_INTERACTIVELY = True 
    print(f"  是否交互式显示图表: {SHOW_PLOTS_INTERACTIVELY}")

    # !!! 关键: 定义您要评估的“最优”模型的具体文件名 !!!
    # !!! 您需要根据实际训练脚本保存的文件名来修改这些变量 !!!
    BEST_LINEAR_SVM_FILENAME = "task1_tuned_linear_svm.pkl"       # 示例文件名，请修改
    BEST_RBF_SVM_FILENAME = "task1_tuned_rbf_svm.pkl"          # 示例文件名，请修改
    BEST_ADABOOST_STUMPS_FILENAME = "task2_adaboost_stumps_final_N300.pkl" # 示例 (N300是之前讨论的最终配置)
    BEST_ADABOOST_SVM_FILENAME = "task2_adaboost_svm_C0.01_N30_LR0.5.pkl" # 示例 (某个较优配置)
    
    # 构建完整路径
    path_best_linear_svm = SAVED_MODELS_BASE_DIR / BEST_LINEAR_SVM_FILENAME
    path_best_rbf_svm = SAVED_MODELS_BASE_DIR / BEST_RBF_SVM_FILENAME
    path_best_ada_stumps = SAVED_MODELS_BASE_DIR / BEST_ADABOOST_STUMPS_FILENAME
    path_best_ada_svm = SAVED_MODELS_BASE_DIR / BEST_ADABOOST_SVM_FILENAME
    
    SCALER_PATH = UTILS_DIR / "mnist_standard_scaler.pkl"
    print("步骤 0 完成。")


    # --- 1. 加载测试数据和Scaler ---
    print("\n--- 步骤 1: 加载测试数据和Scaler ---")
    try:
        X_test_unscaled, y_test = load_processed_test_data(PROCESSED_DIR)
        scaler = load_scaler(SCALER_PATH)
        X_test_scaled = scaler.transform(X_test_unscaled)
        print(f"  测试数据 (形状: {X_test_scaled.shape}) 和Scaler加载并处理完毕。")
        
        # MNIST 类别名称 (0-9)
        mnist_class_names = [str(i) for i in range(10)]

    except FileNotFoundError as e_fnf:
        print(f"错误 (步骤 1): 加载测试数据或Scaler文件失败: {e_fnf}")
        print("      请确保已成功运行 'prepare_data.py' 脚本。")
        sys.exit(1)
    except Exception as e_load_data:
        print(f"错误 (步骤 1): 加载测试数据或Scaler时发生未知错误: {e_load_data}")
        sys.exit(1)

    all_final_model_metrics = [] # 用于存储所有最优模型的指标，以便最后对比

    # --- 2. 评估任务1的最优模型 ---
    metrics_task1_linear = display_and_evaluate_single_model(
        model_file_path=path_best_linear_svm, 
        model_display_name="最佳线性SVM (任务1)", 
        X_test_scaled=X_test_scaled, 
        y_test=y_test, 
        plots_output_dir=EVAL_PLOTS_DIR, 
        show_plots_flag=SHOW_PLOTS_INTERACTIVELY,
        class_names=mnist_class_names
    )
    if metrics_task1_linear: all_final_model_metrics.append(metrics_task1_linear)
    
    metrics_task1_rbf = display_and_evaluate_single_model(
        model_file_path=path_best_rbf_svm, 
        model_display_name="最佳RBF SVM (任务1)", 
        X_test_scaled=X_test_scaled, 
        y_test=y_test, 
        plots_output_dir=EVAL_PLOTS_DIR, 
        show_plots_flag=SHOW_PLOTS_INTERACTIVELY,
        class_names=mnist_class_names
    )
    if metrics_task1_rbf: all_final_model_metrics.append(metrics_task1_rbf)

    # --- 3. 评估任务2的最优模型 ---
    metrics_task2_stumps = display_and_evaluate_single_model(
        model_file_path=path_best_ada_stumps, 
        model_display_name="最佳AdaBoost+树桩 (任务2)", 
        X_test_scaled=X_test_scaled, 
        y_test=y_test, 
        plots_output_dir=EVAL_PLOTS_DIR, 
        show_plots_flag=SHOW_PLOTS_INTERACTIVELY,
        class_names=mnist_class_names
    )
    if metrics_task2_stumps: all_final_model_metrics.append(metrics_task2_stumps)

    metrics_task2_svm = display_and_evaluate_single_model(
        model_file_path=path_best_ada_svm, 
        model_display_name="最佳AdaBoost+线性SVM (任务2)", 
        X_test_scaled=X_test_scaled, 
        y_test=y_test, 
        plots_output_dir=EVAL_PLOTS_DIR, 
        show_plots_flag=SHOW_PLOTS_INTERACTIVELY,
        class_names=mnist_class_names
    )
    if metrics_task2_svm: all_final_model_metrics.append(metrics_task2_svm)
    
    # --- 4. (可选) 生成所有最优模型的最终性能对比图 ---
    if len(all_final_model_metrics) >= 1: # 至少有一个模型被成功评估
        print("\n" + "="*30 + " 生成所有已评估最优模型的最终性能对比图 " + "="*30)
        
        # 从 all_final_model_metrics 提取数据 (注意：'model_name' 已在metrics字典中)
        model_names_plot = [m['model_name'] for m in all_final_model_metrics] 
        accuracies_plot = [m['accuracy'] for m in all_final_model_metrics]
        f1_macros_plot = [m['f1_macro'] for m in all_final_model_metrics]
        # 注意：训练时间不在此脚本中重新计算，如果要对比，需要从之前保存的 metrics.csv 中加载
        # 或者，如果模型对象内部保存了训练时间（不常见），可以提取。
        # 为简单起见，此评估脚本主要关注预测性能的对比图。

        if model_names_plot: # 确保有数据可画
            plot_bar_comparison(
                x_labels=model_names_plot, y_values=accuracies_plot,
                title="所有最优模型 准确率对比 (测试集)", xlabel="模型", ylabel="准确率",
                save_path=EVAL_PLOTS_DIR / "summary_all_best_accuracy.png", 
                show_plot=SHOW_PLOTS_INTERACTIVELY,
                figure_size=(max(8, len(model_names_plot) * 2), 6) # 根据模型数量调整宽度
            )
            plot_bar_comparison(
                x_labels=model_names_plot, y_values=f1_macros_plot,
                title="所有最优模型 F1 Macro 对比 (测试集)", xlabel="模型", ylabel="F1 Macro",
                save_path=EVAL_PLOTS_DIR / "summary_all_best_f1_macro.png", 
                show_plot=SHOW_PLOTS_INTERACTIVELY,
                figure_size=(max(8, len(model_names_plot) * 2), 6)
            )
            print("最终对比图表已生成。")
        else:
            print("没有收集到足够的模型指标用于绘制最终对比图。")
    else:
        print("\n未能成功评估任何模型，无法生成对比图。")

    print("\n" + "="*70)
    print("模型快速评估脚本 (run_evaluation.py) 执行完毕。")
    print(f"所有本脚本生成的图表已保存到: {EVAL_PLOTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    # 设置Matplotlib后端，如果需要在无GUI环境运行或避免某些后端问题
    # import matplotlib
    # matplotlib.use('Agg') # 例如，保存到文件而不显示窗口
    main()