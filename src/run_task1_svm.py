# src/run_task1_svm.py

import sys
import time
from pathlib import Path
import numpy as np
import joblib # 虽然模型保存封装在trainer里，但GridSearchCV结果可能也需要joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix # 直接在这里计算混淆矩阵

# --- 动态路径调整 (确保能找到 src.config 等模块) ---
try:
    current_file_path = Path(__file__).resolve()
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
        load_processed_train_data,
        load_processed_test_data,
        load_scaler
    )
    # 我们将使用GridSearchCV来寻找最佳参数，然后用这些参数调用我们已有的trainer函数
    # trainer函数负责用最佳参数训练最终模型并保存，同时返回训练时间
    from src.models.linear_svm_trainer import train_linear_svm
    from src.models.rbf_svm_trainer import train_rbf_svm
    from src.common.predictor import predict_with_model
    from src.common.evaluator import calculate_classification_metrics
    from src.common.result_utils import append_metrics_to_csv
    from src.common.plotting import (
        plot_bar_comparison,
        plot_confusion_matrix_heatmap,
        plot_grid_search_heatmap
    )
except ImportError as e:
    print(f"CRITICAL: 导入必要的项目模块失败: {e}")
    print("请确保所有依赖的 .py 文件都在正确的位置，并且 src 目录结构正确。")
    print("尝试从项目根目录运行: python -m src.run_task1_svm")
    sys.exit(1)

def ensure_directories_exist():
    """确保所有在config中定义的输出目录都存在"""
    print("  确保输出目录存在...")
    paths_to_check = [
        config.TRAINED_MODEL_DIR, 
        config.METRICS_DIR, 
        config.PLOTS_DIR,
        config.UTILS_DIR, # scaler 也在这里
        config.PROCESSED_DATA_DIR # 预处理数据也需要存在
    ]
    for path_str in paths_to_check:
        Path(path_str).mkdir(parents=True, exist_ok=True)
    print("  输出目录检查完毕。")

def main():
    """
    执行任务1：比较线性核SVM和RBF核SVM的性能，包括超参数调优。
    """
    print("="*70)
    print("开始执行任务1：SVM性能对比 (包含超参数调优)")
    print("="*70)

    # --- 0. 初始化路径和参数 ---
    print("\n--- 步骤 0: 初始化路径、参数和CSV文件表头 ---")
    ensure_directories_exist() # 确保输出目录存在

    PROCESSED_DIR = Path(config.PROCESSED_DATA_DIR)
    SCALER_PATH = Path(config.UTILS_DIR) / "mnist_standard_scaler.pkl"
    
    LINEAR_SVM_MODEL_SAVE_PATH = Path(config.TRAINED_MODEL_DIR) / "task1_tuned_linear_svm.pkl"
    RBF_SVM_MODEL_SAVE_PATH = Path(config.TRAINED_MODEL_DIR) / "task1_tuned_rbf_svm.pkl"
    
    METRICS_CSV_PATH = Path(config.METRICS_DIR) / "task1_svm_comparison_metrics.csv"
    PLOTS_OUTPUT_DIR = Path(config.PLOTS_TASK1_DIR)

    # CSV 表头和顺序 (确保包含所有要记录的参数和指标)
    CSV_FIELD_ORDER = [
        'model_name', 'kernel', 'best_C', 'best_gamma', 
        'cv_best_score', 'tuning_time_seconds', 'final_model_training_time_seconds',
        'test_accuracy', 'test_f1_micro', 'test_f1_macro', 'test_f1_weighted'
    ]

    # 超参数网格定义
    # 注意：为了演示，这里的网格可能较小，实际中可以扩大范围
    # 如果数据集很大，调优会非常耗时，MNIST 60000条数据，GridSearchCV会比较慢
    # 可以考虑减少cv折数，或使用RandomizedSearchCV，或在数据子集上调优
    CV_FOLDS = 3 # 交叉验证折数，3或5是常用的。对于大型数据集，3可以快一些。
    SCORING_METRIC = 'f1_macro' # GridSearchCV的评分标准

    PARAM_GRID_LINEAR = {
        'C': [0.1, 1.0, 10.0] 
    }
    PARAM_GRID_RBF = {
        'C': [1.0, 10.0, 50.0], # RBF 通常对 C 和 gamma 更敏感
        'gamma': [0.001, 0.01, 'scale'] # 'scale' 是一个不错的默认值
    }
    print(f"  结果CSV文件: {METRICS_CSV_PATH}")
    print(f"  CSV表头顺序: {CSV_FIELD_ORDER}")
    print(f"  线性SVM参数网格: {PARAM_GRID_LINEAR}")
    print(f"  RBF SVM参数网格: {PARAM_GRID_RBF}")
    print(f"  交叉验证折数: {CV_FOLDS}")
    print(f"  GridSearchCV评分标准: {SCORING_METRIC}")
    print("步骤 0 完成。")

    # --- 1. 数据加载与预处理 ---
    print("\n--- 步骤 1: 加载数据并进行标准化 ---")
    try:
        print("  加载已划分的训练数据...")
        X_train_unscaled, y_train = load_processed_train_data(PROCESSED_DIR)
        print("  加载已划分的测试数据...")
        X_test_unscaled, y_test = load_processed_test_data(PROCESSED_DIR)
        
        print(f"  原始训练数据形状: X={X_train_unscaled.shape}, y={y_train.shape}")
        print(f"  原始测试数据形状: X={X_test_unscaled.shape}, y={y_test.shape}")

        # --- 数据子采样 (例如 1:10 比例，与任务二保持一致或单独设置) ---
        # 您可以为任务1设置不同的采样比例，如果需要的话
        SUBSAMPLE_TRAIN_RATIO_TASK1 = 0.1 # 定义任务1的采样比例，例如10%
        ACTUAL_TRAIN_SIZE_FULL_TASK1 = X_train_unscaled.shape[0]
        
        # 仅当希望的子样本大小小于完整训练集大小时才进行子采样
        if SUBSAMPLE_TRAIN_RATIO_TASK1 < 1.0 and SUBSAMPLE_TRAIN_RATIO_TASK1 > 0:
            DESIRED_TRAIN_SAMPLE_SIZE_TASK1 = int(ACTUAL_TRAIN_SIZE_FULL_TASK1 * SUBSAMPLE_TRAIN_RATIO_TASK1)
            if DESIRED_TRAIN_SAMPLE_SIZE_TASK1 > 0 : # 确保目标大小有效
                print(f"信息: 任务1原始训练集大小为 {ACTUAL_TRAIN_SIZE_FULL_TASK1}，将进行 {SUBSAMPLE_TRAIN_RATIO_TASK1*100:.0f}% 子采样。")
                print(f"      任务1目标子样本训练集大小为: {DESIRED_TRAIN_SAMPLE_SIZE_TASK1}")
                
                # 使用固定的随机状态确保每次运行的子样本一致
                # 确保 config.py 中有 RANDOM_STATE 定义，或者在此处硬编码一个
                try:
                    # 脚本顶部的导入部分应包含: from src import config
                    # 以及 import numpy as np
                    random_seed_for_subsample_task1 = config.RANDOM_STATE 
                except AttributeError:
                    print("警告: config.py 中未定义 RANDOM_STATE，将使用默认值 42 进行子采样。")
                    random_seed_for_subsample_task1 = 42 # 与任务二的默认值保持一致
                    
                np.random.seed(random_seed_for_subsample_task1) # 设置随机种子
                shuffled_indices_task1 = np.random.permutation(ACTUAL_TRAIN_SIZE_FULL_TASK1)
                subset_indices_task1 = shuffled_indices_task1[:DESIRED_TRAIN_SAMPLE_SIZE_TASK1]
                
                X_train_unscaled_subset = X_train_unscaled[subset_indices_task1] # <--- 新的变量名
                y_train_subset = y_train[subset_indices_task1] # <--- 新的变量名
                
                print(f"  子采样后训练数据形状: X={X_train_unscaled_subset.shape}, y={y_train_subset.shape}")
            else:
                print(f"警告: 计算得到的目标子样本大小 ({DESIRED_TRAIN_SAMPLE_SIZE_TASK1}) 无效，将使用完整训练集。")
                X_train_unscaled_subset = X_train_unscaled
                y_train_subset = y_train
        else:
            X_train_unscaled_subset = X_train_unscaled # 如果不采样或比例无效，则使用完整数据
            y_train_subset = y_train
            print(f"信息: 任务1使用完整的训练数据集 (大小: {ACTUAL_TRAIN_SIZE_FULL_TASK1})。未进行子采样。")
        
        # --- 子采样结束 ---


        print(f"  加载StandardScaler从: {SCALER_PATH}...")
        scaler = load_scaler(SCALER_PATH)
        
        print("  标准化训练数据...")
        X_train_scaled = scaler.transform(X_train_unscaled_subset)
        print("  标准化测试数据...")
        X_test_scaled = scaler.transform(X_test_unscaled)
        print("步骤 1 完成: 数据加载和标准化完毕。")
    except FileNotFoundError as e:
        print(f"错误 (步骤 1): 数据文件或Scaler文件未找到: {e}")
        print("      请确保已成功运行 'prepare_data.py' 脚本。")
        sys.exit(1)
    except Exception as e:
        print(f"错误 (步骤 1): 数据加载或标准化过程中发生错误: {e}")
        sys.exit(1)

    # 用于存储所有实验结果，方便最后绘图比较
    all_experiment_results = []


    # --- 2. 线性核SVM实验 (含超参数调优) ---
    print("\n" + "="*30 + " 开始线性核SVM实验 " + "="*30)
    try:
        print(f"\n--- 步骤 2.1: 使用GridSearchCV调优线性SVM参数 (C) ---")
        linear_svm = SVC(kernel='linear', probability=True, random_state=42)
        grid_search_linear = GridSearchCV(estimator=linear_svm, 
                                          param_grid=PARAM_GRID_LINEAR, 
                                          scoring=SCORING_METRIC, 
                                          cv=CV_FOLDS, 
                                          verbose=1, # 可以设为0减少输出，设为2更详细
                                          n_jobs=-1) # 使用所有可用CPU核心

        print(f"  正在对线性SVM进行GridSearchCV (这可能需要一些时间)...")
        start_tune_time_linear = time.time()
        grid_search_linear.fit(X_train_scaled, y_train_subset)
        tuning_time_linear = time.time() - start_tune_time_linear
        print(f"  GridSearchCV调优完成，耗时: {tuning_time_linear:.2f} 秒。")

        best_params_linear = grid_search_linear.best_params_
        best_cv_score_linear = grid_search_linear.best_score_
        # best_linear_svm_estimator_from_grid = grid_search_linear.best_estimator_ # 这是GridSearchCV refit的模型

        print(f"  找到的最佳参数 (线性SVM): {best_params_linear}")
        print(f"  对应的最佳交叉验证 ({SCORING_METRIC}): {best_cv_score_linear:.4f}")

        print(f"\n--- 步骤 2.2: 使用最佳参数训练最终线性SVM模型并保存 ---")
        # 调用我们自定义的trainer函数，它会训练并保存模型，并返回模型和训练时间
        final_linear_svm_model, final_training_time_linear = train_linear_svm(
            X_train_scaled, y_train_subset,
            model_save_path=LINEAR_SVM_MODEL_SAVE_PATH,
            C_param=best_params_linear['C'] # 从GridSearchCV结果中获取最佳C
        )
        print(f"  最终线性SVM模型训练完成并已保存。训练耗时 (仅最终模型): {final_training_time_linear:.4f} 秒。")

        print(f"\n--- 步骤 2.3: 在测试集上评估最终线性SVM模型 ---")
        y_pred_linear = predict_with_model(final_linear_svm_model, X_test_scaled)
        metrics_linear = calculate_classification_metrics(y_test, y_pred_linear, model_name="调优后的线性SVM")
        
        # 准备记录
        record_linear = {
            'model_name': 'Linear SVM (Tuned)',
            'kernel': 'linear',
            'best_C': best_params_linear.get('C', 'N/A'),
            'best_gamma': 'N/A', # 线性核没有gamma
            'cv_best_score': best_cv_score_linear,
            'tuning_time_seconds': round(tuning_time_linear, 4),
            'final_model_training_time_seconds': round(final_training_time_linear, 4),
            'test_accuracy': metrics_linear['accuracy'],
            'test_f1_micro': metrics_linear['f1_micro'],
            'test_f1_macro': metrics_linear['f1_macro'],
            'test_f1_weighted': metrics_linear['f1_weighted']
        }
        # 修正 record_linear 中的 model_name (确保是我们定义的)
        record_linear['model_name'] = 'Linear SVM (Tuned)' 

        append_metrics_to_csv(METRICS_CSV_PATH, record_linear, field_order=CSV_FIELD_ORDER)
        all_experiment_results.append(record_linear)

        print(f"\n--- 步骤 2.4: 绘制并保存线性SVM的混淆矩阵 ---")
        cm_linear = confusion_matrix(y_test, y_pred_linear)
        plot_confusion_matrix_heatmap(
            cm_linear, 
            class_names=[str(i) for i in range(10)], # MNIST有0-9共10类
            title="线性SVM混淆矩阵 (测试集)",
            save_path=PLOTS_OUTPUT_DIR / "task1_cm_linear_svm_tuned.png"
        )
        print("线性核SVM实验完成。")

    except Exception as e:
        print(f"错误: 线性核SVM实验过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

    # --- 3. RBF核SVM实验 (含超参数调优) ---
    print("\n" + "="*30 + " 开始RBF核SVM实验 " + "="*30)
    try:
        print(f"\n--- 步骤 3.1: 使用GridSearchCV调优RBF SVM参数 (C, gamma) ---")
        rbf_svm = SVC(kernel='rbf', probability=True, random_state=42)
        grid_search_rbf = GridSearchCV(estimator=rbf_svm,
                                       param_grid=PARAM_GRID_RBF,
                                       scoring=SCORING_METRIC,
                                       cv=CV_FOLDS,
                                       verbose=1,
                                       n_jobs=-1)
        
        print(f"  正在对RBF SVM进行GridSearchCV (这可能需要更长时间)...")
        start_tune_time_rbf = time.time()
        grid_search_rbf.fit(X_train_scaled, y_train_subset)
        tuning_time_rbf = time.time() - start_tune_time_rbf
        print(f"  GridSearchCV调优完成，耗时: {tuning_time_rbf:.2f} 秒。")

        best_params_rbf = grid_search_rbf.best_params_
        best_cv_score_rbf = grid_search_rbf.best_score_

        print(f"  找到的最佳参数 (RBF SVM): {best_params_rbf}")
        print(f"  对应的最佳交叉验证 ({SCORING_METRIC}): {best_cv_score_rbf:.4f}")

        print(f"\n--- 步骤 3.2: 使用最佳参数训练最终RBF SVM模型并保存 ---")
        final_rbf_svm_model, final_training_time_rbf = train_rbf_svm(
            X_train_scaled, y_train_subset,
            model_save_path=RBF_SVM_MODEL_SAVE_PATH,
            C_param=best_params_rbf['C'],
            gamma_param=best_params_rbf['gamma']
        )
        print(f"  最终RBF SVM模型训练完成并已保存。训练耗时 (仅最终模型): {final_training_time_rbf:.4f} 秒。")

        print(f"\n--- 步骤 3.3: 在测试集上评估最终RBF SVM模型 ---")
        y_pred_rbf = predict_with_model(final_rbf_svm_model, X_test_scaled)
        metrics_rbf = calculate_classification_metrics(y_test, y_pred_rbf, model_name="调优后的RBF核SVM")

        record_rbf = {
            'model_name': 'RBF SVM (Tuned)',
            'kernel': 'rbf',
            'best_C': best_params_rbf.get('C', 'N/A'),
            'best_gamma': best_params_rbf.get('gamma', 'N/A'),
            'cv_best_score': best_cv_score_rbf,
            'tuning_time_seconds': round(tuning_time_rbf, 4),
            'final_model_training_time_seconds': round(final_training_time_rbf, 4),
            'test_accuracy': metrics_rbf['accuracy'],
            'test_f1_micro': metrics_rbf['f1_micro'],
            'test_f1_macro': metrics_rbf['f1_macro'],
            'test_f1_weighted': metrics_rbf['f1_weighted']
        }
        record_rbf['model_name'] = 'RBF SVM (Tuned)' # 确保是我们定义的
        append_metrics_to_csv(METRICS_CSV_PATH, record_rbf, field_order=CSV_FIELD_ORDER)
        all_experiment_results.append(record_rbf)
        
        print(f"\n--- 步骤 3.4: 绘制并保存RBF SVM的混淆矩阵 ---")
        cm_rbf = confusion_matrix(y_test, y_pred_rbf)
        plot_confusion_matrix_heatmap(
            cm_rbf,
            class_names=[str(i) for i in range(10)],
            title="RBF核SVM混淆矩阵 (测试集)",
            save_path=PLOTS_OUTPUT_DIR / "task1_cm_rbf_svm_tuned.png"
        )

        print(f"\n--- 步骤 3.5: 绘制并保存RBF SVM的GridSearchCV热力图 ---")
        # 确保grid_search_rbf.cv_results_存在且包含所需信息
        if hasattr(grid_search_rbf, 'cv_results_'): 
            plot_grid_search_heatmap(
                cv_results=grid_search_rbf.cv_results_,
                param_x_name='C', # GridSearchCV中参数名是 C
                param_y_name='gamma', # GridSearchCV中参数名是 gamma
                title="RBF SVM GridSearchCV 热力图 (C vs Gamma)",
                save_path=PLOTS_OUTPUT_DIR / "task1_heatmap_rbf_svm_tuning.png",
                score_key=f'mean_test_score' # GridSearchCV结果中的分数键
            )
        else:
            print("  警告: 未找到 RBF SVM 的 GridSearchCV.cv_results_，无法绘制热力图。")
        
        print("RBF核SVM实验完成。")

    except Exception as e:
        print(f"错误: RBF核SVM实验过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

    # --- 4. 结果汇总与对比可视化 ---
    print("\n" + "="*30 + " 结果汇总与对比可视化 " + "="*30)
    if len(all_experiment_results) == 2: # 确保两个实验都产生了结果
        model_names = [res['model_name'] for res in all_experiment_results]
        
        accuracies = [res['test_accuracy'] for res in all_experiment_results]
        f1_macros = [res['test_f1_macro'] for res in all_experiment_results]
        # 使用 final_model_training_time_seconds 进行比较，或者 total_tuning_and_training_time
        # 这里我们用 'tuning_time_seconds' 代表GridSearchCV总时间， 'final_model_training_time_seconds' 是最终模型训练时间
        # 作业要求的是“训练时间”，这里用最终模型的训练时间更贴切
        training_times = [res['final_model_training_time_seconds'] for res in all_experiment_results]

        plot_bar_comparison(
            x_labels=model_names, y_values=accuracies,
            title="SVM模型准确率对比 (测试集)", xlabel="模型", ylabel="准确率",
            save_path=PLOTS_OUTPUT_DIR / "task1_comparison_accuracy.png",
            y_limit=(max(0, min(accuracies)-0.05) if accuracies else 0, min(1, max(accuracies)+0.05) if accuracies else 1) # 动态y轴
        )
        plot_bar_comparison(
            x_labels=model_names, y_values=f1_macros,
            title="SVM模型 F1分数(Macro)对比 (测试集)", xlabel="模型", ylabel="F1分数 (Macro)",
            save_path=PLOTS_OUTPUT_DIR / "task1_comparison_f1_macro.png",
            y_limit=(max(0, min(f1_macros)-0.05) if f1_macros else 0, min(1, max(f1_macros)+0.05) if f1_macros else 1)
        )
        plot_bar_comparison(
            x_labels=model_names, y_values=training_times,
            title="SVM模型训练时间对比 (最终模型)", xlabel="模型", ylabel="训练时间 (秒)",
            save_path=PLOTS_OUTPUT_DIR / "task1_comparison_training_time.png"
        )
        print("对比图表已生成。")
    else:
        print("警告: 未能收集到足够的实验结果进行对比绘图。")

    print("\n" + "="*70)
    print("任务1：SVM性能对比脚本执行完毕！")
    print(f"  性能指标已保存至: {METRICS_CSV_PATH}")
    print(f"  图表已保存至: {PLOTS_OUTPUT_DIR}")
    print(f"  训练好的模型已保存至: {config.TRAINED_MODEL_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()