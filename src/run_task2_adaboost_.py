# src/run_task2_adaboost.py

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd # 用于更方便地处理CSV和筛选数据
import joblib 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# --- 动态路径调整 ---
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
        load_processed_train_data,
        load_processed_test_data,
        load_scaler,
        load_model
    )
    from src.models.adaboost_trainer import (
        train_custom_adaboost_with_stumps,
        train_custom_adaboost_with_linear_svm
    )
    from src.common.predictor import predict_with_model
    from src.common.evaluator import calculate_classification_metrics
    from src.common.result_utils import append_metrics_to_csv
    from src.common.plotting import (
        plot_bar_comparison,
        plot_confusion_matrix_heatmap,
        plot_learning_curve
    )
except ImportError as e_import:
    print(f"CRITICAL: 导入必要的项目模块失败: {e_import}")
    print("请确保所有依赖的 .py 文件都在正确的位置，并且 src 目录结构正确。")
    print("尝试从项目根目录运行: python -m src.run_task2_adaboost")
    sys.exit(1)

def ensure_directories_exist_task2():
    """确保任务2所需的所有输出目录都存在"""
    print("  确保任务2输出目录存在...")
    paths_to_check = [
        config.TRAINED_MODEL_DIR, 
        config.METRICS_DIR, 
        config.PLOTS_TASK2_DIR,
    ]
    for path_str in paths_to_check:
        Path(path_str).mkdir(parents=True, exist_ok=True)
    print("  任务2输出目录检查完毕。")

def safe_filename_component(name: str) -> str:
    """将实验名称转换为对文件名安全的部分"""
    # 移除或替换不安全的字符
    return "".join(c if c.isalnum() else "_" for c in name).strip('_').replace('__','_')

def run_single_experiment(experiment_config: dict,
                          X_train_scaled: np.ndarray, y_train: np.ndarray,
                          X_test_scaled: np.ndarray, y_test: np.ndarray,
                          metrics_csv_path: Path, csv_field_order: list,
                          models_save_dir: Path, plots_output_dir: Path,
                          existing_results_df: pd.DataFrame,
                          force_rerun: bool = False):
    """
    辅助函数，用于运行单个实验配置，包含通用的续跑逻辑。
    """
    experiment_name = experiment_config['experiment_name']
    base_learner_type = experiment_config['base_learner_type']
    
    # 步骤 1: 检查此实验的结果是否已在CSV中
    if not force_rerun and not existing_results_df.empty and \
       experiment_name in existing_results_df['experiment_name'].values:
        print(f"\n--- {experiment_name} 的结果已在CSV中，跳过。 ---")
        record = existing_results_df[existing_results_df['experiment_name'] == experiment_name].iloc[0].to_dict()
        return record

    print(f"\n--- 执行/续跑实验: {experiment_name} ---")
    
    model_obj = None
    train_time = -1.0 # 默认为-1.0，表示续跑或无法获取时间
    
    # 为当前实验配置构建模型文件名
    safe_exp_name_for_file = safe_filename_component(experiment_name)
    model_save_path = models_save_dir / f"task2_{safe_exp_name_for_file}.pkl"

    # 步骤 2: 尝试加载已存在的模型文件 (通用续跑逻辑)
    if not force_rerun and model_save_path.exists():
        print(f"  检测到模型文件 {model_save_path} 已存在。将加载模型并仅进行评估。")
        try:
            model_obj = load_model(model_save_path)
            print(f"  模型加载成功。训练时间将标记为 N/A (Resumed)。")
        except Exception as e_load_resume:
            print(f"  警告: 加载已存在的模型失败: {e_load_resume}。将尝试重新训练。")
            model_obj = None 

    # 步骤 3: 如果模型未被加载，则进行训练
    if model_obj is None:
        print(f"  开始训练模型: {experiment_name}")
        try:
            if base_learner_type == 'Decision Stump':
                if "AdaBoost" in experiment_name:
                    model_obj, train_time = train_custom_adaboost_with_stumps(
                        X_train=X_train_scaled, 
                        y_train=y_train, 
                        model_save_path=model_save_path,
                        n_estimators=experiment_config['ada_n_estimators'],
                        learning_rate=experiment_config['ada_learning_rate'],
                        random_state=config.RANDOM_STATE
                    )
                else: # 基线决策树桩
                    model_obj = DecisionTreeClassifier(**experiment_config['base_params'])
                    start_time = time.time()
                    model_obj.fit(X_train_scaled, y_train)
                    train_time = time.time() - start_time
                    joblib.dump(model_obj, model_save_path)
                    print(f"  基线决策树桩模型已保存至: {model_save_path}")

            elif base_learner_type == 'Linear SVM':
                if "AdaBoost" in experiment_name:
                    model_obj, train_time = train_custom_adaboost_with_linear_svm(
                        X_train=X_train_scaled, 
                        y_train=y_train, 
                        model_save_path=model_save_path,
                        n_estimators=experiment_config['ada_n_estimators'],
                        learning_rate=experiment_config['ada_learning_rate'],
                        base_svm_C=experiment_config['base_svm_C'],
                        base_svm_max_iter=experiment_config['base_svm_max_iter'],
                        random_state=config.RANDOM_STATE
                    )
                else: # 基线线性SVM
                    model_obj = SVC(**experiment_config['base_params'])
                    start_time = time.time()
                    model_obj.fit(X_train_scaled, y_train)
                    train_time = time.time() - start_time
                    joblib.dump(model_obj, model_save_path)
                    print(f"  基线线性SVM模型已保存至: {model_save_path}")
        except Exception as e_train:
            print(f"错误: 训练 {experiment_name} 失败: {e_train}")
            return None
            
    # 步骤 4: 如果模型已成功加载或训练，则进行评估和记录
    if model_obj:
        try:
            print(f"  对 {experiment_name} 进行预测和评估...")
            y_pred = predict_with_model(model_obj, X_test_scaled)
            metrics_dict = calculate_classification_metrics(y_test, y_pred, model_name=experiment_name)

            # 准备记录
            record = {field: experiment_config.get(field, 'N/A') for field in csv_field_order}
            record.update({
                'training_time_seconds': round(train_time, 4) if train_time != -1.0 else 'N/A (Resumed)',
                'test_accuracy': metrics_dict['accuracy'],
                'test_f1_micro': metrics_dict['f1_micro'],
                'test_f1_macro': metrics_dict['f1_macro'],
                'test_f1_weighted': metrics_dict['f1_weighted']
            })
            record['experiment_name'] = experiment_name # 确保 experiment_name 正确
            record['base_learner_type'] = base_learner_type # 确保 base_learner_type 正确

            append_metrics_to_csv(metrics_csv_path, record, field_order=csv_field_order)
            print(f"  {experiment_name} 完成。测试准确率: {metrics_dict['accuracy']:.4f}")
            
            # 为这个模型绘制并保存混淆矩阵
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))
            cm_plot_filename = f"task2_cm_{safe_exp_name_for_file}.png"
            plot_confusion_matrix_heatmap(
                cm, class_names=[str(i) for i in range(10)],
                title=f"{experiment_name}\n混淆矩阵 (测试集)",
                save_path=plots_output_dir / cm_plot_filename,
                show_plot=False 
            )
            return record
        except Exception as e_eval:
            print(f"错误: 评估 {experiment_name} 失败: {e_eval}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"  由于模型训练和加载均失败，跳过 {experiment_name} 的评估。")
        return None

def main():
    """
    执行任务2：自定义AdaBoost，对比决策树桩和线性SVM作为基分类器的性能。
    """
    print("="*70)
    print("开始执行任务2：自定义AdaBoost性能对比与分析 (含续跑和多配置)")
    print("="*70)

    print("\n--- 步骤 0: 初始化路径、参数和CSV文件表头 ---")
    ensure_directories_exist_task2()

    PROCESSED_DIR = Path(config.PROCESSED_DATA_DIR)
    SCALER_PATH = Path(config.UTILS_DIR) / "mnist_standard_scaler.pkl"
    MODELS_SAVE_DIR = Path(config.TRAINED_MODEL_DIR)
    PLOTS_OUTPUT_DIR = Path(config.PLOTS_TASK2_DIR)
    METRICS_CSV_PATH_TASK2 = Path(config.METRICS_DIR) / "task2_adaboost_comparison_metrics.csv"

    CSV_FIELD_ORDER_TASK2 = [
        'experiment_name', 'base_learner_type', 
        'ada_n_estimators', 'ada_learning_rate', 
        'base_svm_C', 'base_svm_max_iter',
        'training_time_seconds',
        'test_accuracy', 'test_f1_micro', 'test_f1_macro', 'test_f1_weighted'
    ]
    print(f"  结果CSV文件: {METRICS_CSV_PATH_TASK2}")
    print(f"  图表保存目录: {PLOTS_OUTPUT_DIR}")

    RANDOM_STATE = getattr(config, 'RANDOM_STATE', 42)

    # --- 定义所有实验配置 ---
    experiment_configurations = []
    
    # 基线模型配置
    experiment_configurations.append({
        'experiment_name': 'Baseline Decision Stump', 'base_learner_type': 'Decision Stump',
        'base_params': {'max_depth': 1, 'random_state': RANDOM_STATE}
    })
    experiment_configurations.append({
        'experiment_name': 'Baseline Linear SVM (C=0.01)', 'base_learner_type': 'Linear SVM',
        'base_params': {'kernel': 'linear', 'C': 0.01, 'max_iter': 1000, 'probability': False, 'random_state': RANDOM_STATE}
    })

    # AdaBoost + 决策树桩 (学习曲线上的所有点)
    N_ESTIMATORS_STUMP_LIST_LC = [10, 20, 50, 70, 100] 
    FINAL_N_ESTIMATORS_STUMP = N_ESTIMATORS_STUMP_LIST_LC[-1]
    LEARNING_RATE_STUMP_LC = 1.0 
    for n_est in N_ESTIMATORS_STUMP_LIST_LC:
        experiment_configurations.append({
            'experiment_name': f"AdaBoost Stumps (N={n_est}, LR={LEARNING_RATE_STUMP_LC})",
            'base_learner_type': 'Decision Stump',
            'ada_n_estimators': n_est, 'ada_learning_rate': LEARNING_RATE_STUMP_LC
        })
    
    # AdaBoost + 线性SVM (所有配置)
    ALL_ADA_SVM_CONFIGS_PARAMS = [
        {'n_estimators': 10, 'learning_rate': 0.5, 'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N10_LR0.5_varyN'},
        {'n_estimators': 20, 'learning_rate': 0.5, 'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N20_LR0.5_varyN'},
        {'n_estimators': 40, 'learning_rate': 0.5, 'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N40_LR0.5_varyN'},
        {'n_estimators': 80, 'learning_rate': 0.5, 'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N80_LR0.5_varyN'},
        {'n_estimators': 10, 'learning_rate': 0.05, 'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N10_LR0.05_varyLR'},
        {'n_estimators': 10, 'learning_rate': 0.1,  'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N10_LR0.1_varyLR'},
        {'n_estimators': 10, 'learning_rate': 0.2,  'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N10_LR0.2_varyLR'},
        {'n_estimators': 10, 'learning_rate': 0.5,  'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N10_LR0.5_varyLR'},
        {'n_estimators': 10, 'learning_rate': 1.0,  'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N10_LR1.0_varyLR'},
        {'n_estimators': 10, 'learning_rate': 2.0,  'C': 0.1, 'max_iter': 50000, 'name_suffix': 'C0.1_N10_LR2.0_varyLR'}
    ]
    for svm_conf in ALL_ADA_SVM_CONFIGS_PARAMS:
        experiment_configurations.append({
            'experiment_name': f"AdaBoost SVM ({svm_conf['name_suffix']})",
            'base_learner_type': 'Linear SVM',
            'ada_n_estimators': svm_conf['n_estimators'], 
            'ada_learning_rate': svm_conf['learning_rate'],
            'base_svm_C': svm_conf['C'], 
            'base_svm_max_iter': svm_conf['max_iter']
        })
    print("步骤 0 完成。定义的实验配置已准备好。")

    # --- 1. 数据加载与预处理 ---
    print("\n--- 步骤 1: 加载数据并进行标准化 (含1:10子采样) ---")
    try:
        X_train_unscaled_full, y_train_full = load_processed_train_data(PROCESSED_DIR)
        X_test_unscaled, y_test = load_processed_test_data(PROCESSED_DIR)
        
        SUBSAMPLE_TRAIN_RATIO = 0.1 
        ACTUAL_TRAIN_SIZE_FULL = X_train_unscaled_full.shape[0]
        DESIRED_TRAIN_SAMPLE_SIZE = int(ACTUAL_TRAIN_SIZE_FULL * SUBSAMPLE_TRAIN_RATIO)

        if 0 < DESIRED_TRAIN_SAMPLE_SIZE < ACTUAL_TRAIN_SIZE_FULL:
            print(f"信息: 原始训练集大小为 {ACTUAL_TRAIN_SIZE_FULL}，将进行 {SUBSAMPLE_TRAIN_RATIO*100:.0f}% 子采样。")
            print(f"      目标子样本训练集大小为: {DESIRED_TRAIN_SAMPLE_SIZE}")
            try: random_seed_for_subsample = config.RANDOM_STATE 
            except AttributeError: random_seed_for_subsample = 42; print("警告: config.py 中未定义 RANDOM_STATE，子采样使用默认值 42。")
            np.random.seed(random_seed_for_subsample)
            shuffled_indices = np.random.permutation(ACTUAL_TRAIN_SIZE_FULL)
            subset_indices = shuffled_indices[:DESIRED_TRAIN_SAMPLE_SIZE]
            X_train_unscaled, y_train = X_train_unscaled_full[subset_indices], y_train_full[subset_indices]
        else:
            X_train_unscaled, y_train = X_train_unscaled_full, y_train_full
            print(f"信息: 使用完整的训练数据集 (大小: {ACTUAL_TRAIN_SIZE_FULL})。")
        print(f"  用于本次实验的训练数据形状: X={X_train_unscaled.shape}, y={y_train.shape}")
        
        scaler = load_scaler(SCALER_PATH)
        X_train_scaled = scaler.transform(X_train_unscaled)
        X_test_scaled = scaler.transform(X_test_unscaled)
        print(f"  数据加载、子采样和标准化完毕。最终训练集形状: {X_train_scaled.shape}, 测试集形状: {X_test_scaled.shape}")
    except Exception as e:
        print(f"错误 (步骤 1): 数据加载或标准化过程中发生错误: {e}")
        sys.exit(1)

    # --- 加载已有的实验结果CSV ---
    print("\n--- 步骤 A: 加载已有的实验结果CSV (如果存在) ---")
    try:
        existing_results_df = pd.read_csv(METRICS_CSV_PATH_TASK2) if METRICS_CSV_PATH_TASK2.exists() and METRICS_CSV_PATH_TASK2.stat().st_size > 0 else pd.DataFrame(columns=['experiment_name'])
        print(f"  已从 '{METRICS_CSV_PATH_TASK2}' 加载 {len(existing_results_df)} 条已有结果。")
    except Exception as e_read_csv:
        print(f"  警告: 读取结果CSV文件 '{METRICS_CSV_PATH_TASK2}' 失败: {e_read_csv}。将执行所有实验。")
        existing_results_df = pd.DataFrame(columns=['experiment_name'])

    # --- 迭代执行所有定义的实验配置 ---
    all_metrics_records = []
    for exp_conf in experiment_configurations:
        record = run_single_experiment(
            experiment_config=exp_conf, X_train_scaled=X_train_scaled, y_train=y_train,
            X_test_scaled=X_test_scaled, y_test=y_test, metrics_csv_path=METRICS_CSV_PATH_TASK2,
            csv_field_order=CSV_FIELD_ORDER_TASK2, models_save_dir=MODELS_SAVE_DIR,
            plots_output_dir=PLOTS_OUTPUT_DIR, existing_results_df=existing_results_df
        )
        if record:
            all_metrics_records.append(record)

    # --- 绘图阶段 (基于所有有效结果) ---
    print("\n" + "="*30 + " 开始生成汇总图表 " + "="*30)
    if not all_metrics_records:
        print("警告: 没有有效的实验结果可供绘图。")
    else:
        results_df_for_plotting = pd.DataFrame(all_metrics_records).drop_duplicates(subset=['experiment_name'], keep='last')
        
        # 1. AdaBoost + 决策树桩 学习曲线
        stump_lc_df = results_df_for_plotting[
            results_df_for_plotting['experiment_name'].str.contains("AdaBoost Stumps", regex=False, na=False)
        ].sort_values(by='ada_n_estimators')
        if len(stump_lc_df) >= 2:
            plot_learning_curve(
                param_values=stump_lc_df['ada_n_estimators'].tolist(),
                metric_scores_dict={'测试集准确率': stump_lc_df['test_accuracy'].tolist(), '测试集F1 Macro': stump_lc_df['test_f1_macro'].tolist()},
                title=f"AdaBoost (决策树桩) 学习曲线 (LR={LEARNING_RATE_STUMP_LC})",
                xlabel="基学习器数量 (n_estimators)", ylabel="性能指标值",
                save_path=PLOTS_OUTPUT_DIR / f"task2_lc_adaboost_stumps_lr{str(LEARNING_RATE_STUMP_LC).replace('.', 'p')}.png", show_plot=False
            )
        else: print("  未能收集到足够数据点绘制AdaBoost+树桩的学习曲线。")

        # 2. AdaBoost + 线性SVM 学习曲线 (vs n_estimators)
        svm_lc_vs_n_df = results_df_for_plotting[
            results_df_for_plotting['experiment_name'].str.contains("varyN", regex=False, na=False)
        ].sort_values(by='ada_n_estimators')
        if len(svm_lc_vs_n_df) >= 2:
            plot_learning_curve(
                param_values=svm_lc_vs_n_df['ada_n_estimators'].tolist(),
                metric_scores_dict={'测试集准确率': svm_lc_vs_n_df['test_accuracy'].tolist(), '测试集F1 Macro': svm_lc_vs_n_df['test_f1_macro'].tolist()},
                title="AdaBoost (线性SVM, C=0.1, LR=0.5) vs 基学习器数量",
                xlabel="基学习器数量 (n_estimators)", ylabel="性能指标值",
                save_path=PLOTS_OUTPUT_DIR / "task2_lc_adaboost_svm_vs_n_estimators.png", show_plot=False
            )
        else: print("  未能收集到足够数据点绘制AdaBoost+SVM (vs N_estimators)的学习曲线。")
        
        # 3. AdaBoost + 线性SVM 学习曲线 (vs learning_rate)
        svm_lc_vs_lr_df = results_df_for_plotting[
            results_df_for_plotting['experiment_name'].str.contains("varyLR", regex=False, na=False)
        ].sort_values(by='ada_learning_rate')
        if len(svm_lc_vs_lr_df) >= 2:
            plot_learning_curve(
                param_values=svm_lc_vs_lr_df['ada_learning_rate'].tolist(),
                metric_scores_dict={'测试集准确率': svm_lc_vs_lr_df['test_accuracy'].tolist(), '测试集F1 Macro': svm_lc_vs_lr_df['test_f1_macro'].tolist()},
                title="AdaBoost (线性SVM, C=0.1, N_est=10) vs 学习率",
                xlabel="学习率 (learning_rate)", ylabel="性能指标值",
                save_path=PLOTS_OUTPUT_DIR / "task2_lc_adaboost_svm_vs_learning_rate.png", show_plot=False
            )
        else: print("  未能收集到足够数据点绘制AdaBoost+SVM (vs Learning Rate)的学习曲线。")

        # 4. 最终性能对比条形图
        representative_exp_names = [
           'Baseline Decision Stump', 
           'Baseline Linear SVM (C=0.01)',
           f"AdaBoost Stumps (N={FINAL_N_ESTIMATORS_STUMP}, LR={LEARNING_RATE_STUMP_LC})",
           # 从 AdaBoost SVM 配置中选择几个有代表性的进行最终对比
           'AdaBoost SVM (C0.1_N80_LR0.5_varyN)', 
           'AdaBoost SVM (C0.1_N10_LR0.05_varyLR)'
        ]
        final_comparison_df = results_df_for_plotting[results_df_for_plotting['experiment_name'].isin(representative_exp_names)]

        if not final_comparison_df.empty:
            plot_names = final_comparison_df['experiment_name'].tolist()
            plot_accuracies = final_comparison_df['test_accuracy'].astype(float).tolist()
            plot_f1s = final_comparison_df['test_f1_macro'].astype(float).tolist()
            plot_times_str = final_comparison_df['training_time_seconds'].tolist()
            plot_times_numeric = [float(t) if str(t).replace('.', '', 1).isdigit() else 0 for t in plot_times_str]
            
            plot_bar_comparison(
                x_labels=plot_names, y_values=plot_accuracies,
                title="任务2 模型最终准确率对比", xlabel="模型配置", ylabel="准确率",
                save_path=PLOTS_OUTPUT_DIR / "task2_final_comparison_accuracy.png",
                figure_size=(max(10, len(plot_names) * 2), 8), show_plot=False
            )
            plot_bar_comparison(
                x_labels=plot_names, y_values=plot_f1s,
                title="任务2 模型最终F1 Macro对比", xlabel="模型配置", ylabel="F1 Macro",
                save_path=PLOTS_OUTPUT_DIR / "task2_final_comparison_f1_macro.png",
                figure_size=(max(10, len(plot_names) * 2), 8), show_plot=False
            )
            plot_bar_comparison(
                x_labels=plot_names, y_values=plot_times_numeric,
                title="任务2 模型最终训练时间对比", xlabel="模型配置", ylabel="训练时间 (秒)",
                save_path=PLOTS_OUTPUT_DIR / "task2_final_comparison_training_time.png",
                figure_size=(max(10, len(plot_names) * 2), 8), show_plot=False
            )
            print("  最终对比图表已生成。")
        else:
            print("警告: 未能筛选出足够的代表性实验结果进行最终对比绘图。")

    print("\n" + "="*70)
    print("任务2：自定义AdaBoost性能对比与分析脚本执行完毕！")
    print(f"  性能指标已保存至: {METRICS_CSV_PATH_TASK2}")
    print(f"  图表已保存至: {PLOTS_OUTPUT_DIR}")
    print(f"  训练好的模型（如果适用）已保存至: {MODELS_SAVE_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()