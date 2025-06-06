# src/common/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any # 引入 Any
import pandas as pd

# 尝试设置全局绘图风格和中文字体支持
try:
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 常用的支持中文的字体，例如“黑体”
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    print("信息: Matplotlib 中文字体已尝试设置为 'SimHei'。")
except Exception as e:
    print(f"警告: 设置中文字体 'SimHei' 失败，图表中的中文可能无法正确显示: {e}")
    print("      如需显示中文，请确保系统中安装了 'SimHei' 或其他支持中文的字体，并正确配置了 Matplotlib。")

def plot_bar_comparison(x_labels: List[str], 
                        y_values: List[float], 
                        title: str, 
                        xlabel: str, 
                        ylabel: str, 
                        save_path: Path,
                        show_plot: bool = False,
                        y_limit: Optional[Tuple[float, float]] = None,
                        figure_size: Tuple[float, float] = (8, 6)):
    """
    绘制并保存条形对比图。

    输入:
        x_labels (List[str]): X轴上每个条形的标签。
        y_values (List[float]): 每个条形对应的值。
        title (str): 图表标题。
        xlabel (str): X轴标签。
        ylabel (str): Y轴标签。
        save_path (pathlib.Path): 图表保存的完整路径。
        y_limit (Optional[Tuple[float, float]]): Y轴的显示范围，例如 (0.0, 1.0)。
        figure_size (Tuple[float, float]): 图表大小。
    """
    if len(x_labels) != len(y_values):
        raise ValueError("x_labels 和 y_values 的长度必须一致。")

    plt.figure(figsize=figure_size)
    bars = plt.bar(x_labels, y_values, color=sns.color_palette("muted", len(x_labels)))
    
    # 在条形图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * (y_limit[1] if y_limit else max(y_values, default=1)), f'{yval:.4f}', ha='center', va='bottom')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if y_limit:
        plt.ylim(y_limit)
    
    plt.xticks(rotation=45, ha="right") # 如果x轴标签较长，旋转一下
    plt.tight_layout() # 调整布局以防止标签重叠

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True) # 确保保存目录存在
        plt.savefig(save_path, dpi=300)
        print(f"条形对比图已保存至: {save_path}")
    except Exception as e:
        print(f"错误: 保存条形对比图 '{save_path}' 失败: {e}")
    finally:
        plt.close() # 关闭图表以释放内存
    
    if show_plot:
        plt.show() # 如果需要显示图表，则显示
        


def plot_grouped_bar_comparison(category_labels: List[str], 
                                metric_names: List[str],
                                values_matrix: np.ndarray, # shape: (n_categories, n_metrics)
                                title: str, 
                                xlabel: str, # 通常是 '模型' 或 '类别'
                                ylabel: str, # 通常是 '指标值'
                                save_path: Path,
                                show_plot: bool = False,
                                figure_size: Tuple[float, float] = (12, 7)):
    """
    绘制并保存分组条形对比图。

    输入:
        category_labels (List[str]): X轴上的主要类别标签。
        metric_names (List[str]): 每个类别下分组的指标名称 (构成不同的颜色条)。
        values_matrix (np.ndarray): 一个二维数组，values_matrix[i, j] 表示 
                                    第 i 个类别下第 j 个指标的值。
        title (str): 图表标题。
        xlabel (str): X轴标签。
        ylabel (str): Y轴标签。
        save_path (pathlib.Path): 图表保存的完整路径。
        figure_size (Tuple[float, float]): 图表大小。
    """
    if values_matrix.shape != (len(category_labels), len(metric_names)):
        raise ValueError(f"values_matrix 的形状 ({values_matrix.shape}) 与 "
                         f"category_labels ({len(category_labels)}) 和 "
                         f"metric_names ({len(metric_names)}) 的长度不匹配。")

    n_categories = len(category_labels)
    n_metrics = len(metric_names)
    
    x = np.arange(n_categories)  # 每组条形的中心位置
    width = 0.8 / n_metrics     # 每个条形的宽度，总宽度为0.8以便留出间隔
    
    plt.figure(figsize=figure_size)
    ax = plt.gca()

    for i in range(n_metrics):
        # 计算当前指标条形组的偏移量
        offset = width * (i - (n_metrics - 1) / 2.0)
        rects = ax.bar(x + offset, values_matrix[:, i], width, label=metric_names[i])
        ax.bar_label(rects, fmt='%.4f', padding=3) # 在条形上显示数值

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    ax.legend(title="指标")
    
    plt.tight_layout()

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"分组条形对比图已保存至: {save_path}")
    except Exception as e:
        print(f"错误: 保存分组条形对比图 '{save_path}' 失败: {e}")
    finally:
        plt.close()

    if show_plot:
        plt.show()

def plot_learning_curve(param_values: List[Any], 
                        metric_scores_dict: Dict[str, List[float]], 
                        title: str, 
                        xlabel: str, 
                        ylabel: str,
                        save_path: Path,
                        show_plot: bool = False,
                        figure_size: Tuple[float, float] = (10, 6)):
    """
    绘制并保存学习曲线或参数影响曲线。

    输入:
        param_values (List[Any]): X轴上的参数值 (例如迭代次数，可以是数字或字符串)。
        metric_scores_dict (Dict[str, List[float]]): 字典，键为曲线标签 (例如 '训练集准确率'), 
                                                    值为对应的性能指标列表。
        title (str): 图表标题。
        xlabel (str): X轴标签。
        ylabel (str): Y轴标签。
        save_path (pathlib.Path): 图表保存的完整路径。
        figure_size (Tuple[float, float]): 图表大小。
    """
    plt.figure(figsize=figure_size)
    
    for label, scores in metric_scores_dict.items():
        if len(param_values) != len(scores):
            print(f"警告: 对于曲线 '{label}'，参数数量 ({len(param_values)}) 与分数数量 ({len(scores)}) 不匹配。跳过绘制此曲线。")
            continue
        plt.plot(param_values, scores, marker='o', linestyle='-', label=label)
        # 可以在每个点上标注数值，但如果点很多会显得拥挤
        # for i, score in enumerate(scores):
        #     plt.text(param_values[i], score, f'{score:.3f}', ha='center', va='bottom')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 如果x轴是数字，可以考虑设置合适的刻度
    if all(isinstance(p, (int, float)) for p in param_values):
        pass # Matplotlib 通常会自动处理
    else: # 如果x轴是类别型（字符串）
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"学习曲线已保存至: {save_path}")
    except Exception as e:
        print(f"错误: 保存学习曲线 '{save_path}' 失败: {e}")
    finally:
        plt.close()

    if show_plot:
        plt.show()

def plot_confusion_matrix_heatmap(cm: np.ndarray, 
                                  class_names: List[str],
                                  title: str, 
                                  save_path: Path, 
                                  show_plot: bool = False,
                                  normalize: bool = False, 
                                  cmap: Any = plt.cm.Blues, # cmap 类型可以是 matplotlib.colors.Colormap
                                  figure_size: Optional[Tuple[float, float]] = None):
    """
    绘制并保存混淆矩阵热力图。

    输入:
        cm (np.ndarray): 混淆矩阵。
        class_names (List[str]): 类别名称列表，用于标记坐标轴。
        title (str): 图表标题。
        save_path (pathlib.Path): 图表保存的完整路径。
        normalize (bool): 是否将混淆矩阵的值归一化到 [0, 1] (显示百分比)。
        cmap (matplotlib.colors.Colormap): 热力图的颜色映射。
        figure_size (Optional[Tuple[float, float]]): 图表大小。如果为None，则自动调整。
    """
    if normalize:
        # 避免除以0的情况 (如果某行总和为0，即该真实类别没有样本)
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        # 对于总和为0的行，保持原样（或设为0），避免NaN
        cm_normalized = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
        fmt = '.2f' # 归一化后用两位小数格式化
        plot_cm = cm_normalized
    else:
        fmt = 'd' # 未归一化用整数格式化
        plot_cm = cm

    if figure_size is None:
        # 根据类别数量动态调整图表大小，使其更易读
        num_classes = len(class_names)
        fig_width = max(8, num_classes * 0.8) 
        fig_height = max(6, num_classes * 0.6)
        figure_size = (fig_width, fig_height)

    plt.figure(figsize=figure_size)
    sns.heatmap(plot_cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.5, cbar_kws={"shrink": .8}) # 调整 colorbar 大小
    
    plt.title(title, fontsize=16, pad=20) # 增加标题和图的间距
    plt.ylabel('真实标签 (True Label)', fontsize=12)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"混淆矩阵热力图已保存至: {save_path}")
    except Exception as e:
        print(f"错误: 保存混淆矩阵 '{save_path}' 失败: {e}")
    finally:
        plt.close()

    if show_plot:
        plt.show()

# --- 新增函数 ---
def plot_grid_search_heatmap(cv_results: Dict[str, Any],
                             param_x_name: str,
                             param_y_name: str,
                             title: str,
                             save_path: Path,
                             show_plot: bool = False,
                             score_key: str = 'mean_test_score',
                             cmap: str = "viridis", # Viridis通常对色盲友好
                             figure_size: Optional[Tuple[float, float]] = (10, 8)):
    """
    从 GridSearchCV 的 cv_results_ 生成并保存参数网格搜索结果的热力图。
    主要用于可视化二维超参数空间中的性能。

    输入:
        cv_results (Dict[str, Any]): GridSearchCV对象的cv_results_属性。
        param_x_name (str): 在热力图x轴上表示的超参数名称 (例如 'C')。
                           函数内部会自动添加 'param_' 前缀。
        param_y_name (str): 在热力图y轴上表示的超参数名称 (例如 'gamma')。
                           函数内部会自动添加 'param_' 前缀。
        title (str): 图表标题。
        save_path (pathlib.Path): 图表保存的完整路径。
        score_key (str, optional): cv_results_中用于性能得分的键名，
                                   默认为 'mean_test_score'。
        cmap (str, optional): 热力图的颜色映射。
        figure_size (Optional[Tuple[float, float]]): 图表大小。
    """
    try:
        df_cv_results = pd.DataFrame(cv_results)

        param_x_col = f'param_{param_x_name}'
        param_y_col = f'param_{param_y_name}'

        required_cols = [param_x_col, param_y_col, score_key]
        missing_cols = [col for col in required_cols if col not in df_cv_results.columns]
        if missing_cols:
            print(f"错误 (plot_grid_search_heatmap): cv_results 中缺少必要的列: {', '.join(missing_cols)}")
            print(f"      可用列包括: {df_cv_results.columns.tolist()}")
            return

        # 将参数列转换为数值型，如果它们是对象类型但表示数值 (例如 GridSearchCV 可能将数值参数变为对象类型)
        # 同时处理像 'scale' 这样的字符串值，它们将作为类别。
        # pivot_table 通常能处理混合类型作为索引/列，但确保数值参数真正是数值型用于排序和显示可能更好。
        # 这里我们依赖 pivot_table 的能力。

        pivot_table = df_cv_results.pivot_table(values=score_key,
                                                index=param_y_col, # y-axis
                                                columns=param_x_col) # x-axis
        
        # 对索引和列进行排序，确保热力图的轴是有序的 (如果它们是数值型)
        # 如果包含字符串（如 'scale'），混合排序可能复杂，pivot_table默认的顺序通常是合理的
        try: # 尝试数值排序
            pivot_table.index = pd.to_numeric(pivot_table.index)
            pivot_table = pivot_table.sort_index()
        except ValueError: # 如果包含非数值，保持原样
            pass 
        try: # 尝试数值排序
            pivot_table.columns = pd.to_numeric(pivot_table.columns)
            pivot_table = pivot_table.sort_index(axis=1)
        except ValueError:
            pass


        actual_fig_size = figure_size
        if actual_fig_size is None:
             actual_fig_size = (max(8, len(pivot_table.columns) * 1.0), max(6, len(pivot_table.index) * 0.7))


        plt.figure(figsize=actual_fig_size)
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap=cmap, linewidths=.5, cbar_kws={"shrink": .8, "label": score_key})
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel(f"参数: {param_x_name}", fontsize=12)
        plt.ylabel(f"参数: {param_y_name}", fontsize=12)
        
        # 获取刻度标签并尝试转换为字符串，避免Matplotlib对非数字类型的警告/错误
        plt.xticks(ticks=np.arange(len(pivot_table.columns)) + 0.5, labels=[str(tick) for tick in pivot_table.columns], rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(pivot_table.index)) + 0.5, labels=[str(tick) for tick in pivot_table.index], rotation=0, va="center")

        plt.tight_layout(pad=0.5)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"网格搜索热力图已保存至: {save_path}")

    except KeyError as e:
        print(f"错误 (plot_grid_search_heatmap): cv_results 中找不到指定的参数名或分数键: {e}")
    except Exception as e:
        print(f"错误 (plot_grid_search_heatmap): 生成网格搜索热力图 '{save_path}' 失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close()

    if show_plot:
        plt.show()


# ==========================
# 测试程序 (Test Harness)
# ==========================
if __name__ == '__main__':
    print(">>> 测试 plotting.py 中的绘图函数 <<<")
    
    current_file_dir = Path(__file__).parent
    TEST_PLOTS_DIR_PLOTTING_PY = current_file_dir / "temp_plotting_module_tests_output" # 与result_utils区分
    
    try:
        from src import config 
        if hasattr(config, 'PLOTS_DIR'):
             TEST_PLOTS_DIR_PLOTTING_PY = Path(config.PLOTS_DIR) / "plotting_module_tests_output"
        else:
            print(f"警告: config.py 中未找到 PLOTS_DIR，测试图片将保存在本地 '{TEST_PLOTS_DIR_PLOTTING_PY}'")
    except (ImportError, AttributeError) as e:
        print(f"警告: 无法从 src.config 导入 PLOTS_DIR (错误: {e})。测试图片将保存在本地 '{TEST_PLOTS_DIR_PLOTTING_PY}'")

    TEST_PLOTS_DIR_PLOTTING_PY.mkdir(parents=True, exist_ok=True)
    print(f"测试图片将保存到: {TEST_PLOTS_DIR_PLOTTING_PY}\n")

    # 1. 测试 plot_bar_comparison
    print("--- 测试 1: plot_bar_comparison ---")
    bar_labels = ['模型 A', '模型 B', '模型 C']
    bar_values = [0.85, 0.92, 0.78]
    plot_bar_comparison(
        x_labels=bar_labels, y_values=bar_values,
        title="模型性能对比 (准确率)", xlabel="模型", ylabel="准确率",
        save_path=TEST_PLOTS_DIR_PLOTTING_PY / "test_bar_comparison_accuracy.png",
        y_limit=(0.7, 1.0)
    )

    # 2. 测试 plot_grouped_bar_comparison
    print("\n--- 测试 2: plot_grouped_bar_comparison ---")
    group_cat_labels = ['线性SVM', 'RBF SVM']
    group_metric_names = ['准确率', 'F1宏平均', '训练时间(s)']
    group_values = np.array([
        [0.91, 0.90, 10.5],  
        [0.95, 0.94, 65.2]  
    ])
    plot_grouped_bar_comparison(
        category_labels=group_cat_labels, metric_names=group_metric_names,
        values_matrix=group_values, title="SVM模型多指标对比",
        xlabel="SVM 类型", ylabel="指标值",
        save_path=TEST_PLOTS_DIR_PLOTTING_PY / "test_grouped_bar_svm_metrics.png"
    )

    # 3. 测试 plot_learning_curve
    print("\n--- 测试 3: plot_learning_curve ---")
    iterations = [10, 50, 100, 150, 200]
    lc_metrics = {
        'AdaBoost (树桩) 测试集': [0.75, 0.85, 0.88, 0.90, 0.91],
        'AdaBoost (树桩) 训练集': [0.80, 0.90, 0.93, 0.95, 0.96]
    }
    plot_learning_curve(
        param_values=iterations, metric_scores_dict=lc_metrics,
        title="AdaBoost 学习曲线 (基于迭代次数)", xlabel="迭代次数 (n_estimators)", ylabel="准确率",
        save_path=TEST_PLOTS_DIR_PLOTTING_PY / "test_learning_curve_adaboost.png"
    )

    # 4. 测试 plot_confusion_matrix_heatmap
    print("\n--- 测试 4: plot_confusion_matrix_heatmap ---")
    cm_sample = np.array([[100, 10, 5], [8,  120, 2], [3,  7,  90]])
    class_names_sample = ['类别0', '类别1', '类别2']
    plot_confusion_matrix_heatmap(
        cm=cm_sample, class_names=class_names_sample, title="混淆矩阵 (原始计数)",
        save_path=TEST_PLOTS_DIR_PLOTTING_PY / "test_confusion_matrix_raw.png"
    )
    plot_confusion_matrix_heatmap(
        cm=cm_sample, class_names=class_names_sample, title="混淆矩阵 (归一化)",
        save_path=TEST_PLOTS_DIR_PLOTTING_PY / "test_confusion_matrix_normalized.png", normalize=True
    )

    # 5. 测试 plot_grid_search_heatmap
    print("\n--- 测试 5: plot_grid_search_heatmap ---")
    # 模拟 GridSearchCV 的 cv_results_
    # 注意：实际的 cv_results_ 会有更多列，这里只模拟必要的
    mock_cv_results = {
        'param_C': np.array([0.1, 0.1, 1, 1, 10, 10]),
        'param_gamma': np.array([0.01, 0.1, 0.01, 0.1, 0.01, 0.1]),
        'mean_test_score': np.array([0.80, 0.85, 0.88, 0.92, 0.90, 0.91]),
        'rank_test_score': np.array([6,4,3,1,2,5]) # 示例，实际可能更多
    }
    # 测试当 gamma 包含字符串时的情况
    mock_cv_results_mixed_gamma = {
        'param_C': np.array([1, 1, 1, 10, 10, 10]),
        'param_gamma': np.array([0.01, 0.1, 'scale', 0.01, 0.1, 'scale'], dtype=object), # gamma 含字符串
        'mean_test_score': np.array([0.88, 0.92, 0.93, 0.90, 0.91, 0.94]),
    }
    plot_grid_search_heatmap(
        cv_results=mock_cv_results, param_x_name='C', param_y_name='gamma',
        title="网格搜索热力图 (C vs gamma)",
        save_path=TEST_PLOTS_DIR_PLOTTING_PY / "test_grid_search_heatmap.png"
    )
    plot_grid_search_heatmap(
        cv_results=mock_cv_results_mixed_gamma, param_x_name='C', param_y_name='gamma',
        title="网格搜索热力图 (C vs gamma, 含 'scale')",
        save_path=TEST_PLOTS_DIR_PLOTTING_PY / "test_grid_search_heatmap_mixed_gamma.png"
    )


    print("\n>>> plotting.py 函数测试结束 <<<")
    print(f"请检查 '{TEST_PLOTS_DIR_PLOTTING_PY}' 目录下的输出图片。")
    # 提示：如果 TEST_PLOTS_DIR_PLOTTING_PY 指向 config.PLOTS_DIR 下的子目录，
    # 并且 config.py 无法正确导入，请确保运行此脚本的方式能让Python找到src包，
    # 例如在项目根目录下使用 python -m src.common.plotting