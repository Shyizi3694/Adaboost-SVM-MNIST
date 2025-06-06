# 数学建模与分析编程：基于SVM的AdaBoost增强手写数字分类

本项目旨在通过实践加深对支持向量机 (SVM) 和 AdaBoost 集成学习算法的理解。主要任务包括比较不同核函数的SVM性能，以及从零开始实现AdaBoost算法并对比其在不同基学习器下的表现。

## 1. 文件结构

```text
PROJECT_ROOT/
├── data/                     # 存放所有数据
│   ├── raw/                  # 原始下载数据 (例如 MNIST 缓存)
│   └── processed/            # 预处理后的数据 (X_train.npy, y_train.npy - 已划分但未标准化)
│
├── results/                  # 存放实验结果
│   ├── metrics/              # 保存性能指标的CSV文件
│   ├── plots/                # 保存生成的图表
│   │   ├── task1/            # 任务1 (SVM对比) 的图表
│   │   ├── task2/            # 任务2 (AdaBoost) 的图表
│   │   └── evaluation_summary_plots/ # 快速评估脚本生成的总结图表
│   └── trained_models/       # 保存训练好的模型对象 (.pkl 文件)
│
├── src/                      # 存放所有源代码
│   ├── __init__.py
│   ├── config.py             # 项目配置文件 (路径、常量等)
│   │
│   ├── data_preprocess/      # 数据预处理模块
│   │   ├── __init__.py
│   │   ├── loader.py         # 数据加载函数
│   │   └── processor.py      # 数据处理函数 (划分、标准化规则生成与保存)
│   │
│   ├── models/               # 模型实现与训练逻辑
│   │   ├── __init__.py
│   │   ├── custom_adaboost.py  # 自定义 AdaBoost 算法 (含二分类和OvR多分类)
│   │   ├── linear_svm_trainer.py # 线性SVM训练函数
│   │   ├── rbf_svm_trainer.py    # RBF核SVM训练函数
│   │   └── adaboost_trainer.py # 自定义AdaBoost训练函数 (调用 custom_adaboost)
│   │
│   ├── common/               # 核心/通用工具代码模块
│   │   ├── __init__.py
│   │   ├── evaluator.py      # 评估指标计算函数
│   │   ├── prediction.py     # 模型预测函数
│   │   ├── plotting.py       # 绘图辅助函数
│   │   └── result_utils.py   # 结果保存辅助函数 (例如追加到CSV)
│   │
│   ├── prepare_data.py       # 一次性数据准备脚本 (下载、处理、保存scaler)
│   ├── run_task1_svm.py      # 执行任务1 (SVM对比) 的主脚本
│   ├── run_task2_adaboost.py # 执行任务2 (AdaBoost) 的主脚本
│   └── run_evaluation.py     # 快速评估已训练最优模型的脚本 (供验收使用)
│
├── utils/                    # (对应 config.UTILS_DIR) 存放辅助性的、非代码的持久化文件
│   └── mnist_standard_scaler.pkl # 例如保存的 StandardScaler 对象
│
├── notebooks/                # (可选) Jupyter Notebooks 用于探索性分析、可视化等
│
├── requirements.txt          # 项目依赖的 Python 包列表
├── evaluate_project.sh       # (Linux/macOS) 验收用一键评估脚本
├── evaluate_project.bat      # (Windows) 验收用一键评估脚本
└── README.md                 # 本文件 - 项目说明
```

## 2. 环境准备

本项目建议在 Python 3.8 或更高版本下运行。

### 2.1 创建并激活虚拟环境 (推荐)

- 使用 `venv`:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

- 或使用 `conda`：

```bash
conda create -n myenv python=3.9
conda activate myenv
```

### 2.2 安装项目依赖

项目根目录下提供了 `requirements.txt` 文件。请在激活虚拟环境后，运行以下命令安装所有必要的库：

```bash
pip install -r requirements.txt
```

## 3. 运行方法

提供了多种运行项目和查看结果的方式：

### 3.1 便捷评估脚本

这些脚本会自动执行完整的评估流程，包括安装依赖（如果requirements.txt存在）、准备数据（下载、预处理、保存scaler），然后加载预先训练并提供的最优模型进行评估，并显示/保存结果。

- 请确保已将为任务1和任务2训练好的“最优”模型文件（.pkl格式）放置在 results/trained_models/ 目录下。
- 请确保 src/run_evaluation.py 脚本中引用的模型文件名与这些文件名一致。
- Windows 用户:
直接双击运行项目根目录下的 evaluate_project.bat 文件，或在命令提示符中执行：

```bash
.\evaluate_project.bat
```

- macOS / Linux 用户:
首先给予脚本执行权限，然后运行：

```bash
chmod +x evaluate_project.sh
./evaluate_project.sh
```

### 3.2 分步执行 Python 脚本

如果想分步执行或重新运行项目的特定部分（例如数据准备、模型训练等），可以使用以下命令（请确保已激活 Python 环境并安装了依赖）。所有命令都建议在项目根目录下执行。

1. 数据准备 (通常只需运行一次)：
此脚本负责下载原始 MNIST 数据（如果本地没有缓存）、进行数据划分和保存（保存到 `data/processed/`），以及拟合 `StandardScaler` 并将其保存到 `utils/` 目录。

```bash
python -m src.prepare_data
```

2. 单独运行模型评估 (加载预训练模型)：

可以使用此脚本快速加载它们并查看评估结果。

```bash
python -m src.run_evaluation
```

3. 任务1: SVM对比实验 (含超参数调优和模型训练)：
此脚本会加载预处理数据和scaler，为线性核SVM和RBF核SVM进行超参数调优，训练最终模型，保存模型，评估性能，并将指标和图表保存到 `results/` 目录下。

```bash
python -m src.run_task1_svm
```

(注意：此脚本包含GridSearchCV，运行时间可能较长，如您之前所体验，线性SVM调优约40分钟，RBF核调优会更长。)

4. 任务2: 自定义AdaBoost实验 (含模型训练)：
此脚本会加载预处理数据和scaler，使用您从零实现的AdaBoost算法（分别以决策树桩和弱线性SVM为基学习器）进行训练和评估，保存模型、指标和图表。

```bash
python -m src.run_task2_adaboost_
```

这个脚本自带续跑功能，如果脚本在运行时因意外产生外部中断，可以直接重新运行该脚本，脚本会自动追踪 `task2_adaboost_comparison_metrics.csv` 中已经记录的子实验，并检查 `results/trained_models` 中已经存在的脚本，从断点处开始重新运行。

### 3.3 直接运行模块进行单元测试

项目中的一些 `.py` 文件（例如 `loader.py`, `processor.py`, `custom_adaboost.py`, `result_utils.py`, `evaluator.py`, `plotting.py`, `adaboost_trainer.py` 等）可能包含 `if __name__ == "__main__":` 测试块，用于对其内部函数进行单元测试或功能演示。您可以这样运行它们（示例）：

```bash
python -m src.data_preprocess.loader
python -m src.models.custom_adaboost
```

## 4. 模型与结果文件位置

- 预处理数据 (Preprocessed Data):
    - 原始数据缓存: `data/raw/` (由 `sklearn.datasets.fetch_openml` 管理)
    - 划分后、未标准化的数据: `data/processed/` (例如 `X_train.npy`, `y_train.npy` 等)
- 辅助工具 (Utility Files):
    - StandardScaler 对象: `utils/mnist_standard_scaler.pkl`
- 训练好的模型 (Trained Models):
    - 所有由 `run_task1_svm.py` 和 `run_task2_adaboost.py` 训练并保存的模型，以及供 `run_evaluation.py` 加载的“最优”模型，都应存放在： `results/trained_models/`
    - 示例文件名 (请根据您实际保存的名称进行更新，尤其是在 `run_evaluation.py` 中指定的)：
        - `task1_tuned_linear_svm.pkl`
        - `task1_tuned_rbf_svm.pkl`
        - `task2_adaboost_stumps_final_N[具体迭代次数].pkl` (例如 `task2_adaboost_stumps_final_N300.pkl`)
        - `task2_adaboost_svm_[具体配置名].pkl` (例如 `task2_adaboost_svm_C0.01_N30_LR0.5.pkl`)
- 性能指标 (Performance Metrics):
    - 任务1 SVM对比指标: `results/metrics/task1_svm_comparison_metrics.csv`
    - 任务2 AdaBoost对比指标: `results/metrics/task2_adaboost_comparison_metrics.csv`
- 图表 (Plots):
    - 任务1图表: `results/plots/task1/`
    - 任务2图表: `results/plots/task2/`
    - `run_evaluation.py` 生成的总结图表: `results/plots/evaluation_summary_plots/`