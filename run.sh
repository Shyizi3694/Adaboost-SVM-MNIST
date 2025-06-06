#!/bin/bash

# 项目评估流程启动脚本 (Linux/macOS)

echo "=============================================================="
echo "   开始运行机器学习大作业 - 成果评估脚本   "
echo "=============================================================="
echo ""
echo "本脚本将执行以下操作："
echo "  1. 安装项目依赖 (根据 requirements.txt)。" # 新增步骤
echo "  2. 准备数据：下载MNIST数据（如果需要），进行预处理，并保存scaler。"
echo "  3. 评估模型：加载预训练的最优模型，在测试集上评估并显示/保存结果。"
echo ""
echo "重要提示："
echo "  - 请确保已激活包含所有项目依赖库的Python环境 (建议使用虚拟环境)。"
echo "  - 'requirements.txt' 文件应位于项目根目录。"
echo "  - 请确保您已将训练好的'最优'模型文件放置在 'results/trained_models/' 目录下。"
echo "  - 'src/run_evaluation.py' 脚本中引用的模型文件名需与您提供的文件匹配。"
echo "=============================================================="
echo ""

# (可选) 切换到脚本所在目录 (假设脚本放在项目根目录)
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# cd "$SCRIPT_DIR"
# echo "当前工作目录: $(pwd)"
# echo ""

# 步骤 1: 安装项目依赖
echo ">>> 步骤 1: 安装项目依赖 (根据 requirements.txt)..."
echo "--------------------------------------------------------------"
if [ -f "requirements.txt" ]; then
    echo "  找到 requirements.txt，开始安装/更新依赖包..."
    python -m pip install -r requirements.txt # 使用 python -m pip 更健壮
    if [ $? -ne 0 ]; then
        echo ""
        echo "错误: 依赖包安装失败！"
        echo "请检查 requirements.txt 文件内容以及您的 pip 和 Python 环境。"
        exit 1
    fi
    echo "  依赖包安装/检查完毕。"
else
    echo ""
    echo "警告: 'requirements.txt' 文件未在项目根目录找到！"
    echo "      将跳过依赖自动安装步骤。请确保您已手动安装所有必要的依赖包。"
fi
echo "--------------------------------------------------------------"
echo "依赖安装步骤结束。"
echo ""
echo ""

# 步骤 2: 数据准备 (原步骤1)
echo ">>> 步骤 2: 执行数据准备脚本 (src.prepare_data)..."
echo "--------------------------------------------------------------"
python -m src.prepare_data
if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 数据准备脚本 (prepare_data.py) 执行失败！"
    echo "请检查错误信息并确保Python环境和项目依赖正确。"
    exit 1
fi
echo "--------------------------------------------------------------"
echo "数据准备脚本执行完毕。"
echo ""
echo ""

# 步骤 3: 模型评估 (原步骤2)
echo ">>> 步骤 3: 执行已训练模型的评估脚本 (src.run_evaluation)..."
echo "--------------------------------------------------------------"
python -m src.run_evaluation
if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 模型评估脚本 (run_evaluation.py) 执行失败！"
    echo "请检查错误信息，并确认预训练模型文件已按要求放置且文件名正确。"
    exit 1
fi
echo "--------------------------------------------------------------"
echo "模型评估脚本执行完毕。"
echo ""
echo ""

echo "=============================================================="
echo "   所有评估步骤已顺利完成！"
echo "   - 性能指标CSV文件请查看: results/metrics/ 目录"
echo "   - 生成的评估图表请查看: results/plots/evaluation_summary_plots/ 目录"
echo "=============================================================="

exit 0