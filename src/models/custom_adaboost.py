# src/models/custom_adaboost.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels # 用于获取唯一标签



# ==================================================
# CustomBinaryAdaBoostClassifier
# ==================================================
class CustomBinaryAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    从零实现的二分类 AdaBoost (Adaptive Boosting) 分类器。

    参数:
    ----------
    base_estimator_prototype : estimator object, default=None
        基学习器原型。这是一个未训练的估计器对象。
        如果为 None，则基学习器默认为 sklearn.tree.DecisionTreeClassifier(max_depth=1)。
        这个基学习器必须在其 fit 方法中支持 sample_weight 参数。

    n_estimators : int, default=50
        要生成的基学习器的最大数量（即提升迭代次数）。
        如果提前达到完美拟合或学习器性能不再提升，可能会提前停止。

    learning_rate : float, default=1.0
        学习率，用以缩减每个分类器的贡献。learning_rate > 0。
        在 learning_rate 和 n_estimators 之间存在权衡。

    Attributes (拟合后生成):
    ----------
    estimators_ : list of estimator
        拟合好的基学习器集合。

    estimator_weights_ : np.ndarray of floats
        每个基学习器的权重 (alpha 值)。

    estimator_errors_ : np.ndarray of floats
        每个基学习器在训练时的（加权）错误率。
        
    classes_ : np.ndarray of shape (2,)
        训练时遇到的原始类别标签。

    _y_transform_map : dict
        用于将原始标签映射到 {-1, 1} 的内部映射。
        
    _y_transform_inv_map : dict
        用于将 {-1, 1} 映射回原始标签的内部映射。
    """

    def __init__(self, base_estimator_prototype=None, 
                 n_estimators: int = 50, 
                 learning_rate: float = 1.0,
                 random_state=None): # 添加 random_state 以便基学习器可以复现

        # 如果用户未指定，默认使用深度为1的决策树（决策树桩）
        if base_estimator_prototype is None:
            from sklearn.tree import DecisionTreeClassifier
            self.base_estimator_prototype = DecisionTreeClassifier(max_depth=1, random_state=random_state)
        else:
            self.base_estimator_prototype = base_estimator_prototype
            # 尝试为用户传入的基学习器设置 random_state (如果它有这个参数)
            if random_state is not None and hasattr(self.base_estimator_prototype, 'random_state'):
                try:
                    self.base_estimator_prototype.set_params(random_state=random_state)
                except Exception:
                    print(f"警告: 尝试为基学习器 {type(self.base_estimator_prototype).__name__} 设置 random_state 失败。")


        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError(f"n_estimators 必须是一个正整数, 得到 {n_estimators}")
        self.n_estimators = n_estimators

        if not (isinstance(learning_rate, float) or isinstance(learning_rate, int)) or learning_rate <= 0:
            raise ValueError(f"learning_rate 必须是一个正数, 得到 {learning_rate}")
        self.learning_rate = learning_rate
        
        # 初始化用于存储拟合结果的属性
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=float)
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=float)
        
        self.classes_ = None
        self._y_transform_map = {}
        self._y_transform_inv_map = {}


    def _setup_label_transform(self, y: np.ndarray):
        """内部函数，设置标签转换规则并转换y。"""
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError(f"输入的目标变量 y 必须是二分类的，但实际检测到 {len(self.classes_)} 个类别: {self.classes_}")

        self._y_transform_map = {self.classes_[0]: -1, self.classes_[1]: 1}
        self._y_transform_inv_map = {-1: self.classes_[0], 1: self.classes_[1]}
        
        return np.array([self._y_transform_map[label] for label in y])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        在数据集 (X, y) 上拟合 AdaBoost 分类器。
        """
        X, y = check_X_y(X, y, accept_sparse=True) # 允许稀疏矩阵
        y_transformed = self._setup_label_transform(y)
        n_samples = X.shape[0]

        # 1. 初始化样本权重
        sample_weights = np.full(n_samples, (1.0 / n_samples))

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=float)
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=float)
        
        actual_n_estimators = 0

        for iboost in range(self.n_estimators):
            actual_n_estimators += 1
            # 2. 使用当前样本权重训练一个基学习器
            estimator = clone(self.base_estimator_prototype)
            try:
                # 一些基学习器可能需要 y 是整数类型
                estimator.fit(X, y_transformed, sample_weight=sample_weights)
            except TypeError as e:
                if "sample_weight" in str(e).lower(): # 更通用的检查
                    raise TypeError(f"基学习器 {type(estimator).__name__} 必须在其 fit 方法中支持 sample_weight 参数。")
                else:
                    raise e # 重新抛出其他类型的 TypeError
            except Exception as e_fit:
                print(f"错误 (迭代 {iboost+1}): 基学习器拟合失败: {e_fit}")
                actual_n_estimators -= 1 # 这个估计器失败了
                break # 或者跳过这个估计器，但AdaBoost通常需要每个估计器都有效

            # 3. 计算基学习器的预测 (应为 {-1, 1})
            y_pred_m = estimator.predict(X)

            # 4. 计算加权错误率 epsilon_m
            incorrect_flags = (y_pred_m != y_transformed)
            epsilon_m = np.sum(sample_weights[incorrect_flags])
            self.estimator_errors_[iboost] = epsilon_m

            # 处理极端错误率
            if epsilon_m <= 1e-10: # 几乎完美的分类器 (使用一个小的容忍度)
                self.estimator_weights_[iboost] = self.learning_rate * 1.0 # 最大化权重（或一个较大的正数）
                self.estimators_.append(estimator)
                print(f"  信息 (迭代 {iboost+1}): 基学习器错误率 ({epsilon_m:.4e}) 极低，接近完美。可能提前停止。")
                break # 提前停止，因为已经找到了一个近乎完美的学习器

            if epsilon_m >= 0.5 - 1e-10: # 等于或差于随机猜测 (使用一个小的容忍度)
                print(f"  警告 (迭代 {iboost+1}): 基学习器错误率 ({epsilon_m:.4f}) >= 0.5。")
                # 如果这是第一个学习器，且总共只有一个学习器，那整个模型就失败了
                if iboost == 0 and self.n_estimators == 1:
                    print("    由于第一个且唯一的基学习器性能不佳，AdaBoost 无法学习。")
                    self.estimators_.append(estimator) # 仍然添加它，但权重会是0或负
                    self.estimator_weights_[iboost] = 0.0 # 赋予0权重
                    break
                # 对于后续学习器，或者有多个学习器时，如果错误率太高，则不再添加后续学习器
                print(f"    由于错误率过高，AdaBoost 提前停止在第 {iboost+1} 次迭代 (实际使用 {len(self.estimators_)} 个基学习器)。")
                actual_n_estimators -=1
                break 

            # 5. 计算基学习器权重 alpha_m
            # epsilon_m 保证在 (0, 0.5) 之间才能使 alpha_m > 0
            alpha_m = self.learning_rate * 0.5 * np.log((1.0 - epsilon_m) / epsilon_m)
            self.estimator_weights_[iboost] = alpha_m

            # 6. 更新样本权重 D
            update_factor = np.exp(-alpha_m * y_transformed * y_pred_m)
            sample_weights *= update_factor
            
            sample_weights_sum = np.sum(sample_weights)
            if sample_weights_sum <= 1e-10: # 权重几乎全为0
                print(f"  警告 (迭代 {iboost+1}): 样本权重之和接近于零，可能导致数值不稳定。提前停止。")
                actual_n_estimators -=1 # 这个迭代的权重更新可能有问题
                break
            sample_weights /= sample_weights_sum # 归一化

            self.estimators_.append(estimator)
        
        # 调整数组大小以匹配实际使用的估计器数量
        if actual_n_estimators < self.n_estimators:
            self.n_estimators = actual_n_estimators # 更新实际使用的数量
        
        self.estimator_weights_ = self.estimator_weights_[:self.n_estimators]
        self.estimator_errors_ = self.estimator_errors_[:self.n_estimators]

        if not self.estimators_:
            print("警告: AdaBoost 未能训练任何有效的基学习器。")
        else:
            print(f"AdaBoost 拟合完成。共使用 {len(self.estimators_)} 个基学习器。")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=True)

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        if not self.estimators_: # 如果没有训练出任何有效的估计器
            return scores # 返回全零分数

        for i in range(len(self.estimators_)):
            estimator = self.estimators_[i]
            alpha_m = self.estimator_weights_[i]
            
            y_pred_m = estimator.predict(X) # 假设返回 {-1, 1}
            scores += alpha_m * y_pred_m
            
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["estimators_", "classes_", "_y_transform_inv_map"])
        
        raw_scores = self.decision_function(X)
        
        # np.sign(0) is 0. Map 0 to one of the classes, e.g., the one mapped to -1.
        transformed_predictions = np.sign(raw_scores)
        transformed_predictions[transformed_predictions == 0] = -1 
        
        # 映射回原始标签
        y_pred = np.array([self._y_transform_inv_map[val] for val in transformed_predictions])
        
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        (粗略实现) 预测每个类别的概率。
        标准的Freund & Schapire AdaBoost不直接输出校准的概率。
        这个实现基于 decision_function 的 sigmoid 变换，仅为API兼容性提供。
        """
        check_is_fitted(self, ["estimators_", "classes_", "_y_transform_map"])
        print("警告: CustomBinaryAdaBoostClassifier 的 predict_proba 方法提供的是基于 decision_function 的粗略概率估计，并非严格校准的概率。")

        df = self.decision_function(X)
        
        # 使用 sigmoid 函数将 decision_function 值映射到 [0, 1] 区间
        # 这对应于映射到 +1 的那个类的概率
        prob_positive_class_transformed = 1.0 / (1.0 + np.exp(-df)) 
        prob_negative_class_transformed = 1.0 - prob_positive_class_transformed
        
        proba = np.zeros((X.shape[0], len(self.classes_)))
        
        # self._y_transform_map = {self.classes_[0]: -1, self.classes_[1]: 1}
        # 所以，prob_negative_class_transformed 对应 self.classes_[0]
        # prob_positive_class_transformed 对应 self.classes_[1]
        
        # 找到原始类别在self.classes_（已排序）中的索引
        # unique_labels 返回的是排序后的标签
        idx_class_neg = 0 # 对应 self.classes_[0]
        idx_class_pos = 1 # 对应 self.classes_[1]

        proba[:, idx_class_neg] = prob_negative_class_transformed
        proba[:, idx_class_pos] = prob_positive_class_transformed
        
        return proba



# ==================================================
# CustomOvRAdaboostClassifier (One-vs-Rest Wrapper)
# ==================================================
class CustomOvRAdaboostClassifier(BaseEstimator, ClassifierMixin):
    """
    从零实现的多分类 AdaBoost 分类器，采用 One-vs-Rest (OvR) 策略。
    它使用多个 CustomBinaryAdaBoostClassifier 实例。

    参数:
    ----------
    base_estimator_prototype_for_binary_ada : estimator object
        传递给每个内部 CustomBinaryAdaBoostClassifier 的基学习器原型。
        例如 DecisionTreeClassifier(max_depth=1)。

    n_estimators_per_classifier : int, default=50
        每个内部二分类 AdaBoost 分类器的基学习器数量。

    learning_rate_per_classifier : float, default=1.0
        每个内部二分类 AdaBoost 分类器的学习率。

    random_state_per_classifier : int, optional
        用于每个内部二分类 AdaBoost 分类器的随机状态，以确保可复现性，
        如果其基学习器也使用随机状态。

    Attributes (拟合后生成):
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        训练时遇到的所有唯一类别标签（已排序）。

    classifiers_ : list of CustomBinaryAdaBoostClassifier
        为每个类别训练的 CustomBinaryAdaBoostClassifier 实例列表。
        列表的顺序与 self.classes_ 中类别的顺序对应。
    """
    def __init__(self, base_estimator_prototype_for_binary_ada,
                 n_estimators_per_binary_classifier: int = 50,
                 learning_rate_per_binary_classifier: float = 1.0,
                 random_state_per_classifier = None): # random_state 会传递给二元分类器
        
        self.base_estimator_prototype_for_binary_ada = base_estimator_prototype_for_binary_ada
        self.n_estimators_per_binary_classifier = n_estimators_per_binary_classifier
        self.learning_rate_per_binary_classifier = learning_rate_per_binary_classifier
        self.random_state_per_classifier = random_state_per_classifier

        self.classes_ = None
        self.classifiers_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        在数据集 (X, y) 上拟合 One-vs-Rest AdaBoost 分类器。
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = unique_labels(y) # 获取所有唯一类别，并已排序
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("目标变量 y 必须至少包含两个类别才能进行分类。")
        if n_classes == 2:
            print("警告: 检测到只有2个类别，将直接使用单个 CustomBinaryAdaBoostClassifier。")
            # 直接训练一个二分类AdaBoost
            binary_ada = CustomBinaryAdaBoostClassifier(
                base_estimator_prototype=self.base_estimator_prototype_for_binary_ada,
                n_estimators=self.n_estimators_per_binary_classifier,
                learning_rate=self.learning_rate_per_binary_classifier,
                random_state=self.random_state_per_classifier 
            )
            binary_ada.fit(X, y) # y 将在内部转换为 {-1, 1}
            self.classifiers_ = [binary_ada] # 只有一个分类器
        else: # K > 2，执行 OvR
            self.classifiers_ = []
            print(f"开始为 {n_classes} 个类别执行 One-vs-Rest AdaBoost 训练...")
            for i, target_class in enumerate(self.classes_):
                print(f"  正在训练分类器 {i+1}/{n_classes} (针对类别 '{target_class}' vs Rest)...")
                # 1. 准备二元标签: target_class 为 +1, 其他为 -1
                y_binary = np.where(y == target_class, 1, -1)
                
                # 2. 创建并训练一个二分类 AdaBoost 模型
                #    每个二元分类器都应该是一个独立的实例
                binary_ada_classifier = CustomBinaryAdaBoostClassifier(
                    base_estimator_prototype=clone(self.base_estimator_prototype_for_binary_ada), # 克隆原型
                    n_estimators=self.n_estimators_per_binary_classifier,
                    learning_rate=self.learning_rate_per_binary_classifier,
                    random_state=self.random_state_per_classifier # 传递随机状态
                )
                # CustomBinaryAdaBoostClassifier 的 fit 方法内部会处理将 y_binary (已是-1,1)
                # 与其内部 classes_ (也会是 [-1, 1]) 的映射。
                binary_ada_classifier.fit(X, y_binary)
                self.classifiers_.append(binary_ada_classifier)
            print("所有 One-vs-Rest 分类器训练完成。")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算输入样本属于每个类别的置信度分数 (OvR策略)。
        返回一个形状为 (n_samples, n_classes) 的数组。
        """
        check_is_fitted(self, ["classes_", "classifiers_"])
        X = check_array(X, accept_sparse=True)

        if not self.classifiers_: # 如果没有分类器被训练
             return np.zeros((X.shape[0], len(self.classes_) if self.classes_ is not None else 1))

        # 如果是二分类情况 (n_classes_was_2_at_fit_time)
        if len(self.classes_) == 2 and len(self.classifiers_) == 1:
            # 对于二分类，decision_function 返回 (n_samples,)
            # 我们需要将其调整为 (n_samples, 1) 或 (n_samples, 2) 的形式以便与多分类统一
            # decision_function 的正值对应 self.classifiers_[0].classes_[1] (即原始的self.classes_[1])
            # 如果返回 (n_samples,), 其符号表示分类。
            # sklearn OneVsRestClassifier 对于二分类情况有特殊处理，我们这里也简化：
            # 直接返回二元分类器的 decision_function，然后 predict 时用 sign
            # 或者，为了统一接口，我们构造成 (n_samples, 2)
            # decision_scores_binary = self.classifiers_[0].decision_function(X)
            # scores_ovr = np.zeros((X.shape[0], 2))
            # scores_ovr[:, 1] = decision_scores_binary # 正类的分数
            # scores_ovr[:, 0] = -decision_scores_binary # 负类的分数 (一个简单的转换)
            # return scores_ovr
            # 简单起见，如果真的是二分类，预测逻辑会稍微不同。
            # 但 CustomOvRAdaboostClassifier 主要目标是 K > 2 的情况。
            # 为了统一，这里我们还是按 K 个分类器来处理，即使 K=1 (对于特殊情况)
            # 但上面 fit 中，如果 n_classes==2, self.classifiers_ 只有一个元素。
            # 所以这里也需要对应。
            decision_scores_binary = self.classifiers_[0].decision_function(X) # (n_samples,)
            # 为了和 predict 的 argmax 逻辑一致，我们返回 (n_samples, K)
            # 其中一列是 decision_scores_binary，另一列是其相反数（或0）
            # 这取决于如何解释OvR中二分类的decision_function
            # 更安全的做法是在predict中处理二分类的特殊情况
            # 这里我们假设，如果 K=2，那么 decision_function 返回的是针对 self.classes_[1] 的分数
            # 而针对 self.classes_[0] 的分数是其相反数 (或一个固定的方式)
            # scores_ovr = np.vstack((-decision_scores_binary, decision_scores_binary)).T # (n_samples, 2)
            # return scores_ovr
            # 为了更符合OvR的通用逻辑，即使是二分类，我们fit时也应该训练两个分类器，
            # 一个是0vs1，一个是1vs0。但那样就不是标准的OvR了。
            # sklearn的OneVsRestClassifier在fit时如果只有两个类，会直接训练一个普通分类器。
            # 我们的fit逻辑也是这样。所以这里的decision_function也应该对应。
            # 它返回一个 (n_samples,) 数组，predict可以直接用sign。
            # 但为了predict的argmax通用，这里应该返回(n_samples, 1)然后predict时特殊处理
            # 或者，这里的 decision_function 是用于多分类的，如果fit时是二分类，它可能不被直接调用
            # 或者，让 predict 处理这个特殊情况。
            # 鉴于 fit 中对 n_classes == 2 有特殊处理（只训练一个二元分类器），
            # predict 和 decision_function 也应反映这一点。
            # 最简单的是，如果 K=2, decision_function 返回二元分类器的 decision_function
            return self.classifiers_[0].decision_function(X) # shape (n_samples,)

        # K > 2 的情况
        all_decision_scores = np.array([clf.decision_function(X) for clf in self.classifiers_])
        return all_decision_scores.T # 转置为 (n_samples, n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["classes_", "classifiers_"])
        
        decision_scores = self.decision_function(X) # 获取置信度分数

        if not self.classifiers_: # 如果没有有效的分类器
            # 返回一个默认预测，例如第一个类别，或者抛出错误
            print("警告: OvR AdaBoost 中没有有效的分类器，无法预测。")
            # 简单的返回第一个类别作为预测 (或者可以做得更复杂)
            return np.full(X.shape[0], self.classes_[0] if self.classes_ is not None and len(self.classes_) > 0 else 0)

        # 如果在 fit 时检测到是二分类，则 decision_scores 的形状是 (n_samples,)
        if len(self.classes_) == 2:
            # decision_scores 是二元分类器针对 self.classes_[1] 的分数
            # 分数 > 0 预测为 self.classes_[1],否则为 self.classes_[0]
            binary_preds_transformed = np.where(decision_scores > 0, 1, -1)
            # 使用二元分类器内部的 _y_transform_inv_map (假设它保存了原始标签)
            # 或者直接用 self.classes_
            # y_pred = np.array([self.classifiers_[0]._y_transform_inv_map[val] for val in binary_preds_transformed])
            y_pred = np.where(binary_preds_transformed == 1, self.classes_[1], self.classes_[0])

        else: # K > 2 的情况，decision_scores 的形状是 (n_samples, n_classes)
            predicted_class_indices = np.argmax(decision_scores, axis=1)
            y_pred = self.classes_[predicted_class_indices]
            
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测每个类别的概率。
        在此OvR实现中，这是一个未精确实现的方法。
        """
        check_is_fitted(self, ["classes_", "classifiers_"])
        X = check_array(X, accept_sparse=True)
        
        # 根据用户之前的决定，直接抛出 NotImplementedError
        raise NotImplementedError("predict_proba 尚未在此多分类AdaBoost (CustomOvRAdaboostClassifier) 中精确实现。")

        # 如果未来要实现一个粗略版本：
        # decision_scores = self.decision_function(X) # shape (n_samples, n_classes) for K>2
        # if len(self.classes_) == 2: # 二分类特殊处理
        #     # 使用 CustomBinaryAdaBoostClassifier 的 predict_proba
        #     return self.classifiers_[0].predict_proba(X)
        #
        # # 对于 K > 2, 可以尝试对 decision_scores 进行 softmax 归一化
        # if decision_scores.ndim == 1: # 以防万一 decision_function 返回一维 (不太可能对于K>2)
        #     print("警告: decision_function 返回一维数组，无法为多分类计算概率。")
        #     # 返回均匀概率或报错
        #     return np.full((X.shape[0], len(self.classes_)), 1.0 / len(self.classes_))
        #
        # exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True)) # 减去max防止溢出
        # proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # return proba






# ==================================================
# CustomOvRAdaboostClassifier 的简单测试代码
# ==================================================
if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification # 用于生成多分类数据
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("\n" + "="*70)
    print(">>> 开始测试 CustomOvRAdaboostClassifier <<<")
    print("="*70)

    # 1. 准备一个多分类数据集 (例如3个类别)
    X_multi, y_multi = make_classification(
        n_samples=300, n_features=10, 
        n_informative=5, n_redundant=0, 
        random_state=42, n_classes=3, n_clusters_per_class=1 # 每个类别一个簇，使其相对可分
    )
    # 可以将标签转换为字符串测试
    # unique_original_labels = np.unique(y_multi)
    # label_map = {orig: f"Class_{orig}" for orig in unique_original_labels}
    # y_multi_str = np.array([label_map[y] for y in y_multi])


    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.33, random_state=123
    )
    print(f"\n多分类测试数据形状: X_train: {X_train_m.shape}, y_train: {y_train_m.shape}, "
          f"X_test: {X_test_m.shape}, y_test: {y_test_m.shape}")
    print(f"训练集标签类别: {np.unique(y_train_m)}, 测试集标签类别: {np.unique(y_test_m)}")


    # --- 测试场景1: OvR + 决策树桩 ---
    print("\n--- 测试场景 1: OvR AdaBoost (基学习器: 决策树桩) ---")
    stump_proto_for_ovr = DecisionTreeClassifier(max_depth=1, random_state=42)
    ovr_adaboost_stump = CustomOvRAdaboostClassifier(
        base_estimator_prototype_for_binary_ada=stump_proto_for_ovr,
        n_estimators_per_binary_classifier=15, # 每个二元分类器用15个树桩
        learning_rate_per_binary_classifier=0.8,
        random_state_per_classifier=42 # 确保内部二元AdaBoost可复现
    )
    print("开始拟合 OvR AdaBoost (决策树桩)...")
    ovr_adaboost_stump.fit(X_train_m, y_train_m)
    
    y_pred_ovr_stump_train = ovr_adaboost_stump.predict(X_train_m)
    y_pred_ovr_stump_test = ovr_adaboost_stump.predict(X_test_m)

    print(f"  OvR (决策树桩) - 训练集准确率: {accuracy_score(y_train_m, y_pred_ovr_stump_train):.4f}")
    print(f"  OvR (决策树桩) - 测试集准确率: {accuracy_score(y_test_m, y_pred_ovr_stump_test):.4f}")
    
    decision_scores_ovr_stump = ovr_adaboost_stump.decision_function(X_test_m)
    print(f"  OvR (决策树桩) - 测试集 decision_function (前3个样本, 每个样本3个类别的分数):\n{decision_scores_ovr_stump[:3]}")

    print("  尝试调用 predict_proba (预期 NotImplementedError)...")
    try:
        proba_ovr_stump = ovr_adaboost_stump.predict_proba(X_test_m)
        print(f"    predict_proba 返回 (这不应该发生): {proba_ovr_stump[:3]}")
    except NotImplementedError as e:
        print(f"    成功捕获预期错误: {e}")
    except Exception as e_proba:
        print(f"    调用 predict_proba 时发生意外错误: {e_proba}")


    # --- 测试场景2: OvR + 弱线性SVM ---
    print("\n--- 测试场景 2: OvR AdaBoost (基学习器: 弱线性SVM C=0.05) ---")
    svm_proto_for_ovr = SVC(kernel='linear', C=0.05, probability=False, random_state=42, max_iter=500) # probability=False 加速
    ovr_adaboost_svm = CustomOvRAdaboostClassifier(
        base_estimator_prototype_for_binary_ada=svm_proto_for_ovr,
        n_estimators_per_binary_classifier=5, # SVM较慢，迭代次数减少
        learning_rate_per_binary_classifier=0.5,
        random_state_per_classifier=42
    )
    print("开始拟合 OvR AdaBoost (线性SVM)...")
    ovr_adaboost_svm.fit(X_train_m, y_train_m)

    y_pred_ovr_svm_train = ovr_adaboost_svm.predict(X_train_m)
    y_pred_ovr_svm_test = ovr_adaboost_svm.predict(X_test_m)

    print(f"  OvR (线性SVM) - 训练集准确率: {accuracy_score(y_train_m, y_pred_ovr_svm_train):.4f}")
    print(f"  OvR (线性SVM) - 测试集准确率: {accuracy_score(y_test_m, y_pred_ovr_svm_test):.4f}")

    decision_scores_ovr_svm = ovr_adaboost_svm.decision_function(X_test_m)
    print(f"  OvR (线性SVM) - 测试集 decision_function (前3个样本, 每个样本3个类别的分数):\n{decision_scores_ovr_svm[:3]}")


    # --- 测试场景3: OvR 处理本身就是二分类的数据 ---
    print("\n--- 测试场景 3: OvR AdaBoost 处理二分类数据 (应直接使用一个二元分类器) ---")
    X_binary_again, y_binary_again = make_classification(n_samples=150, n_features=5, random_state=1, n_classes=2)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_binary_again, y_binary_again, test_size=0.3, random_state=1)
    
    ovr_adaboost_binary_case = CustomOvRAdaboostClassifier(
        base_estimator_prototype_for_binary_ada=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators_per_binary_classifier=10
    )
    print("开始拟合 OvR AdaBoost (二分类数据)...")
    ovr_adaboost_binary_case.fit(X_train_b, y_train_b) # fit 方法内部应检测到是二分类

    y_pred_ovr_binary_test = ovr_adaboost_binary_case.predict(X_test_b)
    print(f"  OvR (二分类数据) - 测试集准确率: {accuracy_score(y_test_b, y_pred_ovr_binary_test):.4f}")
    
    # 检查 decision_function 的输出形状，对于二分类情况，它应该直接返回二元分类器的decision_function结果
    decision_scores_ovr_binary = ovr_adaboost_binary_case.decision_function(X_test_b)
    print(f"  OvR (二分类数据) - 测试集 decision_function (前3个): {decision_scores_ovr_binary[:3]}")
    if decision_scores_ovr_binary.ndim == 1 and decision_scores_ovr_binary.shape[0] == X_test_b.shape[0]:
        print("    decision_function 形状对于二分类情况正确 (应为一维数组)。")
    else:
        print(f"    decision_function 形状对于二分类情况不正确: {decision_scores_ovr_binary.shape}")


    print("\n>>> CustomOvRAdaboostClassifier 所有测试场景结束 <<<")
    print("="*70)




# # ===============================================
# # CustomBinaryAdaBoostClassifier 的简单测试代码
# # ===============================================
# if __name__ == '__main__':
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.svm import SVC
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score

#     print("\n" + "="*70)
#     print(">>> 开始测试 CustomBinaryAdaBoostClassifier <<<")
#     print("="*70)

#     # 1. 准备一个简单的二分类数据集
#     X_sample, y_sample = make_classification(n_samples=300, n_features=10, 
#                                              n_informative=5, n_redundant=0, 
#                                              random_state=42, n_classes=2)
#     # 确保标签是 0 和 1 (或测试其他两个不同的值)
#     # y_sample_str = np.array(['类别A' if x == 0 else '类别B' for x in y_sample]) # 测试字符串标签

#     X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
#         X_sample, y_sample, test_size=0.33, random_state=123
#     )
#     print(f"\n测试数据形状: X_train: {X_train_s.shape}, y_train: {y_train_s.shape}, "
#           f"X_test: {X_test_s.shape}, y_test: {y_test_s.shape}")
#     print(f"训练集标签类别: {np.unique(y_train_s)}, 测试集标签类别: {np.unique(y_test_s)}")

#     # --- 测试场景1: 基学习器为决策树桩 ---
#     print("\n--- 测试场景 1: 基学习器为决策树桩 (max_depth=1) ---")
#     stump_prototype = DecisionTreeClassifier(max_depth=1, random_state=42)
#     adaboost_stump = CustomBinaryAdaBoostClassifier(
#         base_estimator_prototype=stump_prototype,
#         n_estimators=20, # 增加迭代次数
#         learning_rate=0.8 
#     )
#     print("开始拟合 AdaBoost (决策树桩)...")
#     adaboost_stump.fit(X_train_s, y_train_s)
    
#     y_pred_stump_train = adaboost_stump.predict(X_train_s)
#     y_pred_stump_test = adaboost_stump.predict(X_test_s)

#     print(f"  决策树桩 - 训练集准确率: {accuracy_score(y_train_s, y_pred_stump_train):.4f}")
#     print(f"  决策树桩 - 测试集准确率: {accuracy_score(y_test_s, y_pred_stump_test):.4f}")
    
#     decision_scores_stump = adaboost_stump.decision_function(X_test_s)
#     print(f"  决策树桩 - 测试集 decision_function (前3个样本): {decision_scores_stump[:3]}")
    
#     try:
#         proba_stump = adaboost_stump.predict_proba(X_test_s)
#         print(f"  决策树桩 - 测试集 predict_proba (前3个样本):\n{proba_stump[:3]}")
#         if proba_stump.shape == (X_test_s.shape[0], 2) and np.allclose(np.sum(proba_stump, axis=1), 1.0):
#             print("    predict_proba 形状和总和校验通过。")
#         else:
#             print(f"    predict_proba 形状 ({proba_stump.shape}) 或总和校验失败。")
#     except Exception as e:
#         print(f"  决策树桩 - predict_proba 发生错误: {e}")


#     # --- 测试场景2: 基学习器为线性SVM ---
#     print("\n--- 测试场景 2: 基学习器为线性SVM (C=0.1) ---")
#     # 注意: SVC 的 probability=True 会显著增加训练时间。
#     # 如果只是用 predict，可以考虑将其关闭以加速测试。但为了 predict_proba 的粗略实现，暂时保留。
#     svm_prototype = SVC(kernel='linear', C=0.1, probability=False, random_state=42, max_iter=1000) # probability=False 加速
#     adaboost_svm = CustomBinaryAdaBoostClassifier(
#         base_estimator_prototype=svm_prototype,
#         n_estimators=7, # SVM 较慢，用较少的迭代次数测试
#         learning_rate=0.5
#     )
#     print("开始拟合 AdaBoost (线性SVM)...")
#     adaboost_svm.fit(X_train_s, y_train_s)
    
#     y_pred_svm_train = adaboost_svm.predict(X_train_s)
#     y_pred_svm_test = adaboost_svm.predict(X_test_s)

#     print(f"  线性SVM - 训练集准确率: {accuracy_score(y_train_s, y_pred_svm_train):.4f}")
#     print(f"  线性SVM - 测试集准确率: {accuracy_score(y_test_s, y_pred_svm_test):.4f}")

#     decision_scores_svm = adaboost_svm.decision_function(X_test_s)
#     print(f"  线性SVM - 测试集 decision_function (前3个样本): {decision_scores_svm[:3]}")
    
#     try:
#         # SVC(probability=False) 时, sklearn 的 AdaBoostClassifier 会基于 decision_function 推断概率
#         # 我们的 predict_proba 也是基于 decision_function，所以这里应该能工作
#         proba_svm = adaboost_svm.predict_proba(X_test_s)
#         print(f"  线性SVM - 测试集 predict_proba (前3个样本):\n{proba_svm[:3]}")
#         if proba_svm.shape == (X_test_s.shape[0], 2) and np.allclose(np.sum(proba_svm, axis=1), 1.0):
#             print("    predict_proba 形状和总和校验通过。")
#         else:
#             print(f"    predict_proba 形状 ({proba_svm.shape}) 或总和校验失败。")
#     except Exception as e:
#         print(f"  线性SVM - predict_proba 发生错误: {e}")


#     # --- 测试场景3: 字符串标签 ---
#     print("\n--- 测试场景 3: 使用字符串标签 ('ClassA', 'ClassB') ---")
#     y_sample_str = np.array(['类别A' if x == 0 else '类别B' for x in y_sample])
#     X_train_str, X_test_str, y_train_str, y_test_str = train_test_split(
#         X_sample, y_sample_str, test_size=0.33, random_state=123
#     )
#     print(f"训练集字符串标签类别: {np.unique(y_train_str)}")

#     adaboost_stump_str_labels = CustomBinaryAdaBoostClassifier(
#         base_estimator_prototype=DecisionTreeClassifier(max_depth=1, random_state=42),
#         n_estimators=10,
#         learning_rate=0.8
#     )
#     print("开始拟合 AdaBoost (决策树桩, 字符串标签)...")
#     adaboost_stump_str_labels.fit(X_train_str, y_train_str)
#     y_pred_str_test = adaboost_stump_str_labels.predict(X_test_str)
#     print(f"  决策树桩 (字符串标签) - 测试集准确率: {accuracy_score(y_test_str, y_pred_str_test):.4f}")
#     print(f"  预测的标签示例 (字符串): {y_pred_str_test[:5]}")
#     if np.all(np.isin(y_pred_str_test, ['类别A', '类别B'])):
#         print("    预测的标签是原始字符串标签，正确！")
#     else:
#         print("    错误：预测的标签不是原始字符串标签！")


#     print("\n>>> CustomBinaryAdaBoostClassifier 所有测试场景结束 <<<")
#     print("="*70)