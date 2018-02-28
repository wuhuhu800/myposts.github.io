---
layout: post
title: XGBoost 参数官方文档翻译
categories: 机器学习
description: 关于xgboost的官方文档理解翻译
keywords: 机器学习,Xgboost,
tags: jekyll
---

xgboost参数非常之多，打算借着翻译[*官方文档*](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)理解一下xgboost的相关参数，以下是xgboost官方文档关于参数的全部翻译。

XGBoost Parameters
==================
Before running XGboost, we must set three types of parameters: general parameters, booster parameters and task parameters.
- General parameters relates to which booster we are using to do boosting, commonly tree or linear model
- Booster parameters depends on which booster you have chosen
- Learning Task parameters that decides on the learning scenario, for example, regression tasks may use different parameters with ranking tasks.
- Command line parameters that relates to behavior of CLI version of xgboost.

Parameters in R Package
-----------------------
In R-package, you can use .(dot) to replace underscore in the parameters, for example, you can use max.depth as max_depth. The underscore parameters are also valid in R.

General Parameters
------------------
> general Parameters 主要是用于选择booster ，一般是tree或者线性模型

* booster [default=gbtree]
  - which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.

>- booster 默认参数gbtree
>- gbtree 和 dart 用于tree 模型
>- gblinear 用于linear 模型

* silent [default=0]
  - 0 means printing running messages, 1 means silent mode.

>- silent 默认参数0
>- 0 代表 打印运行时过程的信息
>- 1 代表 不打印信息

* nthread [default to maximum number of threads available if not set]
  - number of parallel threads used to run xgboost

>- nthread 默认最大线程数运行xgboost
>- n 为需要运行的线程数

* num_pbuffer [set automatically by xgboost, no need to be set by user]
  - size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step.

>- **num_pbuffer** xgboost 自动设置，无需用户自己设置（高亮的都是自己暂时不理解的）
>- num_pbuffer 是关于预测的缓存的大小，一般设置的是训练实例数。这个缓存用于
保存bootsting 最后一般的预测结果

* num_feature [set automatically by xgboost, no need to be set by user]
  - feature dimension used in boosting, set to maximum dimension of the feature

>- **num_feature** xgboost 自动设置，无需用户自己设置（高亮的都是自己暂时不理解的）
>- 在boosting中使用feature的维度，会设置成feature最大的维度

Parameters for Tree Booster
---------------------------
>- Tree booster 的参数
>- tree booster，它的表现远远胜过linear booster

* eta [default=0.3, alias: learning_rate]
  - step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.
  - range: [0,1]

>- eta 默认是0.3，和sklearn learning_rate概念一样。参数值的范围是0~1之间
>- 为了防止过拟合，更新过程中会进行收缩步长。
>- 在每次提升计算之后，算法会直接获得新特征的权重。
>- eta通过缩减特征的权重使得 提升过程(boosting process) 更加收敛(conservative)也就是说：可以提高模型的鲁棒性（robust）
>- 典型值为0.01-0.2
>- 每次更新树的时候，更新的部分的/之前树的部分

* gamma [default=0, alias: min_split_loss]
  - minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.
  - range: [0,∞]

>- gamma 默认是0，和min_split_loss概念一样？？参数值的范围是0~∞之间
>- 最小化loss reduction 需要进一步对于树的叶子节点进行分割。树越大，算法越收敛（conservative）
>- 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
>- 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的


* max_depth [default=6]
  - maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting. 0 indicates no limit, limit is required for depth-wise grow policy.
  - range: [0,∞]

>- max_depth 默认是6，参数值的范围是0~∞之间
>- 设置树的最大深度，值越大，越复杂，越容易过拟合。
>- 0 代表不做限制，**limit is required for depth-wise grow policy？？**
>- 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。
>- 需要使用CV函数来进行调优。
>- 典型值：3-10


* min_child_weight [default=1]
  - minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
  - range: [0,∞]

>- min_child_weight 默认是1，参数值的范围是0~∞之间
>- **孩子节点中最小的样本权重和**。
>- 当在**tree拆分**过程中，出现一个叶子节点的样本权重和小于min_child_weight时，则拆分结束。在线性回归模型中，这个参数在每个**节点？？指的是什么**仅和所需要的最小样本数。该成熟越大算法越conservative
>- 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。
>- 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。

* max_delta_step [default=0]
  - Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
  - range: [0,∞]

>- max_delta_step 默认是0，参数值的范围是0~∞之间
>- Maximum delta step就是我们对每棵树的权重的评估。
>- 如果值设置为0，代表不做限制
>- 如果值设置为正数，可以使 update step更加收敛。一般来说，这个参数不需要，但是对于逻辑回归出现的类型极端不平衡时候,会有作用。
>- 参数设置为1-10之间控制update
>- 这个参数一般用不到，但是你可以挖掘出来它更多的用处。

* subsample [default=1]
  - subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
  - range: (0,1]

>- subsample 默认是1，参数值的范围是0~1之间，不包括0
>- 根据比例，设置是训练样本的子样本数据。如果设置为0.5，意味着，Xgboost随机选择训练样本的一半数据进行grow tree,这样做目的是防止过拟合。
>- 和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。
>- 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
>- 典型值：0.5-1

* colsample_bytree [default=1]
  - subsample ratio of columns when constructing each tree.
  - range: (0,1]

>- colsample_bytree 默认是1，参数值的范围是0~1之间，不包括0
>- 根据比例，设置是训练样本的列的比例（个人理解，算是特征比例），进行构建树
>- 典型值：0.5-1

* colsample_bylevel [default=1]
  - subsample ratio of columns for each split, in each level.
  - range: (0,1]

>- colsample_bylevel 默认是1，参数值的范围是0~1之间，不包括0
>- 根据比例，设置是训练样本**每个层级**的列的比例
>- 我个人一般不太用这个参数，因为subsample参数和colsample_bytree参数可以起到相同的作用。但是如果感兴趣，可以挖掘这个参数更多的用处。

* lambda [default=1, alias: reg_lambda]
  - L2 regularization term on weights, increase this value will make model more conservative.

>- lambda 默认是1，和**reg_lambda**概念一样(和Ridge regression类似)
>- **L2 正则术语（L2惩罚系数？？）**，增加该值会让模型更加收敛
>- 这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。

* alpha [default=0, alias: reg_alpha]
  - L1 regularization term on weights, increase this value will make model more conservative.

>- alpha 默认是0，和**reg_alpha**概念一样(和Lasso regression类似)
>- **L1 正则术语（L1惩罚系数？？）**，增加该值会让模型更加收敛
>- 可以应用在很高维度的情况下，使得算法的速度更快。

* tree_method, string [default='auto']
  - The tree construction algorithm used in XGBoost(see description in the [reference paper](http://arxiv.org/abs/1603.02754))
  - Distributed and external memory version only support approximate algorithm.
  - Choices: {'auto', 'exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist'}
    - 'auto': Use heuristic to choose faster one.
      - For small to medium dataset, exact greedy will be used.
      - For very large-dataset, approximate algorithm will be chosen.
      - Because old behavior is always use exact greedy in single machine,
        user will get a message when approximate algorithm is chosen to notify this choice.
    - 'exact': Exact greedy algorithm.
    - 'approx': Approximate greedy algorithm using sketching and histogram.
    - 'hist': Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
	- 'gpu_exact': GPU implementation of exact algorithm.
	- 'gpu_hist': GPU implementation of hist algorithm.

>- tree_method, 参数是string类型，默认’auto‘，
>- 主要用于构建树的算法（[参考论文](http://arxiv.org/abs/1603.02754))
>- 参数可以选择{'auto', 'exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist'}
>- ’auto‘：用于启发式，选择更快的方式
   - 数据量小型和中型的，会自动使用’exact‘模式
   - 数据量特别大的话，会自动使用’approx‘模型
   - 由于之前一直在单元的机器学习里都是使用’exact‘模式，所以如果自动选择’approx‘模式会有提示
>- ’exact‘：纯贪婪算法
>- ’approx‘：近似贪婪算法，使用**sketching and histogram**方法
>- ’hist'快速histogram 优化后的近似贪婪算法。用在性能的提高上，例如**“bins caching”**
>- 'gpu_exact'： 用GPU实施exact 算法
>- 'gpu_hist'：用GPU实施hist 算法

* sketch_eps, [default=0.03]
  - This is only used for approximate greedy algorithm.
  - This roughly translated into ```O(1 / sketch_eps)``` number of bins.
    Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy.
  - Usually user does not have to tune this.
    but consider setting to a lower number for more accurate enumeration.
  - range: (0, 1)

  >- **sketch_eps**：默认0.03，范围是0~1，不包括0和1
  >- 本参数只能用在近似贪婪算法上
  >- 比较粗的转成0和1
  >- 一般，用户不需要关注这个参数。
  >- 总体而言，参数值越低，越能得到准确的值

* scale_pos_weight, [default=1]
  - Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative  cases) / sum(positive cases) See [Parameters Tuning](how_to/param_tuning.md) for more discussion. Also see Higgs Kaggle competition demo for examples: [R](../demo/kaggle-higgs/higgs-train.R ), [py1](../demo/kaggle-higgs/higgs-numpy.py ), [py2](../demo/kaggle-higgs/higgs-cv.py ), [py3](../demo/guide-python/cross_validation.py)

  >- scale_pos_weight：默认1
  >- 本参数主要控制正数和负数的权重，用在对于正负不平衡的类型的数据上。一般值为
  负例样本总数/正例样本总数
  >- 可以参考[Parameters Tuning](http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html)有更多讨论
  >- 也可以参考Kaggle比赛demo样例
  [py1](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py)
  [py2](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-cv.py), [py3](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py)
  >- 总体而言，参数值越低，越能得到准确的值

* updater, [default='grow_colmaker,prune']
  - A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitly by a user. The following updater plugins exist:
    - 'grow_colmaker': non-distributed column-based construction of trees.
    - 'distcol': distributed tree construction with column-based data splitting mode.
    - 'grow_histmaker': distributed tree construction with row-based data splitting based on global proposal of histogram counting.
    - 'grow_local_histmaker': based on local histogram counting.
    - 'grow_skmaker': uses the approximate sketching algorithm.
    - 'sync': synchronizes trees in all distributed nodes.
    - 'refresh': refreshes tree's statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
    - 'prune': prunes the splits where loss < min_split_loss (or gamma).
  - In a distributed setting, the implicit updater sequence value would be adjusted as follows:
    - 'grow_histmaker,prune' when  dsplit='row' (or default) and prob_buffer_row == 1 (or default); or when data has multiple sparse pages
    - 'grow_histmaker,refresh,prune' when  dsplit='row' and prob_buffer_row < 1
    - 'distcol' when dsplit='col'

>- **updater**：默认'grow_colmaker,prune',通过构造tree 列表可以tree之间可用逗号(,)进行分开,这种tree 列表方式可以构建和修改trees
>- 这种高级参数，依赖其他的参数，一般设置为自动化。
>- 当然用户也可以明确指明所使用的updater 插件
>- 'grow_colmaker':基于列进行构建 非分布式tree
>- 'distcol':基于列的数据的分割构建分布式tree
>- **'grow_histmaker'**：基于行的数据分割构建分布式tree，行数据的分割是根据gloabl histogram 统计后的建议
>- 'grow_local_histmaker'：基于local histogram统计后的建议构建树。
>- 'grow_skmaker':使用近似sketching算法构建树
>- 'sync'：在所有分布式节点同步树
>- 'refresh':基于目的数据，刷新树的统计和叶子的值。注意非随机的子样本数据的行将被执行
>- 'prune':去掉loss 小于min_split_loss (or gamma)的分支(splits)
>- 在分布式设置里，updater的内在序列值可以这样调整：
>  - 当数据具有非常多的稀疏页或者dsplit='row' (or default) 且 prob_buffer_row == 1 (or default)的时候，参数可以是'grow_histmaker,prune'
>  - 当dsplit='row' 且 prob_buffer_row < 1时，参数可以是'grow_histmaker,refresh,prune'
  - 当dsplit='col'，参数可以为'distcol'


* refresh_leaf, [default=1]
  - This is a parameter of the 'refresh' updater plugin. When this flag is true, tree leafs as well as tree nodes' stats are updated. When it is false, only node stats are updated.

>- refresh_leaf：默认值为1
>- 这是updater参数为'refresh'的一个插件
>- 当值为1，树的叶子和节点值都更新
>- 当值为0，只更新节点

* process_type, [default='default']
  - A type of boosting process to run.
  - Choices: {'default', 'update'}
    - 'default': the normal boosting process which creates new trees.
    - 'update': starts from an existing model and only updates its trees. In each boosting iteration, a tree from the initial model is taken, a specified sequence of updater plugins is run for that tree, and a modified tree is added to the new model. The new model would have either the same or smaller number of trees, depending on the number of boosting iteratons performed. Currently, the following built-in updater plugins could be meaningfully used with this process type: 'refresh', 'prune'. With 'update', one cannot use updater plugins that create new nrees.

>- process_type：默认值为'default'
>- 参数是提高**process** 的一种方式
>- 参数有两个可选值{'default', 'update'}
>- 'default'：标准提高process 方式，通过创建新的树
>- 'update'：从一个已有的模型里开始，只更新该模型的树。在每个提高的运算中，最开始的模型的树，被updater 插件的序列值（参考updater参数说明）运行最初的tree，这样经过修改过的tree，放入新的模型中。新的模型拥有tree的数目和之前比有可能相同或者比之前要小，取决于boosting iteratons的有多少数量被执行。目前，updater的参数是'refresh', 'prune'使用'update'参数非常有用。updater 插件不产生新trees

* grow_policy, string [default='depthwise']
  - Controls a way new nodes are added to the tree.
  - Currently supported only if `tree_method` is set to 'hist'.
  - Choices: {'depthwise', 'lossguide'}
    - 'depthwise': split at nodes closest to the root.
    - 'lossguide': split at nodes with highest loss change.

>- grow_policy：默认值为'depthwise'
>- 该参数是一种在树上添加新节点的方式
>- 只有`tree_method` 的参数是'hist'的时候才有用
>- 两个参数值：'depthwise', 'lossguide'
    - 'depthwise': 拆分出距离根节点最近的节点.
    - 'lossguide': 拆分出loss 变化最大的节点.

* max_leaves, [default=0]
  - Maximum number of nodes to be added. Only relevant for the 'lossguide' grow policy.

>- max_leaves：默认值为 0
>- 添加最大的节点数，只有grow policy设置为'lossguide'才有用

* max_bin, [default=256]
  - This is only used if 'hist' is specified as `tree_method`.
  - Maximum number of discrete bins to bucket continuous features.
  - Increasing this number improves the optimality of splits at the cost of higher computation time.

>- **max_bin**：默认值为 256
>- 只有’tree_method‘的参数是'hist' 才有用
>- 该参数是对于连续特征的最大数量的分割箱
>- 增加数量可以提高splits的优化性，代价是需要更多的运算时间

* predictor, [default='cpu_predictor']
  - The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
    - 'cpu_predictor': Multicore CPU prediction algorithm.
    - 'gpu_predictor': Prediction using GPU. Default for 'gpu_exact' and 'gpu_hist' tree method.

>- predictor：默认值为 'cpu_predictor'
>- 选择GPU还是CPU计算的预测器，当然最终结果是一样的
>- 'cpu_predictor':多核CPU预测算法
>- 'gpu_predictor':用GPU预测,默认'gpu_exact' 和 'gpu_hist' 的tree method

Additional parameters for Dart Booster
--------------------------------------
>- Dart Booster 额外的一些参数

* sample_type [default="uniform"]
  - type of sampling algorithm.
    - "uniform": dropped trees are selected uniformly.
    - "weighted": dropped trees are selected in proportion to weight.

>- sample_type：默认值为 "uniform"
>- 参数是一种取sampling 的算法
>  - "uniform": 均匀的去掉trees.
  - "weighted": 按照一定的权重去掉trees.

* normalize_type [default="tree"]
  - type of normalization algorithm.
    - "tree": new trees have the same weight of each of dropped trees.
      - weight of new trees are 1 / (k + learning_rate)
      - dropped trees are scaled by a factor of k / (k + learning_rate)
    - "forest": new trees have the same weight of sum of dropped trees (forest).
      - weight of new trees are 1 / (1 + learning_rate)
      - dropped trees are scaled by a factor of 1 / (1 + learning_rate)

>- normalize_type：默认值为 "tree"
>- 参数是一种标准化的算法，个人觉得k应该是指dropped掉的trees的数量
>  - "tree":新trees和dropped掉的每颗tree有相同的权重
  - 新trees的权重是1 / (k + learning_rate)
  - dropped掉的 trees 按照  k / (k + learning_rate)缩放
- "forest": 新trees和drop掉的数目的总和有相同的权重
  - 新trees 的权重1 / (1 + learning_rate)
  - dropped掉的 trees 按照  1 / (1 + learning_rate)的权重缩放

* rate_drop [default=0.0]
  - dropout rate (a fraction of previous trees to drop during the dropout).
  - range: [0.0, 1.0]

>- rate_drop：默认值为 0，值的范围[0.0~1.0]
>- dropout的比例（就是对于之前的tree，drop掉的比例）


* one_drop [default=0]
  - when this flag is enabled, at least one tree is always dropped during the dropout (allows Binomial-plus-one or epsilon-dropout from the original DART paper).


  >- **one_drop**：默认值为 0
  >- 如果值为1，至少有一个tree，在dropout期间总会被去掉的

* skip_drop [default=0.0]
  - Probability of skipping the dropout procedure during a boosting iteration.
    - If a dropout is skipped, new trees are added in the same manner as gbtree.
    - Note that non-zero skip_drop has higher priority than rate_drop or one_drop.
  - range: [0.0, 1.0]

>- skip_drop：默认值为 0.0，参数值范围[0.0~1.0]
>- 在提升迭代期间，跳过丢弃过程的概率
>- 如果一个dropout 被跳过，新trees 将会和gbtree一样的形式增加上
>- 注意，非0 skip_drop 比 rate_drop or one_drop有更高有限权

Parameters for Linear Booster
-----------------------------
>- 线性bootser

* lambda [default=0, alias: reg_lambda]
  - L2 regularization term on weights, increase this value will make model more conservative.

>- lambda 默认是0，和**reg_lambda**概念一样
>- **L2 正则术语（L2惩罚系数？？）**，增加该值会让模型更加收敛

* alpha [default=0, alias: reg_alpha]
  - L1 regularization term on weights, increase this value will make model more conservative.

>- alpha 默认是0，和**reg_alpha**概念一样
>- **L1 正则术语（L1惩罚系数？？）**，增加该值会让模型更加收敛

* lambda_bias [default=0, alias: reg_lambda_bias]
  - L2 regularization term on bias (no L1 reg on bias because it is not important)

>- lambda_bias 默认是0，和**reg_lambda_bias**概念一样
>- **在偏置上的L2 正则**，在L1上没有偏置项的正则，因为L1时偏置不重要


Parameters for Tweedie Regression
---------------------------------
>- [Tweedie 分布](https://en.wikipedia.org/wiki/Tweedie_distribution)

* tweedie_variance_power [default=1.5]
  - parameter that controls the variance of the Tweedie distribution
    - var(y) ~ E(y)^tweedie_variance_power
  - range: (1,2)
  - set closer to 2 to shift towards a gamma distribution
  - set closer to 1 to shift towards a Poisson distribution.

>- tweedie_variance_power 默认值为1.5，范围(1~2)
>- 参数变量控制Tweedie 分布
   - var(y) ~ E(y)^tweedie_variance_power
>- 参数值离2越靠近，该分布越接近gamma分布
>- 参数值离1越靠近，该分布越接近泊松分布

Learning Task Parameters
------------------------
>- 学习任务参数

Specify the learning task and the corresponding learning objective. The objective options are below:
>- 指定学习任务和学习目标。学习目标如下：

* objective [default=reg:linear]
  - "reg:linear" --linear regression
  - "reg:logistic" --logistic regression
  - "binary:logistic" --logistic regression for binary classification, output probability
  - "binary:logitraw" --logistic regression for binary classification, output score before logistic transformation
  - "gpu:reg:linear", "gpu:reg:logistic", "gpu:binary:logistic", gpu:binary:logitraw" --versions
    of the corresponding objective functions evaluated on the GPU; note that like the GPU histogram algorithm,
    they can only be used when the entire training session uses the same dataset
  - "count:poisson" --poisson regression for count data, output mean of poisson distribution
    - max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
  - "multi:softmax" --set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
  - "multi:softprob" --same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
  - "rank:pairwise" --set XGBoost to do ranking task by minimizing the pairwise loss
  - "reg:gamma" --gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be [gamma-distributed](https://en.wikipedia.org/wiki/Gamma_distribution#Applications)
  - "reg:tweedie" --Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be [Tweedie-distributed](https://en.wikipedia.org/wiki/Tweedie_distribution#Applications).

>- objective,默认值是reg:linear
>- "reg:linear" ：线性回归
>- "reg:logistic"：逻辑回归
>- "binary:logistic"：二分类逻辑回归，输出概率
>- "binary:logitraw"：二分类逻辑回归，输出是逻辑为0/1的前一步的分数
>- "gpu:reg:linear", "gpu:reg:logistic", "gpu:binary:logistic"：用GPU跑对应的回归，请注意，像GPU直方图算法一样，在整个训练过程中只能用相同的dataset
>- count:poisson"：计数问题的泊松回归，输出为泊松分布的平均值
  - 在泊松分布中，max_delta_step参数值默认为0.7（用于维护优化）
>- "multi:softmax"：用于Xgboost 做多分类问题，需要设置num_class（分类的个数）
>- "multi:softprob"：和softmax一样，只是输出的是一个向量（vector）ndata*nclass，进一步可以reshape成ndata，nclass的矩阵。结果就是包含每个数据属于每一类的概率
>- "rank:pairwise"：让Xgboost 做排名任务，通过最小化[**pairwise loss**](https://en.wikipedia.org/wiki/Learning_to_rank#Pairwise_approach)(Learn to rank的一种方法)
>- "reg:gamma"：gamma回归带有**日志链接（log-link）？？** 。输出gamma分布的均值。对于 保险索赔严重性建模 或者是任何符合[gamma分布]((https://en.wikipedia.org/wiki/Gamma_distribution#Applications))输出 可能是有用
>- "reg:tweedie"：Tweedie回归带有**日志链接（log-link）？？** . 对于 保险中的全部损失进行建模建模 或者是任何符合[tweedie分布](https://en.wikipedia.org/wiki/Tweedie_distribution#Applications))输出 可能是有用。


* base_score [default=0.5]
  - the initial prediction score of all instances, global bias
  - for sufficient number of iterations, changing this value will not have too much effect.

>- base_score,默认值为0.5
>- 所有实例的初始预测分数，全局偏置
>- 有足够的迭代次数，改变该值并没有多少影响

* eval_metric [default according to objective]
  - evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking )
  - User can add multiple evaluation metrics, for python user, remember to pass the metrics in as list of parameters pairs instead of map, so that latter 'eval_metric' won't override previous one
  - The choices are listed below:
    - "rmse": [root mean square error](http://en.wikipedia.org/wiki/Root_mean_square_error)
    - "mae": [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
    - "logloss": negative [log-likelihood](http://en.wikipedia.org/wiki/Log-likelihood)
    - "error": Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
    - "error@t": a different than 0.5 binary classification threshold value could be specified by providing a numerical value through 't'.
    - "merror": Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
    - "mlogloss": [Multiclass logloss](https://www.kaggle.com/wiki/LogLoss)
    - "auc": [Area under the curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_curve) for ranking evaluation.
    - "ndcg":[Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/NDCG)
    - "map":[Mean average precision](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision)
    - "ndcg@n","map@n": n can be assigned as an integer to cut off the top positions in the lists for evaluation.
    - "ndcg-","map-","ndcg@n-","map@n-": In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding "-" in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions.training repeatedly
  - "poisson-nloglik": negative log-likelihood for Poisson regression
  - "gamma-nloglik": negative log-likelihood for gamma regression
  - "gamma-deviance": residual deviance for gamma regression
  - "tweedie-nloglik": negative log-likelihood for Tweedie regression (at a specified value of the tweedie_variance_power parameter)

>- eval_metric:默认值取决于目标
>- 对于验证集的评估指标。默认指标根据目标的不同有所不同（回归问题指标：rmse，分类问题：error，排序问题：mean）
>- 用户可以添加多个评估指标。对于Python用户要以list传递参数对给程序，而不是map，这样后一个指标就不会覆盖前一个
>- 参数列表如下：
 - ’rmse'[root mean square error](http://en.wikipedia.org/wiki/Root_mean_square_error)剩余标准差(均方根误差)
 - "mae": [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)平均绝对误差
 - **"logloss"**: negative 负对数似然函数值 [log-likelihood](http://en.wikipedia.org/wiki/Log-likelihood)
 - ”error"：二分类错误率，计算公式（错误cases）/（全部cases），评估中，对于预测值大于0.5位正例，剩下为负例
 - "error@t"：二分类错误率，通过t指定阈值，而非0.5，超过阈值为正例，其余为负例
 - "merror"：多分类错误率。计算公式：（错误cases）/（全部cases）
 -  "mlogloss": [Multiclass logloss](https://www.kaggle.com/wiki/LogLoss)多分类logloass
 - "**auc**": [Area under the curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_curve) for ranking evaluation.曲线下面积-相关的ROC曲线
 - "**ndcg"**:[Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/NDCG)折扣累积收益（DCG）是衡量排名质量的一个指标。属于信息检索范围（Information retrieval）
 - "**map**":[Mean average precision](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision) 属于信息检索范围（Information retrieval）
 -  **"ndcg@n","map@n"**: 其中n为整数，在评估列表中截断顶部位置
 - "ndcg-","map-","ndcg@n-","map@n-":在XGBoost中，NDCG和MAP将评估没有任何正面样本的列表的评分为1.通过在评估量度中添加“ - ”，XGBoost会在某些情况下评估这些评分为0，然后不断的训练
>- "poisson-nloglik"：泊松分布回归的负似然性
>- "gamma-nloglik": gamma回归的负[gamma分布似然性](https://en.wikipedia.org/wiki/Likelihood_function#Example:_the_gamma_distribution)
>- "gamma-deviance"：γ回归的**剩余偏差（residual deviance）**
>- "tweedie-nloglik"：Tweedie回归负向似然值

* seed [default=0]
  - random number seed.

>- seed：默认为p0
>- 随机的number种子

Command Line Parameters
-----------------------
>- 命令行参数

The following parameters are only used in the console version of xgboost
* use_buffer [default=1]
  - Whether to create a binary buffer from text input. Doing so normally will speed up loading times
* num_round
  - The number of rounds for boosting
* data
  - The path of training data
* test:data
  - The path of test data to do prediction
* save_period [default=0]
  - the period to save the model, setting save_period=10 means that for every 10 rounds XGBoost will save the model, setting it to 0 means not saving any model during the training.
* task [default=train] options: train, pred, eval, dump
  - train: training using data
  - pred: making prediction for test:data
  - eval: for evaluating statistics specified by eval[name]=filename
  - dump: for dump the learned model into text format (preliminary)
* model_in [default=NULL]
  - path to input model, needed for test, eval, dump, if it is specified in training, xgboost will continue training from the input model
* model_out [default=NULL]
  - path to output model after training finishes, if not specified, will output like 0003.model where 0003 is number of rounds to do boosting.
* model_dir [default=models]
  - The output directory of the saved models during training
* fmap
  - feature map, used for dump model
* name_dump [default=dump.txt]
  - name of model dump file
* name_pred [default=pred.txt]
  - name of prediction file, used in pred mode
* pred_margin [default=0]
  - predict margin instead of transformed probability

>- 以下参数只能用在xgboost的控制台版本
>- use_buffer ：默认值为1
  - 是否为输入创建二进制的缓存文件，缓存文件通常可以加速加载次数
>- num_round
  - booting迭代的次数
>- data
  - 训练数据的路径
>- test:data
  - 测试数据的路径
>- save_period ：默认为0
  - 保存模型的周期，例如save_period=10，意味着训练每10周xgboost保存一次模型，0 代表不保存模型
>- task：默认train，可以选择train, pred, eval, dump
  - train：训练使用数据
  - pred：对测试数据进行预测
  - eval：通过eval[name]=filename定义评价指标
  - dump：将习得的模型保存文本而是
>- **model_in**：默认NULL
  - 指向模型的路径，测试、评价、保存数据都会用到，如果指明需要训练，xgboost将接着输入的模型进行训练
>- model_out：默认NULL
  - 训练结束后，输出的模型的路径，如果没有明确指明，模型输出将会是0003.model这样，0003代表boosting的第3次数训练的结果
>- model_dir ：默认models
  - 模型在训练期间保存的路径
>- **fmap**
  - 特征地图，用作保存模型
>- name_dump ：默认dump.txt
  - 模型的保存文件
>- name_pred：默认pred.txt
  - 预测文件，用在pred 模式
>- **pred_margin**：默认值0
  - 输出预测的边界而不是转换之后的概率

翻译中部分参考[XGBoost参数调优完全指南](http://blog.csdn.net/han_xiaoyang/article/details/52665396)
