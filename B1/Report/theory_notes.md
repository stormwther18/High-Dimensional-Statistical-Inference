# 理论片段接口与核查说明

本文件仅供报告作者整合。`theory.tex` 是可 `\input` 的独立内容片段，不含导言区和参考文献。需要 `amsmath`、`amssymb`、`hyperref`；所有标签均以前缀 `eq:th_` 或 `sec:theory_` 命名，无额外自定义命令。引用键：`dw`、`richards`、`bellec`、`bach`、`wu`、`lin`。

## 两个 oracle 必须分开

- 一般先验平均矩阵 `C_X` 不含不可约噪声，目标为 `sigma2 + pi.T C_X pi`；适用于固定或只依赖 X 的权重。知生成参数的 oracle 不能知道本次 beta 或 epsilon 的实现值。
- 真实条件矩阵 `K_D` 含不可约噪声 `sigma2 * 11'`，条目为 `sigma2 + (bhat_j-beta)' Sigma (bhat_k-beta)`。它只用于评估或 beta-aware 事后 oracle。
- 交叉残差矩阵估计的是第二个 `K_D`，并非直接估计第一个 `C_X`。固定权重下先验期望可连接，但 Y 自适应权重不能简单代入 C_X 后声称为实际风险。

## 核对的主要文献

1. Dobriban, E., Wager, S. (2018). *High-Dimensional Asymptotics of Prediction: Ridge Regression and Classification*. Annals of Statistics 46(1), 247–279. <https://arxiv.org/abs/1507.03003>。单参数总体谱函数和导数为现有结论；本片段做差商后组合有限 K 矩阵，不作为原创理论宣称。需要固定正参数、有界总体谱等比例渐近条件。
2. Richards, D., Mourtada, J., Rosasco, L. (2021). *Asymptotics of Ridge(less) Regression under General Source Condition*. AISTATS, PMLR 130:3889–3897. <https://proceedings.mlr.press/v130/richards21b.html>。已有结构化信号先验/比例高维风险分析，故 source condition 本身不是新问题；一般 Ψ 的双 resolvent 偏差不能由各向同性单标量差商推出。
3. Bellec, P. C., Du, J.-H., Koriyama, T., Patil, P., Tan, K. (2025). *Corrected generalized cross-validation for finite ensembles of penalized estimators*. JRSSB 87(2):289–318, DOI 10.1093/jrsssb/qkae092. 可读全文：<https://arxiv.org/html/2310.01374v2>。
   - Section 3.1 Eq.(11), Eq.(12) 的 cross-risk intermediate estimators 允许不同惩罚。
   - Section 4.1 Theorem 1 Eq.(20) 是逐对条目的非渐近误差界，不要求共同 penalty。Eq.(21) 的 aggregate relative bound 才增加共同 penalty。
   - Section 6 明确不同 regularization parameters 可进入 intermediate estimators。
   - 取全数据 `I_j=[n]` 时 k=n，抽样策略成为退化的确定集合；Eq.(11)/(12) 条目均化为 `r_j'r_k/[n(1-df_j/n)(1-df_k/n)]`。
   - 不要用针对同 penalty、随机 k<n 子样本的 ordinary GCV 不一致定理来替代我们同一 X 不同 lambda 的反例。
4. Bach, F. (2021). *On the Effectiveness of Richardson Extrapolation in Data Science*. SIAM J Math Data Sci 3(4):1251–1277. DOI 10.1137/21M1397349. 作者正式 PDF：<https://www.di.ens.fr/~fbach/Richardson_SIAM_2021.pdf>。已含岭/kernel ridge 的跨正则参数外推，signed mixture 不是空白思路。
5. Wu, D., Xu, J. (2020). *On the Optimal Weighted l2 Regularization in Overparameterized Linear Regression*. NeurIPS 33. <https://proceedings.neurips.cc/paper/2020/hash/72e6d3238361fe70f22fb0ac624a7072-Abstract.html>。
6. Lin, L., Wu, J., Kakade, S. M., Bartlett, P. L., Lee, J. D. (2024). *Scaling Laws in Linear Regression: Compute, Parameters, and Data*. NeurIPS 37. <https://arxiv.org/html/2406.08466v2>。Assumption 3 的 source condition 包含先验期望和跨坐标零二阶矩；不是逐次随机 beta 的确定性坐标约束。

## 固定候选集推论的证明范围

在 Gaussian X/noise、固定 K 和 L、p/n 有界、||Sigma|| 有界、lambda_j 位于固定正紧区间、预测能量 beta'Sigma beta 有界的序列中，Bellec Theorem 1 Eq.(20) 给归一化 cross-risk error O(n^-1/2)。全数据时归一化分母是 d_j d_k，其中 d_j=1-df_j/n。

- Gaussian S 的 op norm 为 O_p(1)，且 d_j >= lambda_min/(||S||+lambda_min)，故分母以高概率远离0。
- Ridge singular operator bound `||bhat_j|| <= ||y||/(2 sqrt(n lambda_min))`，结合 `||y||²/n=O_p(1)` 与 bounded Sigma，得到真实风险对角 `K_D,jj=O_p(1)`。
- Markov 和固定有限 K 的并界得 `max|Khat-K_D|=O_p(n^-1/2)`。
- 确定性 L1 误差传递得 adaptive weights conditional-oracle excess <= `2 L² max_error + solver_error`。

这条推论是现有逐对界与初等推导的组合；不覆盖任意 K 增长、lambda_min→0 或不受约束 signed weights。若 L 随 n 增长，充分条件是 `L_n² max_error_n→0`，但片段没有替这种序列证明 max_error 的速率。

## 普通 GCV 的精确单纯形换元

令 d_j=1-df_j/n>0，Khat 为 cross-corrected Gram 矩阵。`u=D*pi/(d'pi)` 与 `pi=D^-1*u/(1'D^-1*u)` 双射单纯形。于是普通 aggregate GCV 的最小值与 cross-corrected QP 最小值相同，但前者实际采用 pi，后者采用 u。因此无需将普通 GCV 写为非凸 SLSQP 比率优化；同一个稳定 simplex QP 解经逆变换即可。

这只是有限样本代数机制，不意味着 naive GCV 每次都更差，也未确认原创性。signed L1 有界集合在这个换元下不保持原约束，不能原样沿用。

建议保存 shared_min_score、cross_weights u、naive_weights pi、d_j、L1(pi-u) 和 transform residual，以便独立验收。

## 幂律实验的限定

若 mu_i ∝ i^-a 重标到 tr Sigma=p，则 max(mu) 随 p 增长，不满足理论中统一有界谱；若不重标，经验谱可能退化 delta0。n=100、固定 p 的所有精确风险公式仍成立，Gaussian finite-dimensional source alignment 实验也有效，但不能用该增维路径宣称 DW/Bellec 渐近结论直接覆盖，或宣称复现无限维 Scaling Law 指数。
