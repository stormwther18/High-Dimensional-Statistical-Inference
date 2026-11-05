> **GitHub 上传版：**完整逐候选权重表无损存为 `Results/weights.csv.gz`。首次直接核验既有结果前，请在本目录运行 `python3 restore_data.py` 还原并校验；全量重跑实验会自动重新生成该表。其他报告、科学代码和结果保持不变。

# B：多岭参数平均的风险、信号谱对齐与选权代价

本目录交付完整首轮研究：代码、2,820 份独立训练数据的原始评估结果、图表、数学证明和实际编译的 LaTeX 报告。
原来的 A1/A2 文件和组员周报未作修改。

## 先看结果

- 报告：`Report/B_研究报告.pdf`；完整可编辑源文件：`Report/B_研究报告.tex`。
- 主实验：18 配置 × 100 独立重复 = 1,800 份训练数据；12 方法共享每份数据，合计 21,600 条方法记录。
- 独立验证：300 份数据检验 MP/GCV 反例；720 份数据检验有界谱确定性等价。
- 连续参数增广诊断复用 1,800 个主实验设计，不计为新的独立样本。
- 观察到有限的 oracle 结构收益；实际交叉残差/LOO 选权改善普通 GCV 平均，但本次没有胜过 Ridge-GCV。负权重扩大可达家族，也扩大了实际估计代价。
- 多参数平均、source condition 和风险校正已有文献。本项目不宣称已确认论文级原创性。

## 复现

执行环境已测试 Python 3.12，依赖版本固定在 `requirements.txt`。
生成 PDF 还需要支持 `ctexart`、Fandol 中文字体和 `newtx` 的 XeLaTeX（如完整 TeX Live/MacTeX）。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash run_all.sh
```

主数值实验本机约两分钟；首次安装库和其他机器的计算时间会不同。
若只需重建图表和报告，可保留已经交付的 Results，执行：

```bash
python3 Code/verify_results.py
python3 Code/analyze_results.py
python3 Code/build_report.py
```

`verify_results.py` 会重跑每配置的首个重复，检查种子复现、oracle 顺序、各向同性 Bayes 控制、权重变换和优化证书。
`test_core.py` 包含 6 组测试，其中一组覆盖曾触发数值求解回退的病态权重问题。
不同平台的数值库可能产生微小差异；实际风险差与优化间隙应共同检查。

## 方法与风险定义

岭回归采用 `||y-X beta||^2/n + lambda ||beta||^2`。
主实验固定 n=100、p/n∈{0.8,1,1.5}、a∈{1.5,2}、b-a∈{-0.3,0,0.3}；
总体谱按 i^(-a) 排列并规范到平均特征值 1。
系数方差按 i^(a-b) 分配，并固定总体预测信噪比 1。
响应仍为正确的线性高斯模型；改变的是信号先验与协方差谱的对齐。

`conditional_risk` 是给定 X 后、对系数先验和训练噪声平均的精确风险，仅报告于相应 oracle。
`risk` 是拟合后用 `1+(beta_hat-beta)' Sigma (beta_hat-beta)` 独立评估的总体风险；新测试点已精确积分。
数据选权方法的 `conditional_risk` 刻意留 NaN，防止把固定权重公式错误用于依赖 y 的权重。

Oracle 方法只用 X 和已知生成参数，不使用本次真实 beta 或训练噪声。
实际方法仅用训练 X,y；真实 beta 只用于评估。
主候选为 [1e-4,100] 上 21 个对数点，再加入 p/n。
连续单岭 oracle 在 [1e-6,1e4] 数值搜索；各向同性使用已知精确参数 p/n。
`check_continuous_anchor.py` 只对 oracle 追加其连续最优参数，诊断有限网格混杂。

## 文件说明

- `Code/ridge_core.py`：完整样本谱、一般先验风险矩阵、岭路径、风险评估、凸/负权重求解及 GCV 换元。
- `Code/run_experiment.py`：主实验和种子设计。
- `Code/validate_asymptotics.py`：MP 与有界谱验证。
- `Code/check_continuous_anchor.py`：复用同一 X 的连续参数增广诊断。
- `Code/analyze_results.py`：配对统计、校准、尾部统计和六组矢量图。
- `Code/verify_results.py`：完整结果审计与首个重复重放。
- `Code/build_report.py`：将原始汇总表与数学片段生成完整 LaTeX 并实际编译。
- `Results/main_raw.csv`：21,600 条方法记录。
- `Results/weights.csv`：逐候选权重与残差自由度分母。
- `Results/paired_risk.csv`、`oracle_gains.csv`、`calibration.csv`：配对差、条件收益与均值校准比。
- `Results/anchor_diagnostic.csv`、`anchor_summary.csv`：增广候选后的条件收益与证书。
- `Results/mechanism.csv`：普通 GCV 与交叉残差权重的换元误差、相同最小分数。
- `Results/manifest.json`、`verification.json`：数据集数量、软件版本、种子方案、求解精度和重放记录。
- `Figures/`：PDF 矢量图，另附 SVG 和 PNG。
- `Report/theory.tex`、`theory_notes.md`：数学命题、证明及现有文献的具体条件。

## 统计解释的限制

误差条为独立训练重复上的 Monte Carlo 标准误；差值使用共享重复的配对标准误。
95% 区间为逐项正态近似，没有多重比较校正。
最小范数在 p=n 的高斯设计下理论平均风险无穷，因此其样本均值仅作描述，普通均值 SE 留空，并另报尾部分位数。
幂律谱重标后随维数增大不满足统一有界谱条件；本次幂律结果是有限维机制实验，渐近公式验证另用单位谱和有界两块谱。
没有将这些结果解释为无限维 Scaling Law 指数定理，或任意候选数、任意趋零正则参数下的选权保证。
