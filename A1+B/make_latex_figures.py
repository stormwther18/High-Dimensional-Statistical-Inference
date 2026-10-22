#!/usr/bin/env python3
"""Generate self-contained native PGFPlots figure commands from saved summaries.

The input data are read without modification.  All plotted coordinates are
embedded in latex_figures.tex, so the resulting LaTeX needs no image files or
external data files.  Requires Python's standard library only.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("BinaryTree", "Exponential")
COLORS = ("aoneblue", "aoneorange", "aonegreen")
MARKS = ("*", "square*", "triangle*")


def read_csv(filename):
    with (ROOT / "data" / filename).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(value):
    return format(float(value), ".11g")


def coords(rows, ykey):
    return " ".join(f"({num(r['lambda_'])},{num(r[ykey])})" for r in rows)


def plot(options, coordinates):
    return rf"\addplot+[{options}] coordinates {{{coordinates}}};"


def common_style():
    return r"""width=0.43\textwidth,height=48mm,
  tick label style={font=\scriptsize},label style={font=\small},
  title style={font=\small},axis line style={black!65},
  tick align=outside,major grid style={black!12},ymajorgrids=true,
  every axis plot/.append style={mark=none},
  legend style={font=\scriptsize,draw=none,fill=none,
    /tikz/column sep=5pt,legend cell align=left},
  legend columns=5"""


def legend(name, include_theory=False):
    result = []
    for i, color in enumerate(COLORS):
        result += [rf"\addlegendimage{{{color},mark=none,line width=0.9pt}}",
                   rf"\addlegendentry{{S{i+1}}}"]
    if include_theory:
        result += [r"\addlegendimage{black,dashed,mark=none,line width=0.9pt}",
                   r"\addlegendentry{$R_\lambda^{\mathrm{DE}}$}",
                   r"\addlegendimage{black!45,densely dotted,mark=none,line width=0.7pt}",
                   r"\addlegendentry{$\lambda=1$}"]
    return "\n".join(result)


def risk_figure(scan, optima, seeds):
    out = [r"\newcommand{\RiskCurvesFigure}{%", r"\begin{tikzpicture}",
           r"\begin{groupplot}[" + common_style() + r""",
  group style={group size=2 by 2,horizontal sep=0.11\textwidth,vertical sep=22mm},
  xlabel={$\lambda$},ylabel={Prediction risk (MSE)}]"""]
    for row_index in range(2):
        for model_index, model in enumerate(MODELS):
            selected = [r for r in scan if r["model"] == model]
            theory = sorted([r for r in selected if int(r["master_seed"]) == seeds[0]],
                            key=lambda r: float(r["lambda_"]))
            options = []
            if row_index == 0:
                ymin = min(float(r["test_mean"]) for r in selected)
                ymax = max(float(r["test_mean"]) for r in selected)
                ymax = max(ymax, max(float(r["theory"]) for r in theory))
                options += ["xmode=log", "xmin=0.01", "xmax=10",
                            "xtick={0.01,0.1,1,10}",
                            f"ymin={num(ymin-0.08)}", f"ymax={num(ymax+0.15)}",
                            rf"title={{({chr(97+model_index)}) {model}}}"]
            else:
                zoom = [r for r in selected if .5 <= float(r["lambda_"]) <= 1.6]
                lows = [float(r["test_mean"]) for r in zoom] + [float(r["theory"]) for r in zoom]
                ymin, ymax = min(lows), max(lows)
                pad = (ymax-ymin)*.1
                options += ["xmin=0.5", "xmax=1.6", "xtick={0.5,1,1.5}",
                            f"ymin={num(ymin-pad)}", f"ymax={num(ymax+pad)}",
                            "scaled y ticks=false", "yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3}",
                            rf"title={{({chr(99+model_index)}) {model}: minimum region}}"]
            if row_index == model_index == 0:
                options.append("legend to name=AOneRiskLegend")
            out.append(r"\nextgroupplot[" + ",".join(options) + "]")
            for seed_index, seed in enumerate(seeds):
                rows = sorted([r for r in selected if int(r["master_seed"]) == seed],
                              key=lambda r: float(r["lambda_"]))
                out.append(plot(f"{COLORS[seed_index]},mark=none,line width=0.85pt,forget plot", coords(rows, "test_mean")))
            out.append(plot("black,dashed,mark=none,line width=1pt,forget plot", coords(theory, "theory")))
            out.append(r"\draw[black!45,densely dotted,line width=0.7pt] (axis cs:1,\pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:1,\pgfkeysvalueof{/pgfplots/ymax});")
            if row_index:
                for seed_index, seed in enumerate(seeds):
                    optimum = next(r for r in optima if r["model"] == model and int(r["master_seed"]) == seed)
                    out.append(plot(f"{COLORS[seed_index]},only marks,mark={MARKS[seed_index]},mark size=2pt,mark options={{fill={COLORS[seed_index]},draw=white,line width=0.4pt}},forget plot",
                                    f"({num(optimum['observed_optimal_lambda'])},{num(optimum['observed_optimal_test_mean'])})"))
            if row_index == model_index == 0:
                out.append(legend("AOneRiskLegend", include_theory=True))
    out += [r"\end{groupplot}",
            r"\node[anchor=south,yshift=9mm,xshift=6mm] at (group c1r1.north east) {\pgfplotslegendfromname{AOneRiskLegend}};",
            r"\end{tikzpicture}%", "}"]
    return "\n".join(out)


def components_figure(optima, seeds):
    # Intervals are mean +/- 1.96 SE.  Paired SE is essential for T-C.
    all_bounds = []
    for r in optima:
        for estimate, se in ((float(r["test_minus_theory"]), float(r["test_se_at_theory_optimum"])),
                             (float(r["conditional_minus_theory"]), float(r["conditional_se_at_theory_optimum"])),
                             (float(r["test_minus_conditional"]), float(r["paired_difference_se"]))):
            all_bounds.extend((estimate-1.96*se, estimate+1.96*se))
    limit = max(abs(v) for v in all_bounds)*1.12
    out = [r"\newcommand{\ErrorComponentsFigure}{%", r"\begin{tikzpicture}",
           r"\begin{groupplot}[" + common_style() + rf""",
  height=55mm,group style={{group size=2 by 1,horizontal sep=0.075\textwidth}},
  xmin=0.55,xmax=3.45,ymin={num(-limit)},ymax={num(limit)},
  xtick={{1,2,3}},xticklabels={{$\overline T-R_1^{{\mathrm{{DE}}}}$,$\overline C-R_1^{{\mathrm{{DE}}}}$,$\overline T-\overline C$}},
  ylabel={{Mean risk difference}},scaled y ticks=false,
  yticklabel style={{/pgf/number format/fixed,/pgf/number format/precision=2}}]"""]
    for model_index, model in enumerate(MODELS):
        opts = rf"title={{({chr(97+model_index)}) {model}}}"
        if model_index == 0:
            opts += ",legend to name=AOneComponentsLegend"
        out.append(r"\nextgroupplot["+opts+"]")
        out.append(r"\draw[black!55,line width=0.6pt] (axis cs:0.55,0) -- (axis cs:3.45,0);")
        for seed_index, seed in enumerate(seeds):
            r = next(r for r in optima if r["model"] == model and int(r["master_seed"]) == seed)
            values = ((r["test_minus_theory"], r["test_se_at_theory_optimum"]),
                      (r["conditional_minus_theory"], r["conditional_se_at_theory_optimum"]),
                      (r["test_minus_conditional"], r["paired_difference_se"]))
            points = " ".join(f"({num(j+1+(seed_index-1)*.16)},{num(mean)}) +- (0,{num(1.96*float(se))})"
                              for j, (mean, se) in enumerate(values))
            out.append(plot(f"{COLORS[seed_index]},only marks,mark={MARKS[seed_index]},mark size=2pt,mark options={{fill={COLORS[seed_index]},draw={COLORS[seed_index]}}},line width=0.8pt,error bars/.cd,y dir=both,y explicit,error bar style={{line width=0.8pt}},error mark options={{mark size=2pt}},/pgfplots/forget plot", points))
        if model_index == 0:
            for seed_index, color in enumerate(COLORS):
                out += [rf"\addlegendimage{{only marks,mark={MARKS[seed_index]},{color},mark options={{fill={color},draw={color}}}}}",
                        rf"\addlegendentry{{S{seed_index+1}}}"]
    out += [r"\end{groupplot}",
            r"\node[anchor=south,yshift=9mm,xshift=6mm] at (group c1r1.north east) {\pgfplotslegendfromname{AOneComponentsLegend}};",
            r"\end{tikzpicture}%", "}"]
    return "\n".join(out)


def optima_figure(optima, seeds):
    out = [r"\newcommand{\OptimaFigure}{%", r"\begin{tikzpicture}",
           r"\begin{groupplot}[" + common_style() + r""",
  height=54mm,group style={group size=2 by 1,horizontal sep=0.075\textwidth},
  xmin=0.5,xmax=7.5,xtick={1,2,3,5,6,7},
  xticklabels={BT1,BT2,BT3,EX1,EX2,EX3},
  xlabel={Model and seed}]"""]
    out += [r"\nextgroupplot[title={(a) Optimum displacement},ylabel={$\widehat\lambda-1$},",
            r" ymin=-0.08,ymax=0.065,ytick={-0.06,-0.03,0,0.03,0.06},scaled y ticks=false,",
            r" yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2}]",
            r"\draw[black!55,line width=0.6pt] (axis cs:0.5,0)--(axis cs:7.5,0);"]
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(seeds):
            r = next(r for r in optima if r["model"] == model and int(r["master_seed"]) == seed)
            out.append(plot(f"{COLORS[seed_index]},only marks,mark={MARKS[seed_index]},mark size=2.4pt,mark options={{fill={COLORS[seed_index]},draw={COLORS[seed_index]}}},forget plot",
                            f"({model_index*4+seed_index+1},{num(float(r['observed_optimal_lambda'])-1)})"))
    out += [r"\nextgroupplot[title={(b) Theoretical regret},",
            r" ylabel={$R_{\widehat\lambda}^{\mathrm{DE}}-R_1^{\mathrm{DE}}$},",
            r" ymin=-0.000025,ymax=0.0004,ytick={0,0.0001,0.0002,0.0003,0.0004},scaled y ticks=false,",
            r" yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=4,/pgf/number format/zerofill}]"]
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(seeds):
            r = next(r for r in optima if r["model"] == model and int(r["master_seed"]) == seed)
            out.append(plot(f"{COLORS[seed_index]},only marks,mark={MARKS[seed_index]},mark size=2.4pt,mark options={{fill={COLORS[seed_index]},draw={COLORS[seed_index]}}},forget plot",
                            f"({model_index*4+seed_index+1},{num(r['theory_regret_at_observed_optimum'])})"))
    out += [r"\end{groupplot}", r"\end{tikzpicture}%", "}"]
    return "\n".join(out)


def main():
    scan = read_csv("scan_summary.csv")
    optima = read_csv("optima_summary.csv")
    seeds = sorted({int(r["master_seed"]) for r in scan})
    assert len(seeds) == 3 and len(optima) == 6
    header = r"""% Generated by make_latex_figures.py; all coordinates derive from saved CSVs.
% Requires pgfplots (compat=1.18), groupplots; no external images or data files.
% S1=2026102201, S2=2026102202, S3=2026102203.
% RiskCurvesFigure: only mean curves; no confidence bands are shown.
% ErrorComponentsFigure: bars are pointwise 95% normal intervals, +/-1.96 SE.
% In its caption define overline T=test mean and overline C=conditional mean at lambda=1;
% Their difference uses the saved paired-difference SE, never independent-SE propagation.
% OptimaFigure: BT1-3=BinaryTree/S1-3; EX1-3=Exponential/S1-3.
% Point locations are observed grid minima; no argmin uncertainty is implied.
\definecolor{aoneblue}{HTML}{0072B2}
\definecolor{aoneorange}{HTML}{D55E00}
\definecolor{aonegreen}{HTML}{009E73}
"""
    output = ROOT / "latex_figures.tex"
    output.write_text(header + "\n\n" + "\n\n".join((risk_figure(scan, optima, seeds),
                        components_figure(optima, seeds), optima_figure(optima, seeds))) + "\n",
                      encoding="utf-8")
    print(f"Generated {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
