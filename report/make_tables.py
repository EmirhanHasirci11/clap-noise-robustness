"""Turn results/*.csv into LaTeX tables and number macros used by report.tex.

    python report/make_tables.py            # writes report/generated/*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "generated"

METHOD_TEX = {"zero-shot": "Zero-shot", "tip-clean": "Tip (clean cache)", "tip-aug": "Tip (clean+noisy) \\textbf{[ours]}",
              "tip-noisy-only": "Tip (noisy only)", "lp-clean": "LP (clean)", "lp-aug": "LP (clean+noisy)"}
ORDER = list(METHOD_TEX)


DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
          "6": "six", "7": "seven", "8": "eight", "9": "nine"}
NUMS = {"20": "twenty", "16": "sixteen", "10": "ten"}


def clean_name(name: str) -> str:
    for k, v in NUMS.items():
        name = name.replace(k, v)
    name = "".join(DIGITS.get(ch, ch) for ch in name)
    return "".join(ch for ch in name if ch.isalpha())


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{clean_name(name)}}}{{{value}}}\n"


def plain(m: str) -> str:
    return METHOD_TEX.get(m, m).replace(" \\textbf{[ours]}", "")


def f1(x):
    return f"{x:.1f}"


def compact_table(res: Path, name: str) -> list[str]:
    """Methods x K with clean / per-noise-type averages, from one results directory."""
    s = pd.read_csv(res / "summary_fewshot.csv")
    s["group"] = s["condition"].map(lambda c: "clean" if c == "clean" else c.split("_")[0])
    g = s.groupby(["method", "k", "group"])["mean"].mean().unstack("group")
    g["noisy"] = s[s.group != "clean"].groupby(["method", "k"])["mean"].mean()
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Method & $K$ & Clean & White avg & Pink avg & Brown avg & Noisy avg \\\\", "\\midrule"]
    macros = []
    zs = g.loc[("zero-shot", 0)]
    lines.append(f"Zero-shot & -- & {zs['clean']:.1f} & {zs['white']:.1f} & {zs['pink']:.1f} & {zs['brown']:.1f} & {zs['noisy']:.1f} \\\\")
    for k in sorted(k for k in g.index.get_level_values("k").unique() if k > 0):
        lines.append("\\midrule")
        sub = g.xs(k, level="k")
        best = sub.max()
        for m in [m for m in ORDER if m in sub.index and m != "zero-shot"]:
            row = sub.loc[m]
            cells = []
            for c in ["clean", "white", "pink", "brown", "noisy"]:
                t = f"{row[c]:.1f}"
                cells.append(f"\\textbf{{{t}}}" if abs(row[c] - best[c]) < 1e-9 else t)
            lines.append(f"{METHOD_TEX[m]} & {k} & " + " & ".join(cells) + " \\\\")
            for c in ["clean", "white", "pink", "brown", "noisy"]:
                macros.append(macro(f"{name}{m.replace('-', '')}K{k}{c}", f"{row[c]:.1f}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"tab_compact_{name}.tex").write_text("\n".join(lines))
    return macros


def extended_tables(ext: Path) -> list[str]:
    macros = compact_table(ext, "ext")
    p = pd.read_csv(ext / "table3_paired.csv")
    for r in p.itertuples():
        comp = r.comparison.replace(" - ", "vs").replace("-", "").replace("zeroshot", "zs")
        macros.append(macro(f"extd{comp}K{r.k}{r.noise}", f"{r.mean_diff_pp:+.2f}"))
        macros.append(macro(f"extci{comp}K{r.k}{r.noise}", f"[{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}]"))
    runs = pd.read_csv(ext / "fewshot_runs.csv")
    c = runs[(runs.condition == "clean") & runs.method.str.startswith("tip")]
    macros.append(macro("extalphaedge", str(int((c.alpha == c.alpha.max()).sum()))))
    macros.append(macro("extalphaN", str(len(c))))
    return macros


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    macros = []

    macros += compact_table(RES, "main")
    runs0 = pd.read_csv(RES / "fewshot_runs.csv")
    c0 = runs0[(runs0.condition == "clean") & runs0.method.str.startswith("tip")]
    macros.append(macro("mainalphaedge", str(int((c0.alpha == c0.alpha.max()).sum()))))
    macros.append(macro("mainalphaN", str(len(c0))))

    # ---- reproduction
    rep = pd.read_csv(RES / "reproduction.csv")
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Checkpoint & Prompt & All 2000 & Fold mean $\\pm$ sd & Fold-1 R@1 & Fold-1 R@5 \\\\", "\\midrule"]
    tmpl_tex = {"This is a sound of {}.": "\\texttt{This is a sound of \\{\\}.}",
                "This a sound of {}.": "\\texttt{This a sound of \\{\\}.}", "{}": "\\texttt{\\{\\}}"}
    prev = None
    for r in rep.to_dict("records"):
        ck = r["checkpoint"] if r["checkpoint"] != prev else ""
        prev = r["checkpoint"]
        lines.append(f"\\texttt{{{ck}}} & {tmpl_tex.get(r['template'], r['template'])} & {100 * r['acc_all']:.2f} & "
                     f"{100 * r['acc_fold_mean']:.2f} $\\pm$ {100 * r['acc_fold_std']:.2f} & "
                     f"{100 * r['fold1_R@1']:.2f} & {100 * r['fold1_R@5']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_reproduction.tex").write_text("\n".join(lines))
    for r in rep.to_dict("records"):
        key = r["checkpoint"].replace("630k-", "").replace("-best", "").replace("-", "")
        t = {"This is a sound of {}.": "A", "This a sound of {}.": "B", "{}": "C"}.get(r["template"], "X")
        macros.append(macro(f"rep{key}{t}", f"{100 * r['acc_all']:.2f}"))
        macros.append(macro(f"repfold{key}{t}", f"{100 * r['fold1_R@1']:.2f}"))

    # ---- few-shot summary
    s = pd.read_csv(RES / "summary_fewshot.csv")
    runs = pd.read_csv(RES / "fewshot_runs.csv")
    conds = list(dict.fromkeys(runs["condition"]))
    noisy = [c for c in conds if c != "clean"]
    for k in sorted(runs.loc[runs.k > 0, "k"].unique()):
        sub = s[(s.k == k) | (s.method == "zero-shot")]
        piv = sub.pivot_table(index="method", columns="condition", values="mean")
        sd = sub.pivot_table(index="method", columns="condition", values="std")
        piv = piv.reindex([m for m in ORDER if m in piv.index])
        groups = {g: [c for c in noisy if c.startswith(g + "_")] for g in ["white", "pink", "brown"]}
        groups = {g: v for g, v in groups.items() if v}
        cols = ["clean"] + [c for g in groups.values() for c in g]
        head = "Method & Clean & " + " & ".join(c.split("_")[1].replace("dB", "") for c in cols[1:]) + " & Noisy avg \\\\"
        gl = " & & " + " & ".join(f"\\multicolumn{{{len(v)}}}{{c}}{{{g} (dB)}}" for g, v in groups.items()) + " & \\\\"
        lines = ["\\begin{tabular}{l" + "r" * (len(cols) + 1) + "}", "\\toprule", gl, head, "\\midrule"]
        best = {c: piv[c].max() for c in cols}
        for m, row in piv.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                txt = f1(v)
                if abs(v - best[c]) < 1e-9:
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
            cells.append(f1(row[noisy].mean()))
            lines.append(f"{METHOD_TEX[m]} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / f"tab_fewshot_k{k}.tex").write_text("\n".join(lines))
        for m, row in piv.iterrows():
            mk = m.replace("-", "").replace("zeroshot", "zs")
            macros.append(macro(f"acc{mk}K{k}clean", f1(row["clean"])))
            macros.append(macro(f"acc{mk}K{k}noisy", f1(row[noisy].mean())))
            for g, cs in groups.items():
                macros.append(macro(f"acc{mk}K{k}{g}", f1(row[cs].mean())))
            for c in cols:
                tag = c.replace("_", "").replace("dB", "db")
                macros.append(macro(f"acc{mk}K{k}{tag}", f1(row[c])))
        sdm = sd.reindex(piv.index)
        for m, row in sdm.iterrows():
            mk = m.replace("-", "").replace("zeroshot", "zs")
            macros.append(macro(f"sd{mk}K{k}clean", f1(row["clean"])))

    # ---- paired differences
    p = pd.read_csv(RES / "table3_paired.csv")
    keep = ["tip-aug - tip-clean", "lp-aug - lp-clean", "tip-clean - zero-shot", "tip-aug - zero-shot",
            "tip-noisy-only - tip-clean", "tip-aug - lp-aug"]
    lines = ["\\begin{tabular}{lrlrrc}", "\\toprule",
             "Comparison & $K$ & Noise & $\\Delta$ (pp) & 95\\% CI & better/worse runs \\\\", "\\midrule"]
    for comp in keep:
        sub = p[p.comparison == comp]
        first = True
        for r in sub[sub.k == sub.k.max()].sort_values(["noise"]).itertuples():
            name = comp.replace("tip-", "Tip-").replace("lp-", "LP-") if first else ""
            first = False
            lines.append(f"{name} & {r.k} & {r.noise} & {r.mean_diff_pp:+.2f} & [{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}] & "
                         f"{r.runs_better}/{r.runs_worse} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (OUT / "tab_paired.tex").write_text("\n".join(lines))
    for r in p.itertuples():
        comp = r.comparison.replace(" - ", "vs").replace("-", "").replace("zeroshot", "zs")
        kk = str(r.k)
        macros.append(macro(f"d{comp}K{kk}{r.noise}", f"{r.mean_diff_pp:+.2f}"))
        macros.append(macro(f"ci{comp}K{kk}{r.noise}", f"[{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}]"))

    # ---- resources
    enc = json.loads((RES / "resources_encoder.json").read_text())
    for k, v in enc.items():
        if isinstance(v, float):
            macros.append(macro("enc" + k, f"{v:.1f}"))
        elif isinstance(v, str):
            macros.append(macro("enc" + k, v.replace("_", "\\_")))
    ad = pd.read_csv(RES / "resources_adapters.csv")
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Method & $K$ & Entries & Size (MB) & Tuning (s) & Inference (ms/clip) \\\\", "\\midrule"]
    for r in ad.sort_values(["k", "method"]).itertuples():
        lines.append(f"{plain(r.method)} & {r.k} & {r.cache_entries:.0f} & "
                     f"{r.cache_MB:.3f} & {r.tune_s:.3f} & {r.infer_ms_per_clip:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_resources.tex").write_text("\n".join(lines))

    # ---- failure analysis
    pc = pd.read_csv(RES / "per_class.csv")
    top = pc.sort_values("drop_pp", ascending=False).head(8)
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Class & Clean & Noisy avg & Drop (pp) \\\\", "\\midrule"]
    for r in top.itertuples():
        lines.append(f"{r._1} & {r.zs_clean:.1f} & {r.zs_noisy_avg:.1f} & {r.drop_pp:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_drop.tex").write_text("\n".join(lines))
    if "aug_gain_pp" in pc:
        lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Class & Tip clean & Tip clean+noisy & $\\Delta$ \\\\", "\\midrule"]
        g = pc.sort_values("aug_gain_pp", ascending=False)
        for r in pd.concat([g.head(5), g.tail(5)]).itertuples():
            tc, ta = pc.loc[r.Index, "tip-clean_noisy_avg"], pc.loc[r.Index, "tip-aug_noisy_avg"]
            lines.append(f"{r._1} & {tc:.1f} & {ta:.1f} & {r.aug_gain_pp:+.1f} \\\\")
            if r.Index == g.head(5).index[-1]:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / "tab_classgain.tex").write_text("\n".join(lines))
        macros.append(macro("nClassGain", str(int((pc.aug_gain_pp > 0).sum()))))
        macros.append(macro("nClassLoss", str(int((pc.aug_gain_pp < 0).sum()))))
    col = pd.read_csv(RES / "table7_collapse.csv")
    lines = ["\\begin{tabular}{llp{7.2cm}r}", "\\toprule", "Method & Condition & Top-3 predicted classes (count) & Top-3 share (\\%) \\\\", "\\midrule"]
    for r in col.itertuples():
        lines.append(f"{plain(r.method)} & {r.condition.replace('_', ' ')} & {r.top_predicted} & {r.share_top3_pct:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_collapse.tex").write_text("\n".join(lines))

    ext = RES / "extended_grid"
    if (ext / "summary_fewshot.csv").exists():
        macros += extended_tables(ext)

    env = json.loads((RES / "environment.json").read_text()) if (RES / "environment.json").exists() else {}
    macros.append(macro("envgpu", str(env.get("gpu", "n/a")).replace("_", "\\_")))
    macros.append(macro("envcommit", str(env.get("git_commit", "n/a"))[:10]))
    (OUT / "macros.tex").write_text("".join(dict.fromkeys(macros)))
    print(f"wrote {len(macros)} macros and tables to {OUT}")


if __name__ == "__main__":
    main()
