import argparse
import json
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX results tables.")
    parser.add_argument(
        "--output",
        default="sn-article-template/results_tables.tex",
        help="Output path for LaTeX tables relative to project root.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    methods = [
        ("bsvae", "B-SVAE", "Results/final/runs_v2/bsvae"),
        ("mnar_svae", "Ablation", "Results/final/runs_v2/mnar_svae"),
        ("mice", "MICE", "Results/final/runs_v2/mice"),
        ("missforest", "missForest", "Results/final/runs_v2_tesla/missforest"),
        ("gain", "GAIN", "Results/final/runs_v2_tesla/gain"),
        ("miwae", "MIWAE", "Results/final/runs_v2/miwae"),
        (
            "mc_dropout_cox",
            "\\begin{tabular}[c]{@{}l@{}}MC Dropout\\\\Cox\\end{tabular}",
            "Results/final/runs_v2_tesla/mc_dropout_cox",
        ),
        ("deephit", "DeepHit", "Results/final/runs_v2_tesla/deephit"),
    ]

    imputing_methods = [
        ("bsvae", "B-SVAE", "Results/final/runs_v2/bsvae"),
        ("mnar_svae", "Ablation", "Results/final/runs_v2/mnar_svae"),
        ("mice", "MICE", "Results/final/runs_v2/mice"),
        ("missforest", "missForest", "Results/final/runs_v2_tesla/missforest"),
        ("gain", "GAIN", "Results/final/runs_v2_tesla/gain"),
        ("miwae", "MIWAE", "Results/final/runs_v2/miwae"),
    ]

    datasets = [
        ("metabric", "METABRIC"),
        ("mimic", "MIMIC-IV"),
        ("eicu", "eICU"),
    ]

    scenarios = ["light", "moderate", "severe"]

    missing_paths = []
    for m_key, _, m_rel in methods:
        base_dir = os.path.join(project_root, m_rel)
        for d_key, _ in datasets:
            for s_key in scenarios:
                summary_file = os.path.join(base_dir, f"{d_key}_{s_key}_summary.json")
                if not os.path.isfile(summary_file):
                    missing_paths.append(summary_file)
                for f in range(1, 6):
                    fold_file = os.path.join(base_dir, f"{d_key}_{s_key}_fold{f}_test.json")
                    if not os.path.isfile(fold_file):
                        missing_paths.append(fold_file)

    if missing_paths:
        for p in missing_paths:
            print(f"Missing file: {p}", file=sys.stderr)
        sys.exit(1)

    summaries = {}
    test_folds = {}

    for m_key, _, m_rel in methods:
        summaries[m_key] = {}
        test_folds[m_key] = {}
        base_dir = os.path.join(project_root, m_rel)
        for d_key, _ in datasets:
            summaries[m_key][d_key] = {}
            test_folds[m_key][d_key] = {}
            for s_key in scenarios:
                sum_path = os.path.join(base_dir, f"{d_key}_{s_key}_summary.json")
                with open(sum_path, "r", encoding="utf-8") as fp:
                    summaries[m_key][d_key][s_key] = json.load(fp)

                test_folds[m_key][d_key][s_key] = {}
                for f in range(1, 6):
                    fold_path = os.path.join(base_dir, f"{d_key}_{s_key}_fold{f}_test.json")
                    with open(fold_path, "r", encoding="utf-8") as fp:
                        test_folds[m_key][d_key][s_key][f] = json.load(fp)

    def get_mean_std_val(m_key, d_key, s_key, metric):
        data = summaries[m_key][d_key][s_key]
        if metric in data and isinstance(data[metric], dict):
            m_val = data[metric].get("mean")
            s_val = data[metric].get("std")
            if m_val is not None and s_val is not None:
                return float(m_val), float(s_val)
        return None, None

    def fmt_cell(m_key, d_key, s_key, metric, is_bold=False):
        m_val, s_val = get_mean_std_val(m_key, d_key, s_key, metric)
        if m_val is None:
            return "--"
        s = f"{m_val:.3f} ({s_val:.3f})"
        if is_bold:
            return f"\\textbf{{{s}}}"
        return s

    def fmt_cell_makecell(m_val, s_val, is_bold=False, count_lt=None, decimals=3):
        if m_val is None:
            return "--"
        m_str = f"{m_val:.{decimals}f}"
        s_str = f"({s_val:.{decimals}f})"
        if count_lt is not None:
            c_str = f"[{count_lt}]"
            if is_bold:
                m_str = f"\\textbf{{{m_str}}}"
                s_str = f"\\textbf{{{s_str}}}"
                c_str = f"\\textbf{{{c_str}}}"
            return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{m_str}\\\\{s_str}\\\\{{{c_str}}}\\end{{tabular}}"
        if is_bold:
            m_str = f"\\textbf{{{m_str}}}"
            s_str = f"\\textbf{{{s_str}}}"
        return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{m_str}\\\\{s_str}\\end{{tabular}}"

    def build_transposed_comparison_table(
        metric_key,
        metric_name,
        label_suffix,
        method_subset,
        bold_criterion=None,
        extra_caption="",
        append_lt_005=False,
        decimals=3,
        font_size="\\scriptsize",
        tabcolsep="2pt",
    ):
        cols = []
        for d_key, _ in datasets:
            for s_key in scenarios:
                cols.append((d_key, s_key))

        best_per_col = {}
        for d_key, s_key in cols:
            vals = []
            for m in method_subset:
                m_key = m[0]
                m_val, _ = get_mean_std_val(m_key, d_key, s_key, metric_key)
                if m_val is not None:
                    vals.append(m_val)
            if vals and bold_criterion is not None:
                if bold_criterion == "min":
                    best_per_col[(d_key, s_key)] = min(vals)
                elif bold_criterion == "max":
                    best_per_col[(d_key, s_key)] = max(vals)
            else:
                best_per_col[(d_key, s_key)] = None

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
            "\\scriptsize",
            f"\\caption{{{metric_name}. Mean (SD) over five test folds.{extra_caption}}}\\label{{tab:res_{label_suffix}}}",
            "\\begin{tabular}{lccccccccc}",
            "\\toprule",
            " & \\multicolumn{3}{c}{METABRIC} & \\multicolumn{3}{c}{MIMIC-IV} & \\multicolumn{3}{c}{eICU} \\\\",
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
            "Method & Light & Mod. & Sev. & Light & Mod. & Sev. & Light & Mod. & Sev. \\\\",
            "\\midrule",
        ]

        for m in method_subset:
            m_key = m[0]
            m_lbl = m[1]
            row_cells = [m_lbl]
            for d_key, s_key in cols:
                m_val, s_val = get_mean_std_val(m_key, d_key, s_key, metric_key)
                is_best = False
                best_val = best_per_col.get((d_key, s_key))
                if best_val is not None and m_val is not None:
                    if abs(m_val - best_val) < 1e-12:
                        is_best = True

                count_lt = None
                if append_lt_005:
                    count_lt = 0
                    for f in range(1, 6):
                        f_m = test_folds[m_key][d_key][s_key][f].get("metrics", {})
                        val = f_m.get(metric_key)
                        if val is not None and float(val) < 0.05:
                            count_lt += 1

                cell_str = fmt_cell_makecell(
                    m_val, s_val, is_bold=is_best, count_lt=count_lt, decimals=decimals
                )
                row_cells.append(cell_str)
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    t1_rmse = build_transposed_comparison_table(
        "rmse_std", "Root Mean Squared Error (RMSE)", "rmse", imputing_methods, bold_criterion="min"
    )
    t2_cindex = build_transposed_comparison_table(
        "cindex",
        "Concordance Index (C-index)",
        "cindex",
        methods,
        bold_criterion="max",
        tabcolsep="1.4pt",
    )
    t3_ibs = build_transposed_comparison_table(
        "ibs",
        "Integrated Brier Score (IBS)",
        "ibs",
        methods,
        bold_criterion="min",
        tabcolsep="1.4pt",
    )
    t4_tdauc = build_transposed_comparison_table(
        "td_auc_mean",
        "Time-Dependent AUC",
        "td_auc",
        methods,
        bold_criterion="max",
        tabcolsep="1.4pt",
    )
    t5_dcal = build_transposed_comparison_table(
        "dcal_pval",
        "D-calibration p-value",
        "dcal",
        methods,
        bold_criterion=None,
        extra_caption=" Brackets: number of folds with $p < 0.05$.",
        append_lt_005=True,
        font_size="\\scriptsize",
    )

    t6_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\scriptsize",
        "\\caption{B-SVAE Uncertainty Calibration. Mean (SD) over five test folds.}\\label{tab:res_bsvae_uncertainty}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Cohort & Severity & PICP & MPIW & ECE \\\\",
        "\\midrule",
    ]
    for d_key, d_lbl in datasets:
        for s_idx, s_key in enumerate(scenarios):
            c_label = d_lbl if s_idx == 0 else ""
            s_label = s_key.capitalize()
            picp_str = fmt_cell("bsvae", d_key, s_key, "picp", is_bold=False)
            mpiw_str = fmt_cell("bsvae", d_key, s_key, "mpiw", is_bold=False)
            ece_str = fmt_cell("bsvae", d_key, s_key, "ece", is_bold=False)
            t6_lines.append(f"{c_label} & {s_label} & {picp_str} & {mpiw_str} & {ece_str} \\\\")
        if d_key != datasets[-1][0]:
            t6_lines.append("\\midrule")
    t6_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    t6_uncertainty = "\n".join(t6_lines)

    t7_mae = build_transposed_comparison_table(
        "mae_std", "Mean Absolute Error (MAE)", "mae", imputing_methods, bold_criterion="min"
    )

    iwae_methods = [
        ("bsvae", "B-SVAE", "Results/final/runs_v2/bsvae"),
        ("mnar_svae", "Ablation", "Results/final/runs_v2/mnar_svae"),
    ]
    t8_iwae = build_transposed_comparison_table(
        "iwae_mean",
        "Importance-Weighted Autoencoder Bound (IWAE)",
        "iwae",
        iwae_methods,
        bold_criterion=None,
        extra_caption=" Values are not comparable across models with different observation likelihoods.",
        decimals=1,
    )

    comp_cindex_methods = [m for m in methods if m[0] != "bsvae"]
    comp_rmse_methods = [m for m in imputing_methods if m[0] != "bsvae"]

    cols = []
    for d_key, _ in datasets:
        for s_key in scenarios:
            cols.append((d_key, s_key))

    t9_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\scriptsize",
        "\\caption{Fold-wise Wins of B-SVAE. Number of test folds in which B-SVAE was better.}\\label{tab:res_wins}",
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{METABRIC} & \\multicolumn{3}{c}{MIMIC-IV} & \\multicolumn{3}{c}{eICU} & \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "Method & Light & Mod. & Sev. & Light & Mod. & Sev. & Light & Mod. & Sev. & Total \\\\",
        "\\midrule",
        "\\multicolumn{11}{l}{\\textbf{C-index}} \\\\",
    ]

    for m in comp_cindex_methods:
        m_key = m[0]
        m_lbl = m[1]
        row_cells = [m_lbl]
        total_wins = 0
        for d_key, s_key in cols:
            wins = 0
            for f in range(1, 6):
                bsvae_m = test_folds["bsvae"][d_key][s_key][f].get("metrics", {})
                other_m = test_folds[m_key][d_key][s_key][f].get("metrics", {})
                b_val = bsvae_m.get("cindex")
                o_val = other_m.get("cindex")
                if b_val is not None and o_val is not None:
                    if float(b_val) > float(o_val):
                        wins += 1
            row_cells.append(str(wins))
            total_wins += wins
        row_cells.append(str(total_wins))
        t9_lines.append(" & ".join(row_cells) + " \\\\")

    t9_lines.append("\\midrule")
    t9_lines.append("\\multicolumn{11}{l}{\\textbf{RMSE}} \\\\")

    for m in comp_rmse_methods:
        m_key = m[0]
        m_lbl = m[1]
        row_cells = [m_lbl]
        total_wins = 0
        for d_key, s_key in cols:
            wins = 0
            for f in range(1, 6):
                bsvae_m = test_folds["bsvae"][d_key][s_key][f].get("metrics", {})
                other_m = test_folds[m_key][d_key][s_key][f].get("metrics", {})
                b_val = bsvae_m.get("rmse_std")
                o_val = other_m.get("rmse_std")
                if b_val is not None and o_val is not None:
                    if float(b_val) < float(o_val):
                        wins += 1
            row_cells.append(str(wins))
            total_wins += wins
        row_cells.append(str(total_wins))
        t9_lines.append(" & ".join(row_cells) + " \\\\")

    t9_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    t9_wins = "\n".join(t9_lines)

    tables = [
        t1_rmse,
        t2_cindex,
        t3_ibs,
        t4_tdauc,
        t5_dcal,
        t6_uncertainty,
        t7_mae,
        t8_iwae,
        t9_wins,
    ]

    out_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n\n".join(tables) + "\n")

    print(f"Written {len(tables)} tables to {out_path}")


if __name__ == "__main__":
    main()
