import argparse
import contextlib
import csv
import json
import os
import re
import sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_ipcw(project_root):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import harness_data
    import harness_metrics

    search_dirs = [
        os.path.join(project_root, "Results", "final", "runs_v2"),
        os.path.join(project_root, "Results", "final", "runs_v2_tesla"),
    ]

    target_folders = []
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for entry in sorted(os.listdir(base_dir)):
            sub_dir = os.path.join(base_dir, entry)
            if os.path.isdir(sub_dir):
                npz_files = [f for f in os.listdir(sub_dir) if f.endswith("_test.npz")]
                if npz_files:
                    target_folders.append((entry, sub_dir))

    pattern = re.compile(r"^(.+)_([^_]+)_fold(\d+)_test\.npz$")

    rows = []
    n_processed = 0
    n_skipped = 0
    max_check_auc_diff = 0.0
    max_check_cindex_diff = 0.0

    for method_folder, folder_path in target_folders:
        filenames = sorted(os.listdir(folder_path))
        for fname in filenames:
            m = pattern.match(fname)
            if not m:
                continue

            dataset = m.group(1)
            scenario = m.group(2)
            k = int(m.group(3))

            npz_path = os.path.join(folder_path, fname)
            try:
                npz = np.load(npz_path)
            except Exception:
                n_skipped += 1
                continue

            if "cohort_index" not in npz or "eta" not in npz or "time_grid" not in npz:
                n_skipped += 1
                continue

            try:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull):
                        fd = harness_data.load_fold(dataset, scenario, k)
                        tr = fd.part("train")
                        te = harness_data.load_test(dataset, scenario, k, fd).part("test")
            except Exception:
                n_skipped += 1
                continue

            if not np.array_equal(te["cohort_index"], npz["cohort_index"]):
                n_skipped += 1
                continue

            eta = npz["eta"]
            times = npz["time_grid"]

            # Checks against stored values
            auc_res = harness_metrics.td_auc_unweighted(te["T"], te["E"], eta, times)
            mean_calc_auc = (
                float(auc_res[1]) if isinstance(auc_res, tuple) else float(np.mean(auc_res))
            )
            if "td_auc" in npz:
                mean_stored_auc = float(np.mean(npz["td_auc"]))
                check_auc_diff = abs(mean_calc_auc - mean_stored_auc)
            else:
                check_auc_diff = 0.0

            calc_cindex = float(harness_metrics.cindex(te["T"], te["E"], eta))
            json_fname = f"{dataset}_{scenario}_fold{k}_test.json"
            json_path = os.path.join(folder_path, json_fname)
            check_cindex_diff = 0.0
            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        test_json = json.load(jf)
                    stored_cindex = test_json.get("metrics", {}).get(
                        "cindex", test_json.get("cindex")
                    )
                    if stored_cindex is not None:
                        check_cindex_diff = abs(calc_cindex - float(stored_cindex))
                except Exception:
                    pass

            # New metrics
            auc_arr, auc_ipcw, n_dropped = harness_metrics.td_auc_ipcw(
                tr["T"], tr["E"], te["T"], te["E"], eta, times
            )
            tau = float(np.max(times))
            uno_c = harness_metrics.cindex_uno(
                tr["T"], tr["E"], te["T"], te["E"], eta, tau=tau
            )

            row = {
                "method_folder": method_folder,
                "dataset": dataset,
                "scenario": scenario,
                "fold": k,
                "auc_ipcw": float(auc_ipcw) if not np.isnan(auc_ipcw) else None,
                "uno_c": float(uno_c) if not np.isnan(uno_c) else None,
                "n_times_dropped": int(n_dropped),
                "check_auc_diff": float(check_auc_diff),
                "check_cindex_diff": float(check_cindex_diff),
            }
            rows.append(row)
            n_processed += 1
            max_check_auc_diff = max(max_check_auc_diff, check_auc_diff)
            max_check_cindex_diff = max(max_check_cindex_diff, check_cindex_diff)

    out_json = os.path.join(project_root, "Results", "final", "ipcw_metrics.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)

    # Print required outputs
    print(f"Files processed: {n_processed}, skipped: {n_skipped}")
    print(
        f"Maximum check_auc_diff: {max_check_auc_diff:.6e}, maximum check_cindex_diff: {max_check_cindex_diff:.6e}"
    )
    print()

    groups = defaultdict(lambda: {"auc": [], "uno": []})
    for r in rows:
        key = (r["method_folder"], r["dataset"], r["scenario"])
        if r["auc_ipcw"] is not None:
            groups[key]["auc"].append(r["auc_ipcw"])
        if r["uno_c"] is not None:
            groups[key]["uno"].append(r["uno_c"])

    def sort_key(key_tuple):
        m, d, s = key_tuple
        d_order = {"metabric": 0, "mimic": 1, "eicu": 2}.get(d.lower(), 9)
        s_order = {"light": 0, "moderate": 1, "severe": 2}.get(s.lower(), 9)
        return (m, d_order, s_order)

    header = f"{'Method Folder':<20} | {'Dataset':<10} | {'Scenario':<10} | {'IPCW TD-AUC (SD)':<22} | {'Uno C-index (SD)':<22}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    for key in sorted(groups.keys(), key=sort_key):
        m_folder, d_set, scen = key
        aucs = groups[key]["auc"]
        unos = groups[key]["uno"]

        auc_str = (
            f"{np.mean(aucs):.4f} ({np.std(aucs, ddof=1 if len(aucs) > 1 else 0):.4f})"
            if len(aucs) > 0
            else "N/A"
        )
        uno_str = (
            f"{np.mean(unos):.4f} ({np.std(unos, ddof=1 if len(unos) > 1 else 0):.4f})"
            if len(unos) > 0
            else "N/A"
        )
        print(f"{m_folder:<20} | {d_set:<10} | {scen:<10} | {auc_str:<22} | {uno_str:<22}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX results tables.")
    parser.add_argument(
        "--output",
        default="sn-article-template/results_tables.tex",
        help="Output path for LaTeX tables relative to project root.",
    )
    parser.add_argument(
        "--ipcw",
        action="store_true",
        help="Compute IPCW metrics and exit.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    if args.ipcw:
        run_ipcw(project_root)
        sys.exit(0)

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

    all_methods = methods + [
        ("bsvae_nolcal", "B-SVAE ($\\lambda = 0$)", "Results/final/runs_v2/bsvae_nolcal"),
        ("mice_mi", "MICE, posterior sampling", "Results/final/runs_v2/mice_mi"),
    ]

    datasets = [
        ("metabric", "METABRIC"),
        ("mimic", "MIMIC-IV"),
        ("eicu", "eICU"),
    ]

    scenarios = ["light", "moderate", "severe"]

    missing_paths = []
    for m_key, _, m_rel in all_methods:
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

    for m_key, _, m_rel in all_methods:
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

    ipcw_path = os.path.join(project_root, "Results", "final", "ipcw_metrics.json")
    if not os.path.isfile(ipcw_path):
        sys.exit(f"Error: {ipcw_path} not found.")

    with open(ipcw_path, "r", encoding="utf-8") as fp:
        ipcw_data = json.load(fp)

    ipcw_lookup = {}
    ipcw_uno_lookup = {}
    for entry in ipcw_data:
        mf = entry.get("method_folder", "")
        ds = entry.get("dataset", "")
        sc = entry.get("scenario", "")
        fd = entry.get("fold", 0)
        val = entry.get("auc_ipcw")
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            ipcw_lookup[(mf, ds, sc, int(fd))] = float(val)
            ipcw_lookup[(os.path.basename(mf), ds, sc, int(fd))] = float(val)
            ipcw_lookup[(mf.lower(), ds.lower(), sc.lower(), int(fd))] = float(val)
            ipcw_lookup[(os.path.basename(mf).lower(), ds.lower(), sc.lower(), int(fd))] = float(val)
        val_uno = entry.get("uno_c")
        if val_uno is not None and not (isinstance(val_uno, float) and np.isnan(val_uno)):
            ipcw_uno_lookup[(mf, ds, sc, int(fd))] = float(val_uno)
            ipcw_uno_lookup[(os.path.basename(mf), ds, sc, int(fd))] = float(val_uno)
            ipcw_uno_lookup[(mf.lower(), ds.lower(), sc.lower(), int(fd))] = float(val_uno)
            ipcw_uno_lookup[(os.path.basename(mf).lower(), ds.lower(), sc.lower(), int(fd))] = float(val_uno)

    for m_key, _, m_rel in all_methods:
        m_folder = os.path.basename(m_rel)
        for d_key, _ in datasets:
            for s_key in scenarios:
                for f in range(1, 6):
                    val = ipcw_lookup.get((m_folder, d_key, s_key, f))
                    if val is None:
                        val = ipcw_lookup.get((m_folder.lower(), d_key.lower(), s_key.lower(), f))
                    if val is not None:
                        if (
                            m_key in test_folds
                            and d_key in test_folds[m_key]
                            and s_key in test_folds[m_key][d_key]
                            and f in test_folds[m_key][d_key][s_key]
                        ):
                            metrics_dict = test_folds[m_key][d_key][s_key][f].setdefault("metrics", {})
                            metrics_dict["td_auc_mean"] = val
                            metrics_dict["td_auc"] = val
                    val_uno = ipcw_uno_lookup.get((m_folder, d_key, s_key, f))
                    if val_uno is None:
                        val_uno = ipcw_uno_lookup.get((m_folder.lower(), d_key.lower(), s_key.lower(), f))
                    if val_uno is not None:
                        if (
                            m_key in test_folds
                            and d_key in test_folds[m_key]
                            and s_key in test_folds[m_key][d_key]
                            and f in test_folds[m_key][d_key][s_key]
                        ):
                            metrics_dict = test_folds[m_key][d_key][s_key][f].setdefault("metrics", {})
                            metrics_dict["uno_c"] = val_uno

    method_to_folder = {m[0]: os.path.basename(m[2]) for m in all_methods}

    def get_mean_std_val(m_key, d_key, s_key, metric):
        if metric in ("td_auc_mean", "td_auc"):
            m_folder = method_to_folder.get(m_key, m_key)
            vals = []
            for f in range(1, 6):
                val = ipcw_lookup.get((m_folder, d_key, s_key, f))
                if val is None:
                    val = ipcw_lookup.get((m_folder.lower(), d_key.lower(), s_key.lower(), f))
                if val is None:
                    sys.exit(
                        f"Error: fold {m_folder}_{d_key}_{s_key}_fold{f} needed by table has no auc_ipcw value in ipcw_metrics.json"
                    )
                vals.append(float(val))
            if len(vals) == 5:
                return float(np.mean(vals)), float(np.std(vals, ddof=1))

        if metric == "uno_c":
            m_folder = method_to_folder.get(m_key, m_key)
            vals = []
            for f in range(1, 6):
                val = ipcw_uno_lookup.get((m_folder, d_key, s_key, f))
                if val is None:
                    val = ipcw_uno_lookup.get((m_folder.lower(), d_key.lower(), s_key.lower(), f))
                if val is None:
                    sys.exit(
                        f"Error: fold {m_folder}_{d_key}_{s_key}_fold{f} needed by table has no uno_c value in ipcw_metrics.json"
                    )
                vals.append(float(val))
            if len(vals) == 5:
                return float(np.mean(vals)), float(np.std(vals, ddof=1))

        if m_key in test_folds and d_key in test_folds[m_key] and s_key in test_folds[m_key][d_key]:
            vals = [
                test_folds[m_key][d_key][s_key][f]["metrics"][metric]
                for f in range(1, 6)
                if metric in test_folds[m_key][d_key][s_key].get(f, {}).get("metrics", {})
            ]
            if len(vals) == 5:
                return float(np.mean(vals)), float(np.std(vals, ddof=1))
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
        notes=None,
        placement="[h]",
        blocks=None,
    ):
        cols = []
        for d_key, _ in datasets:
            for s_key in scenarios:
                cols.append((d_key, s_key))

        lines = [
            f"\\begin{{table}}{placement}",
            "\\centering",
            f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
            "\\scriptsize",
            f"\\caption{{{metric_name}{extra_caption}}}\\label{{tab:res_{label_suffix}}}",
            "\\begin{tabular}{lccccccccc}",
            "\\toprule",
            " & \\multicolumn{3}{c}{METABRIC} & \\multicolumn{3}{c}{MIMIC-IV} & \\multicolumn{3}{c}{eICU} \\\\",
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
            "Method & Light & Moderate & Severe & Light & Moderate & Severe & Light & Moderate & Severe \\\\",
            "\\midrule",
        ]

        if blocks is None:
            blocks = [(None, metric_key)]

        for b_idx, (block_title, b_metric_key) in enumerate(blocks):
            if b_idx > 0:
                lines.append("\\midrule")
            if block_title:
                lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{block_title}}}}} \\\\")

            best_per_col = {}
            for d_key, s_key in cols:
                vals = []
                for m in method_subset:
                    m_key = m[0]
                    m_val, _ = get_mean_std_val(m_key, d_key, s_key, b_metric_key)
                    if m_val is not None:
                        vals.append(round(m_val, decimals))
                if vals and bold_criterion is not None:
                    if bold_criterion == "min":
                        best_per_col[(d_key, s_key)] = min(vals)
                    elif bold_criterion == "max":
                        best_per_col[(d_key, s_key)] = max(vals)
                else:
                    best_per_col[(d_key, s_key)] = None

            for m in method_subset:
                m_key = m[0]
                m_lbl = m[1]
                row_cells = [m_lbl]
                for d_key, s_key in cols:
                    m_val, s_val = get_mean_std_val(m_key, d_key, s_key, b_metric_key)
                    is_best = False
                    best_val = best_per_col.get((d_key, s_key))
                    if best_val is not None and m_val is not None:
                        if abs(round(m_val, decimals) - best_val) < 1e-12:
                            is_best = True

                    count_lt = None
                    if append_lt_005:
                        count_lt = 0
                        for f in range(1, 6):
                            f_m = test_folds[m_key][d_key][s_key][f].get("metrics", {})
                            val = f_m.get(b_metric_key)
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
        ])
        if notes:
            lines.extend([
                "\\begin{minipage}{\\linewidth}",
                "\\footnotesize",
                "\\vspace{1mm}",
                notes,
                "\\end{minipage}",
            ])
        lines.append("\\end{table}")
        return "\n".join(lines)

    t1_rmse = build_transposed_comparison_table(
        "rmse_std",
        "Root mean square error (RMSE).",
        "rmse",
        imputing_methods,
        bold_criterion="min",
        notes="Standardized scale, computed on the artificially masked cells only. In each cell, the upper value is the mean and the lower value in parentheses is the standard deviation over five test folds. Bold: lowest mean in each column. MC Dropout Cox and DeepHit are not listed because they used the MICE imputation.",
    )
    t2_cindex = build_transposed_comparison_table(
        "cindex",
        "Concordance index: Harrell's C and Uno's C.",
        "cindex",
        methods,
        bold_criterion="max",
        tabcolsep="1.4pt",
        notes="Harrell's C and Uno's C on the test partition. In each cell, the upper value is the mean and the lower value in parentheses is the standard deviation over five test folds. Bold: highest mean in each column within each block. Uno's C was truncated at the end of the evaluation grid (Appendix~\\ref{app:metrics}). In MIMIC-IV and eICU, all censoring occurred at day 28, after the end of the evaluation grid, so the censoring weights equaled 1 and the two blocks agree to three decimals.",
        blocks=[("Harrell's C", "cindex"), ("Uno's C", "uno_c")],
    )
    t3_ibs = build_transposed_comparison_table(
        "ibs",
        "Integrated Brier score (IBS).",
        "ibs",
        methods,
        bold_criterion="min",
        tabcolsep="1.4pt",
        notes="Integrated Brier score with inverse probability of censoring weights; lower is better. In each cell, the upper value is the mean and the lower value in parentheses is the standard deviation over five test folds. Bold: lowest mean in each column.",
    )
    t4_tdauc = build_transposed_comparison_table(
        "td_auc_mean",
        "Time-dependent AUC with inverse probability of censoring weights.",
        "td_auc",
        methods,
        bold_criterion="max",
        tabcolsep="1.4pt",
        notes="Time-dependent AUC, averaged over the evaluation time points (Section~\\ref{sec:evalmetrics}). In each cell, the upper value is the mean and the lower value in parentheses is the standard deviation over five test folds. Bold: highest mean in each column.",
    )
    t5_dcal = build_transposed_comparison_table(
        "dcal_pval",
        "D-calibration $p$-value.",
        "dcal",
        methods,
        bold_criterion=None,
        extra_caption="",
        append_lt_005=True,
        font_size="\\scriptsize",
        notes="In each cell, the upper value is the mean p-value, the middle value in parentheses is the standard deviation over five test folds, and the lower value in brackets is the number of folds with $p < 0.05$. $p < 0.05$ indicates miscalibration of the predicted survival distribution.",
    )

    t6_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Imputation uncertainty calibration of B-SVAE and of MICE with posterior sampling.}\\label{tab:res_bsvae_uncertainty}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{B-SVAE} & \\multicolumn{3}{c}{MICE, posterior sampling} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
        "Cohort & Severity & PICP & MPIW & ECE & PICP & MPIW & ECE \\\\",
        "\\midrule",
    ]
    for d_key, d_lbl in datasets:
        for s_idx, s_key in enumerate(scenarios):
            c_label = d_lbl if s_idx == 0 else ""
            s_label = s_key.capitalize()
            picp_str = fmt_cell("bsvae", d_key, s_key, "picp", is_bold=False)
            mpiw_str = fmt_cell("bsvae", d_key, s_key, "mpiw", is_bold=False)
            ece_str = fmt_cell("bsvae", d_key, s_key, "ece", is_bold=False)
            mice_picp = fmt_cell("mice_mi", d_key, s_key, "picp", is_bold=False)
            mice_mpiw = fmt_cell("mice_mi", d_key, s_key, "mpiw", is_bold=False)
            mice_ece = fmt_cell("mice_mi", d_key, s_key, "ece", is_bold=False)
            t6_lines.append(f"{c_label} & {s_label} & {picp_str} & {mpiw_str} & {ece_str} & {mice_picp} & {mice_mpiw} & {mice_ece} \\\\")
        if d_key != datasets[-1][0]:
            t6_lines.append("\\midrule")
    t6_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\begin{minipage}{\\linewidth}",
        "\\footnotesize",
        "\\vspace{1mm}",
        "PICP, prediction interval coverage probability, and MPIW, mean prediction interval width, of the 95\\% predictive intervals; ECE, expected calibration error. All metrics were computed on the masked cells; MPIW on the standardized scale. In each cell, the upper value is the mean and the lower value in parentheses is the standard deviation over five test folds. MICE, posterior sampling: 20 imputations per fold with the selected MICE configuration; the intervals use the mean and variance of the 20 imputed values.",
        "\\end{minipage}",
        "\\end{table}",
    ])
    t6_uncertainty = "\n".join(t6_lines)

    t7_mae = build_transposed_comparison_table(
        "mae_std",
        "Mean absolute error (MAE).",
        "mae",
        imputing_methods,
        bold_criterion="min",
        notes="Standardized scale, computed on the artificially masked cells only. In each cell, the upper value is the mean and the lower value in parentheses is the standard deviation over five test folds. Bold: lowest mean in each column. MC Dropout Cox and DeepHit are not listed because they used the MICE imputation.",
        placement="[!htbp]",
    )

    iwae_methods = [
        ("bsvae", "B-SVAE", "Results/final/runs_v2/bsvae"),
        ("mnar_svae", "Ablation", "Results/final/runs_v2/mnar_svae"),
    ]
    t8_iwae = build_transposed_comparison_table(
        "iwae_mean",
        "Importance-weighted autoencoder bound (IWAE).",
        "iwae",
        iwae_methods,
        bold_criterion=None,
        extra_caption="",
        decimals=1,
        notes="Values are not comparable across models with different observation likelihoods. In each cell, the upper value is the mean and the lower value in parentheses is the standard deviation over five test folds.",
        placement="[!htbp]",
    )

    comp_cindex_methods = [m for m in methods if m[0] != "bsvae"]
    comp_rmse_methods = [m for m in imputing_methods if m[0] != "bsvae"]

    cols = []
    for d_key, _ in datasets:
        for s_key in scenarios:
            cols.append((d_key, s_key))

    t9_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\scriptsize",
        "\\caption{Fold-wise wins of B-SVAE.}\\label{tab:res_wins}",
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{METABRIC} & \\multicolumn{3}{c}{MIMIC-IV} & \\multicolumn{3}{c}{eICU} & \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "Method & Light & Moderate & Severe & Light & Moderate & Severe & Light & Moderate & Severe & Total \\\\",
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
        "\\begin{minipage}{\\linewidth}",
        "\\footnotesize",
        "\\vspace{1mm}",
        "Number of the five test folds in which B-SVAE had a higher C-index or a lower RMSE than each method. Ties count as no win. MC Dropout Cox and DeepHit are not listed for RMSE because they used the MICE imputation.",
        "\\end{minipage}",
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
    ]

    out_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n\n".join(tables) + "\n")

    print(f"Written {len(tables)} tables to {out_path}")

    # Build difference tables T10 (C-index) and T11 (RMSE)
    def fmt_math(val, decimals=3):
        s = f"{val:.{decimals}f}"
        if s.startswith("-"):
            return f"-{s[1:]}"
        return s

    def build_difference_table(metric_key, metric_title, label_suffix, method_subset, caption_favors=None, is_cindex=True, notes=None):
        cols = []
        for d_key, _ in datasets:
            for s_key in scenarios:
                cols.append((d_key, s_key))

        lines = [
            "\\begin{table}[!htbp]",
            "\\centering",
            "\\setlength{\\tabcolsep}{1.2pt}",
            "\\scriptsize",
            f"\\caption{{{metric_title}.}}\\label{{tab:{label_suffix}}}",
            "\\begin{tabular}{@{}lccccccccc@{}}",
            "\\toprule",
            " & \\multicolumn{3}{c}{METABRIC} & \\multicolumn{3}{c}{MIMIC-IV} & \\multicolumn{3}{c}{eICU} \\\\",
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
            "Method & Light & Moderate & Severe & Light & Moderate & Severe & Light & Moderate & Severe \\\\",
            "\\midrule",
        ]

        excl_zero_count = 0
        total_count = 0

        for m in method_subset:
            m_key = m[0]
            m_lbl = m[1]
            if m_key == "mc_dropout_cox":
                m_lbl = "\\begin{tabular}[c]{@{}l@{}}MC Dropout\\\\Cox\\end{tabular}"
            elif m_key == "bsvae_nolcal":
                m_lbl = "\\begin{tabular}[c]{@{}l@{}}B-SVAE\\\\($\\lambda = 0$)\\end{tabular}"
            row_cells = [m_lbl]
            for d_key, s_key in cols:
                total_count += 1
                diffs = []
                for f in range(1, 6):
                    b_val = test_folds["bsvae"][d_key][s_key][f].get("metrics", {}).get(metric_key)
                    m_val = test_folds[m_key][d_key][s_key][f].get("metrics", {}).get(metric_key)
                    if b_val is not None and m_val is not None:
                        diffs.append(float(b_val) - float(m_val))

                if len(diffs) == 5:
                    d_bar = float(np.mean(diffs))
                    s_d = float(np.std(diffs, ddof=1))
                    h = 2.7764 * s_d * np.sqrt(1.0 / 5.0 + 0.20 / 0.64)
                    lo = d_bar - h
                    hi = d_bar + h

                    excludes_zero = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
                    if excludes_zero:
                        excl_zero_count += 1

                    star = "\\textsuperscript{*}" if excludes_zero else ""
                    d_str = f"${fmt_math(d_bar)}${star}"
                    lo_str = f"${fmt_math(lo)}$"
                    hi_str = f"${fmt_math(hi)}$"
                    cell_str = f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{d_str}\\\\{{[{lo_str},}}\\\\{{{hi_str}]}}\\end{{tabular}}"
                else:
                    cell_str = "--"

                row_cells.append(cell_str)
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
        ])
        if notes:
            lines.extend([
                "\\begin{minipage}{\\linewidth}",
                "\\footnotesize",
                "\\vspace{1mm}",
                notes,
                "\\end{minipage}",
            ])
        lines.append("\\end{table}")
        return "\n".join(lines), excl_zero_count, total_count

    diff_cindex_methods = comp_cindex_methods + [
        ("bsvae_nolcal", "B-SVAE ($\\lambda = 0$)", "Results/final/runs_v2/bsvae_nolcal")
    ]
    t10_cindex, t10_excl, t10_tot = build_difference_table(
        "cindex",
        "Difference in C-index, B-SVAE minus each method, with 95\\% confidence intervals",
        "diff_cindex",
        diff_cindex_methods,
        caption_favors=None,
        is_cindex=True,
        notes=None,
    )

    diff_rmse_methods = comp_rmse_methods + [
        ("bsvae_nolcal", "B-SVAE ($\\lambda = 0$)", "Results/final/runs_v2/bsvae_nolcal")
    ]
    combined_diff_notes = (
        "In each cell, the upper value is the mean difference and the lower values in brackets "
        "denote the 95\\% confidence interval $[lo, hi]$. For C-index, positive values favor B-SVAE; "
        "for RMSE, negative values favor B-SVAE. Intervals use the corrected resampled $t$ statistic "
        "and are not adjusted for multiple comparisons. * indicates the confidence interval excludes 0. "
        "B-SVAE ($\\lambda = 0$): B-SVAE without the calibration regularizer."
    )
    t11_rmse, t11_excl, t11_tot = build_difference_table(
        "rmse_std",
        "Difference in imputation RMSE, B-SVAE minus each method, with 95\\% confidence intervals",
        "diff_rmse",
        diff_rmse_methods,
        caption_favors=None,
        is_cindex=False,
        notes=combined_diff_notes,
    )

    t7_mae_body = t7_mae[: t7_mae.rfind("\\end{table}")].rstrip()
    t7_mae_body = t7_mae_body.replace(
        "\\begin{table}[!htbp]\n\\centering\n",
        "\\begin{table}[!htbp]\n\\centering\n\\renewcommand{\\arraystretch}{0.92}\n",
        1,
    )
    t8_iwae_body = t8_iwae[t8_iwae.find("\n") + 1 : t8_iwae.rfind("\\end{table}")].rstrip()
    t9_wins_body = t9_wins[t9_wins.find("\n") + 1 :]
    t7_t8_t9_combined = (
        t7_mae_body
        + "\n\n\\vspace{2.5mm}\n\n"
        + t8_iwae_body
        + "\n\n\\vspace{2.5mm}\n\n"
        + t9_wins_body
    )

    t10_cindex_body = t10_cindex[: t10_cindex.rfind("\\end{table}")].rstrip()
    t10_cindex_body = t10_cindex_body.replace(
        "\\begin{table}[!htbp]\n\\centering\n",
        "\\begin{table}[!htbp]\n\\centering\n\\renewcommand{\\arraystretch}{0.75}\n",
        1,
    )
    t11_rmse_body = t11_rmse[t11_rmse.find("\n") + 1 :]
    t10_t11_combined = t10_cindex_body + "\n\n\\vspace{1mm}\n\n" + t11_rmse_body

    appendix_tables = [t7_t8_t9_combined, t10_t11_combined]
    appendix_out_path = os.path.join(project_root, "sn-article-template/results_tables_appendix.tex")
    os.makedirs(os.path.dirname(appendix_out_path), exist_ok=True)
    with open(appendix_out_path, "w", encoding="utf-8") as fp:
        fp.write("\n\n".join(appendix_tables) + "\n")

    print(f"Written {len(appendix_tables)} tables to {appendix_out_path}")
    print(f"Table 10 (C-index differences): {t10_excl} of {t10_tot} intervals exclude 0; Table 11 (RMSE differences): {t11_excl} of {t11_tot} intervals exclude 0.")

    make_dcal_figure(project_root)


def make_dcal_figure(project_root=None):
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    dcal_methods = [
        ("bsvae", "B-SVAE", "#2a78d6", "o", -0.15),
        ("bsvae_nolcal", r"B-SVAE ($\lambda = 0$)", "#eb6834", "s", 0.0),
        ("mnar_svae", "Ablation", "#1baf7a", "^", 0.15),
    ]

    cohorts = [
        ("metabric", "METABRIC"),
        ("mimic", "MIMIC-IV"),
        ("eicu", "eICU"),
    ]

    scenarios = ["light", "moderate", "severe"]
    bins_x = np.arange(1, 11)
    tick_labels = [f"0.{i}" for i in range(10)]

    plt.rcParams.update({
        "font.size": 8,
        "font.family": "sans-serif",
        "mathtext.fontset": "stix",
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2), sharey=False)
    fig.subplots_adjust(top=0.79, bottom=0.25, left=0.10, right=0.98, wspace=0.28)

    for ax_idx, (cohort_key, cohort_name) in enumerate(cohorts):
        ax = axes[ax_idx]
        ax.set_title(cohort_name, fontsize=8, pad=5)
        ax.axhline(0.0, color="#888888", linestyle="-", linewidth=0.8, alpha=0.7)

        for m_key, m_label, color, marker, offset in dcal_methods:
            dev_list = []
            for sc in scenarios:
                for f in range(1, 6):
                    npz_path = os.path.join(
                        project_root,
                        "Results",
                        "final",
                        "runs_v2",
                        m_key,
                        f"{cohort_key}_{sc}_fold{f}_test.npz",
                    )
                    if os.path.exists(npz_path):
                        with np.load(npz_path) as data:
                            b = data["dcal_bins"].astype(float)
                            s = b.sum()
                            if s > 0:
                                shares = b / s
                                dev = (shares - 0.10) * 100.0
                                dev_list.append(dev)
                            else:
                                dev_list.append(np.zeros_like(b))

            if dev_list:
                dev_arr = np.array(dev_list)
                mean_dev = np.mean(dev_arr, axis=0)
                std_dev = np.std(dev_arr, axis=0)
                ax.errorbar(
                    bins_x + offset,
                    mean_dev,
                    yerr=std_dev,
                    color=color,
                    marker=marker,
                    linestyle="none",
                    elinewidth=0.6,
                    capsize=0,
                    markersize=3.2,
                    alpha=0.9,
                    label=m_label,
                )

        ax.set_xlim(0.3, 10.7)
        ax.set_xticks(bins_x)
        ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=35, ha="right")

        if ax_idx == 0:
            ax.set_ylabel("Deviation from expected share (pp)", fontsize=8)

        ax.grid(True, axis="y", color="#e8e8e8", linestyle="-", linewidth=0.6, alpha=0.8)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("Predicted survival at event time (bin)", fontsize=8, y=0.03)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.6,
        handletextpad=0.4,
    )

    out_path = os.path.join(project_root, "sn-article-template", "fig_dcal_bins.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved D-calibration figure to {out_path}")

    export_selected_hyperparameters(project_root)


def export_selected_hyperparameters(project_root=None):
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    method_configs = [
        ("bsvae", "B-SVAE", "Results/final/runs_v2/bsvae"),
        ("mnar_svae", "MNAR-SVAE ablation", "Results/final/runs_v2/mnar_svae"),
        ("mice", "MICE + Cox", "Results/final/runs_v2/mice"),
        ("miwae", "MIWAE + Cox", "Results/final/runs_v2/miwae"),
        ("bsvae_nolcal", "B-SVAE (lambda = 0)", "Results/final/runs_v2/bsvae_nolcal"),
        ("missforest", "missForest + Cox", "Results/final/runs_v2_tesla/missforest"),
        ("gain", "GAIN + Cox", "Results/final/runs_v2_tesla/gain"),
        ("mc_dropout_cox", "MC Dropout Cox", "Results/final/runs_v2_tesla/mc_dropout_cox"),
        ("deephit", "DeepHit", "Results/final/runs_v2_tesla/deephit"),
    ]

    cohort_map = {
        "metabric": "METABRIC",
        "mimic": "MIMIC-IV",
        "eicu": "eICU",
    }
    severities = ["light", "moderate", "severe"]

    def clean_param_name(m_name, p_name):
        if p_name in ("latent", "latent_dim"):
            return "latent dimension"
        if p_name == "hidden":
            return "hidden units" if m_name == "MIWAE + Cox" else "hidden layers"
        if p_name in ("dropout", "dropout_prob"):
            return "dropout"
        if p_name == "learning_rate":
            return "learning rate"
        if p_name == "l2":
            return "L2 penalty"
        if p_name == "max_iter":
            return "iterations"
        if p_name == "n_nearest_features":
            return "nearest features"
        if p_name == "hint_rate":
            return "hint rate"
        return p_name

    def clean_param_value(p_val):
        s_val = str(p_val)
        if s_val == "half":
            return "floor(p/2)"
        if "_" in s_val and all(part.isdigit() for part in s_val.split("_")):
            return f"[{','.join(s_val.split('_'))}]"
        return p_val

    rows = []
    for m_key, m_name, m_rel in method_configs:
        m_dir = os.path.join(project_root, m_rel)
        for c_key in ["metabric", "mimic", "eicu"]:
            c_name = cohort_map[c_key]
            for s in severities:
                for f in range(1, 6):
                    fname = f"{c_key}_{s}_fold{f}_selected.json"
                    fpath = os.path.join(m_dir, fname)
                    if os.path.isfile(fpath):
                        with open(fpath, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                        trial = data.get("selected_trial")
                        params = data.get("params", {})
                        for p_name, p_val in params.items():
                            p_clean = clean_param_name(m_name, p_name)
                            v_clean = clean_param_value(p_val)
                            rows.append([m_name, c_name, s, f, trial, p_clean, v_clean])

        syn_fname = "synthetic_moderate_fold1_selected.json"
        syn_fpath = os.path.join(m_dir, syn_fname)
        if os.path.isfile(syn_fpath):
            with open(syn_fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            trial = data.get("selected_trial")
            params = data.get("params", {})
            for p_name, p_val in params.items():
                p_clean = clean_param_name(m_name, p_name)
                v_clean = clean_param_value(p_val)
                rows.append([m_name, "Synthetic", "moderate", 1, trial, p_clean, v_clean])

    out_csv = os.path.join(
        project_root, "sn-article-template", "additional_file_1_selected_hyperparameters.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["method", "cohort", "severity", "fold", "selected_trial", "parameter", "value"]
        )
        writer.writerows(rows)
    print(f"Saved selected hyperparameters to {out_csv}")


if __name__ == "__main__":
    main()

