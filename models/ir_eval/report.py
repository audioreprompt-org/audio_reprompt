"""Markdown / CSV / JSON report (WS-C).

CONTRACT:

    generate_report(results: dict, output_dir: str) -> str   # returns report path

``results`` schema (built by cli.py):
    {
      "meta": {"seed": int, "timestamp": str, "git_commit": str | None,
               "embedding_model": str, "queries_path": str},
      "table_stats": diagnostics.table_stats(...),
      "dimension_coverage": diagnostics.dimension_coverage(...),
      "similarity_distribution": {table: diagnostics.similarity_distribution(...)},
      "query_spread": {table: diagnostics.query_spread(...)},
      "hubness": {table: {"leave_one_out": diagnostics.hubness(...),
                          "queries": diagnostics.hubness(..., query_X=...) | None}},
      "anisotropy": {table: diagnostics.anisotropy(...)},
      "ann_vs_exact": {table: retrieval.ann_vs_exact(...)} | {},
      "crossmodal": {"cut_true": retrieval.evaluate_crossmodal(..., cut_results=True),
                     "cut_false": retrieval.evaluate_crossmodal(..., cut_results=False)} | {},
      "audio": retrieval.evaluate_audio(...) | {},
    }

Outputs in output_dir: ir_eval_report.md, results.json (full, NaN -> null),
ir_diagnostics.csv (flat), crossmodal_per_query_{cut_true,cut_false}.csv,
audio_per_query.csv.

Report rules (review §3):
  - Every retrieval metric shows mean [95% CI] and n, next to the random baseline.
  - The production setting (cut_results=True) is the primary table.
  - Interpretations state only what the numbers support. No "strong
    discriminative power" from std of random pairs, no "isotropy" score, no
    Good/Acceptable label from the agreement diagnostic, no audio quality verdict.
  - Crossmodal verdict only from structural metrics vs baseline: say whether the
    paired CI excludes 0 and give the effect size; mention n = 15 queries.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

import numpy as np


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert NaNs to None (JSON null) and numpy types to Python types."""
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    return obj


def _fmt_val(v: Any, fmt: str = ".4f") -> str:
    """Format numeric values; NaN -> 'n/a'."""
    if v is None:
        return "n/a"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "n/a"
        return f"{v:{fmt}}"
    return str(v)


def _fmt_ci(ci: dict | None, fmt: str = ".3f") -> str:
    """Format a bootstrap CI dict as 'mean [ci_low, ci_high] (n=N)'."""
    if not ci or not isinstance(ci, dict):
        return "n/a"
    m = ci.get("mean")
    lo = ci.get("ci_low")
    hi = ci.get("ci_high")
    n = ci.get("n", 0)
    if m is None or (isinstance(m, (float, np.floating)) and math.isnan(m)):
        return "n/a"
    return f"{m:{fmt}} [{lo:{fmt}}, {hi:{fmt}}] (n={n})"


def _fmt_diff(diff: dict | None, fmt: str = ".3f") -> str:
    """Format paired difference dict as 'mean_diff [ci_low, ci_high]'."""
    if not diff or not isinstance(diff, dict):
        return "n/a"
    d = diff.get("mean_diff")
    lo = diff.get("ci_low")
    hi = diff.get("ci_high")
    if d is None or (isinstance(d, (float, np.floating)) and math.isnan(d)):
        return "n/a"
    return f"{d:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def _build_flat_diagnostics(results: dict) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    for k, v in results.get("meta", {}).items():
        flat[f"meta_{k}"] = v

    for k, v in results.get("table_stats", {}).items():
        flat[f"stats_{k}"] = v

    dc = results.get("dimension_coverage", {})
    if dc:
        flat["coverage_total"] = dc.get("total")
        flat["coverage_gini_coefficient"] = dc.get("gini_coefficient")
        flat["coverage_gini_max"] = dc.get("gini_max")
        flat["coverage_gini_normalized"] = dc.get("gini_normalized")

    for tbl, s in results.get("similarity_distribution", {}).items():
        for k, v in s.items():
            flat[f"sim_dist_{tbl}_{k}"] = v

    for tbl, s in results.get("query_spread", {}).items():
        for k, v in s.items():
            flat[f"query_spread_{tbl}_{k}"] = v

    for tbl, modes in results.get("hubness", {}).items():
        for m_name, m_data in modes.items():
            if m_data and isinstance(m_data, dict):
                for k, v in m_data.items():
                    if not isinstance(v, (list, dict)):
                        flat[f"hubness_{tbl}_{m_name}_{k}"] = v

    for tbl, a in results.get("anisotropy", {}).items():
        for k, v in a.items():
            if not isinstance(v, (list, dict)):
                flat[f"space_{tbl}_{k}"] = v

    for tbl, ann in results.get("ann_vs_exact", {}).items():
        for k, v in ann.items():
            if not isinstance(v, (list, dict)):
                flat[f"ann_{tbl}_{k}"] = v

    for cut_key in ("cut_true", "cut_false"):
        cm = results.get("crossmodal", {}).get(cut_key, {})
        if cm:
            flat[f"cm_{cut_key}_n_failed"] = cm.get("n_failed")
            flat[f"cm_{cut_key}_n_no_relevant_in_corpus"] = cm.get("n_no_relevant_in_corpus")
            flat[f"cm_{cut_key}_n_duplicates_removed"] = cm.get("n_duplicates_removed")
            for m_name, ci in cm.get("system", {}).items():
                flat[f"cm_{cut_key}_sys_{m_name}_mean"] = ci.get("mean")
                flat[f"cm_{cut_key}_sys_{m_name}_ci_low"] = ci.get("ci_low")
                flat[f"cm_{cut_key}_sys_{m_name}_ci_high"] = ci.get("ci_high")
            for m_name, ci in cm.get("baseline", {}).items():
                flat[f"cm_{cut_key}_base_{m_name}_mean"] = ci.get("mean")
            for m_name, diff in cm.get("system_vs_baseline", {}).items():
                flat[f"cm_{cut_key}_diff_{m_name}_mean"] = diff.get("mean_diff")
                flat[f"cm_{cut_key}_diff_{m_name}_p_val"] = diff.get("p_value")
            for m_name, ci in cm.get("agreement_diagnostic", {}).items():
                flat[f"cm_{cut_key}_{m_name}_mean"] = ci.get("mean")

    au = results.get("audio", {})
    if au:
        flat["audio_n_failed"] = au.get("n_failed")
        flat["audio_n_collapsed"] = au.get("n_collapsed")
        for k in ("top1_sim", "avg_sim", "sim_spread"):
            if k in au and isinstance(au[k], dict):
                flat[f"audio_{k}_mean"] = au[k].get("mean")
                flat[f"audio_{k}_ci_low"] = au[k].get("ci_low")
                flat[f"audio_{k}_ci_high"] = au[k].get("ci_high")
        for k, ci in au.get("agreement_diagnostic", {}).items():
            if isinstance(ci, dict):
                flat[f"audio_{k}_mean"] = ci.get("mean")

    return flat


def _write_per_query_csv(per_query: list[dict], path: str) -> None:
    if not per_query:
        with open(path, "w", encoding="utf-8") as f:
            f.write("query_id,query_text,n_results\n")
        return
    rows = [{k: v for k, v in q.items() if k != "results_detail"} for q in per_query]
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(results: dict, output_dir: str) -> str:
    """Generate Markdown report, JSON results, and CSV tables.

    Returns the absolute path to the generated ir_eval_report.md.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. results.json (full, NaN -> null)
    cleaned_results = _clean_for_json(results)
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, indent=2)

    # 2. ir_diagnostics.csv (flat)
    flat_diag = _build_flat_diagnostics(results)
    diag_csv_path = os.path.join(output_dir, "ir_diagnostics.csv")
    with open(diag_csv_path, "w", encoding="utf-8", newline="") as f:
        if flat_diag:
            writer = csv.DictWriter(f, fieldnames=list(flat_diag.keys()))
            writer.writeheader()
            writer.writerow(flat_diag)
        else:
            f.write("metric,value\n")

    # 3. crossmodal_per_query_{cut_true,cut_false}.csv
    cm = results.get("crossmodal", {})
    cm_true_pq = cm.get("cut_true", {}).get("per_query", []) if cm else []
    cm_false_pq = cm.get("cut_false", {}).get("per_query", []) if cm else []
    _write_per_query_csv(cm_true_pq, os.path.join(output_dir, "crossmodal_per_query_cut_true.csv"))
    _write_per_query_csv(cm_false_pq, os.path.join(output_dir, "crossmodal_per_query_cut_false.csv"))

    # 4. audio_per_query.csv
    au = results.get("audio", {})
    au_pq = au.get("per_query", []) if au else []
    _write_per_query_csv(au_pq, os.path.join(output_dir, "audio_per_query.csv"))

    # 5. ir_eval_report.md
    meta = results.get("meta", {})
    table_stats = results.get("table_stats", {})
    dim_cov = results.get("dimension_coverage", {})
    sim_dist = results.get("similarity_distribution", {})
    query_spread = results.get("query_spread", {})
    hubness = results.get("hubness", {})
    aniso = results.get("anisotropy", {})
    ann = results.get("ann_vs_exact", {})

    lines: list[str] = [
        "# IR Quality Evaluation Report",
        "",
        f"**Embedding Model**: `{meta.get('embedding_model', 'n/a')}`",
        f"**Distance Metric**: Cosine (pgvector `<=>` operator)",
        f"**Seed**: `{meta.get('seed', 'n/a')}`",
        f"**Timestamp**: `{meta.get('timestamp', 'n/a')}`",
        f"**Git Commit**: `{meta.get('git_commit') or 'n/a'}`",
        "",
        "---",
        "",
        "## Part B: Database Diagnostics",
        "",
        "### D1. Table Statistics",
        "",
        "| Table | Total Rows | Distinct Text Keys |",
        "|---|---|---|",
        f"| `audio_descriptors` | {table_stats.get('audio_descriptors_rows', 0):,} | {table_stats.get('audio_distinct_captions', 0):,} captions |",
        f"| `crossmodal_food_embeddings` | {table_stats.get('crossmodal_food_rows', 0):,} | {table_stats.get('food_distinct_dim_descriptor_pairs', 0):,} distinct (dimension, descriptor) pairs |",
        "",
        f"- Distinct food dimensions: **{table_stats.get('food_distinct_dimensions', 0)}**",
        f"- Distinct food descriptors: **{table_stats.get('food_distinct_descriptors', 0):,}**",
        f"- Distinct food items: **{table_stats.get('food_distinct_items', 0):,}**",
        "",
        "### D2. Pairwise Cosine Similarity Distribution",
        "",
        "| Table | Mean | Std | Min | Median | Max | P10 | P90 | Near-Duplicates (>=0.9999) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for tbl, data in sim_dist.items():
        lines.append(
            f"| `{tbl}` | {_fmt_val(data.get('mean'))} | {_fmt_val(data.get('std'))} | "
            f"{_fmt_val(data.get('min'))} | {_fmt_val(data.get('median'))} | "
            f"{_fmt_val(data.get('max'))} | {_fmt_val(data.get('p10'))} | "
            f"{_fmt_val(data.get('p90'))} | {data.get('n_near_duplicate_pairs', 0):,} |"
        )

    lines.extend([
        "",
        "> **Note on Pairwise Distribution**: Mean pairwise cosine measures directional concentration in the vector space.",
        "> Near-duplicate pairs represent identical or virtually identical text encodings in the dataset.",
        "",
        "### D3. Query Similarity Drop-off (Top-1 vs Top-K Gap)",
        "",
        "| Table | Avg Top-1 Sim | Gap (1 vs 5) | Gap (1 vs 10) | Gap (1 vs 20) | Queries w/ Near-Duplicate |",
        "|---|---|---|---|---|---|",
    ])

    for tbl, data in query_spread.items():
        g5 = data.get("avg_top1_vs_top5_gap")
        g10 = data.get("avg_top1_vs_top10_gap")
        g20 = data.get("avg_top1_vs_top20_gap")
        nd = data.get("n_queries_with_near_duplicate", 0)
        lines.append(
            f"| `{tbl}` | {_fmt_val(data.get('avg_top1_sim'))} | "
            f"{_fmt_val(g5)} | {_fmt_val(g10)} | {_fmt_val(g20)} | {nd} |"
        )

    lines.extend([
        "",
        "> **Note on Top-1 vs Top-K Spread**: Self-matches are excluded via leave-one-out masking.",
        "> Queries with an exact or near duplicate elsewhere in the corpus achieve top-1 similarity near 1.0 due to duplicate rows.",
        "",
        "### D4. Dimension Coverage (Crossmodal)",
        "",
        "| Dimension | Count | Proportion |",
        "|---|---|---|",
    ])

    for dim, cnt in dim_cov.get("counts", {}).items():
        prop = dim_cov.get("proportions", {}).get(dim, 0.0)
        lines.append(f"| `{dim}` | {cnt:,} | {prop:.2%} |")

    g_coeff = dim_cov.get("gini_coefficient")
    g_max = dim_cov.get("gini_max")
    g_norm = dim_cov.get("gini_normalized")
    lines.extend([
        "",
        f"- Gini coefficient: **{_fmt_val(g_coeff)}** (theoretical max for {len(dim_cov.get('counts', {}))} categories: **{_fmt_val(g_max)}**, normalized: **{_fmt_val(g_norm)}**)",
        "",
        "### D5. Nearest-Neighbor Hubness",
        "",
        "| Table | Mode | Points | Queries | k | Skewness (all points) | Antihubs (N_k=0) | Max Occurrence | Mean Occurrence |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for tbl, modes in hubness.items():
        for m_name, data in modes.items():
            if not data or not isinstance(data, dict):
                continue
            lines.append(
                f"| `{tbl}` | `{m_name}` | {data.get('n_points', 0):,} | {data.get('n_queries', 0):,} | "
                f"{data.get('k', 0)} | {_fmt_val(data.get('skewness'), '.3f')} | "
                f"{data.get('pct_antihubs', 0.0):.2%} | {data.get('max_occurrence', 0):,} | "
                f"{_fmt_val(data.get('mean_occurrence'), '.2f')} |"
            )

    lines.extend([
        "",
        "> **Note on Hubness**: Skewness and antihubs are computed over all N points in the corpus (including zeros).",
        "",
        "---",
        "",
        "## Part C: Vector Space Utilization and Dimensionality",
        "",
        "| Table | Samples | Total Dim | Eff. Dim (95%) | Eff. Dim (99%) | Part. Ratio (Centered) | Part. Ratio (Uncentered) | Mean Vector Norm | Numerical Rank | Ratio λ_min / λ_max | Marchenko-Pastur Ref |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for tbl, data in aniso.items():
        lines.append(
            f"| `{tbl}` | {data.get('n_samples', 0):,} | {data.get('total_dimensions', 0)} | "
            f"{data.get('effective_dim_95pct', 0)} | {data.get('effective_dim_99pct', 0)} | "
            f"{_fmt_val(data.get('participation_ratio_centered'), '.1f')} | "
            f"{_fmt_val(data.get('participation_ratio_uncentered'), '.1f')} | "
            f"{_fmt_val(data.get('mean_vector_norm'), '.4f')} | {data.get('numerical_rank', 0)} | "
            f"{_fmt_val(data.get('eig_ratio_min_max'), '.6g')} | {_fmt_val(data.get('eig_ratio_mp_reference'), '.6g')} |"
        )

    lines.extend([
        "",
        "> **Interpretation of Vector Space Metrics**:",
        "> - Effective dimensions and participation ratio measure the spread of variance across dimensions.",
        "> - Mean vector norm reflects the common offset across all vectors.",
        "> - The ratio λ_min / λ_max is contextualized by the Marchenko-Pastur reference for uncorrelated Gaussian data.",
        "",
        "---",
        "",
        "## Part D: Approximate vs Exact Index Scan (ANN Check)",
        "",
    ])

    if ann:
        lines.extend([
            "| Table | Vector Indexes | Test Queries | k | Mean Overlap@k | Min Overlap@k |",
            "|---|---|---|---|---|---|",
        ])
        for tbl, data in ann.items():
            idxs = ", ".join(data.get("indexes", [])) or "none"
            lines.append(
                f"| `{tbl}` | `{idxs}` | {data.get('n_queries', 0)} | {data.get('k', 0)} | "
                f"{_fmt_val(data.get('mean_overlap_at_k'), '.2%')} | {_fmt_val(data.get('min_overlap_at_k'), '.2%')} |"
            )
    else:
        lines.append("*ANN check skipped or no data recorded.*")

    lines.extend([
        "",
        "---",
        "",
        "## Part E: Crossmodal Food Descriptor Retrieval",
        "",
    ])

    cm_true = cm.get("cut_true", {}) if cm else {}
    cm_false = cm.get("cut_false", {}) if cm else {}

    if cm_true:
        n_q = cm_true.get("n_queries", 0)
        n_fail = cm_true.get("n_failed", 0)
        n_no_rel = cm_true.get("n_no_relevant_in_corpus", 0)
        n_dups = cm_true.get("n_duplicates_removed", 0)

        lines.extend([
            "### Primary Evaluation: Production Setting (`cut_results=True`)",
            "",
            f"- Total queries: **{n_q}** | Failed queries (0 results): **{n_fail}** | No relevant item in corpus: **{n_no_rel}** | Duplicates removed: **{n_dups}**",
            "",
            "| Metric | System (cut_results=True) | Random Baseline | Difference (System - Baseline) | p-value |",
            "|---|---|---|---|---|",
        ])

        metrics_order = [
            ("precision_at_5", "Precision@5"),
            ("precision_at_10", "Precision@10"),
            ("ndcg_at_5", "nDCG@5"),
            ("ndcg_at_10", "nDCG@10"),
            ("recall_at_5", "Recall@5"),
            ("recall_at_10", "Recall@10"),
            ("mrr", "MRR"),
            ("map_at_10", "MAP@10"),
        ]

        sys_agg = cm_true.get("system", {})
        base_agg = cm_true.get("baseline", {})
        diff_agg = cm_true.get("system_vs_baseline", {})

        for m_key, m_label in metrics_order:
            sys_ci_str = _fmt_ci(sys_agg.get(m_key))
            base_ci_str = _fmt_ci(base_agg.get(m_key))
            d_dict = diff_agg.get(m_key)
            diff_str = _fmt_diff(d_dict)
            p_val = d_dict.get("p_value") if d_dict else None
            p_str = _fmt_val(p_val, ".4f")
            lines.append(f"| {m_label} | {sys_ci_str} | {base_ci_str} | {diff_str} | {p_str} |")

        # Agreement diagnostic
        agr = cm_true.get("agreement_diagnostic", {})
        lines.extend([
            "",
            "#### Retriever-Agreement Diagnostic (`cut_results=True`)",
            "",
            "> **Diagnostic Label**: This metric uses the same MiniLM encoder as the retriever and is circular.",
            "> It measures retriever self-consistency, not ground-truth relevance. It must not be interpreted as a relevance verdict.",
            "",
            f"- Agreement P@5: **{_fmt_ci(agr.get('agreement_p_at_5'))}**",
            f"- Agreement P@10: **{_fmt_ci(agr.get('agreement_p_at_10'))}**",
            "",
        ])

        # Crossmodal verdict based strictly on structural metrics vs baseline
        ndcg10_diff = diff_agg.get("ndcg_at_10", {})
        p10_diff = diff_agg.get("precision_at_10", {})
        mean_d = ndcg10_diff.get("mean_diff")
        ci_lo = ndcg10_diff.get("ci_low")
        ci_hi = ndcg10_diff.get("ci_high")
        p_val = ndcg10_diff.get("p_value")

        lines.extend([
            "### Crossmodal Retrieval Assessment",
            "",
        ])
        if mean_d is not None and not math.isnan(mean_d):
            excludes_zero = (ci_lo > 0) or (ci_hi < 0)
            status_text = "statistically excludes 0" if excludes_zero else "does not exclude 0"
            lines.append(
                f"Based strictly on structural evaluation against the random baseline across n = {n_q} queries:\n"
                f"- For nDCG@10, the paired difference between system and random baseline has mean difference "
                f"**{mean_d:+.3f}** (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}], p = {p_val:.4f}), which {status_text}.\n"
                f"- For Precision@10, the paired difference has mean difference "
                f"**{p10_diff.get('mean_diff', float('nan')):+.3f}** (95% CI: [{p10_diff.get('ci_low', float('nan')):.3f}, {p10_diff.get('ci_high', float('nan')):.3f}], p = {p10_diff.get('p_value', float('nan')):.4f}).\n"
                f"- Note: Findings are established on a small set of n = {n_q} queries."
            )
        else:
            lines.append(f"Insufficient query data for crossmodal verdict (n = {n_q} queries).")

    if cm_false:
        lines.extend([
            "",
            "### Secondary Comparison: Setting (`cut_results=False`)",
            "",
            "| Metric | System (cut_results=False) | Random Baseline | Difference (System - Baseline) | p-value |",
            "|---|---|---|---|---|",
        ])
        sys_agg_f = cm_false.get("system", {})
        base_agg_f = cm_false.get("baseline", {})
        diff_agg_f = cm_false.get("system_vs_baseline", {})
        for m_key, m_label in metrics_order:
            sys_ci_str = _fmt_ci(sys_agg_f.get(m_key))
            base_ci_str = _fmt_ci(base_agg_f.get(m_key))
            d_dict = diff_agg_f.get(m_key)
            diff_str = _fmt_diff(d_dict)
            p_val = d_dict.get("p_value") if d_dict else None
            p_str = _fmt_val(p_val, ".4f")
            lines.append(f"| {m_label} | {sys_ci_str} | {base_ci_str} | {diff_str} | {p_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## Part F: Audio Caption Retrieval",
        "",
        "> **Notice on Audio Caption Retrieval Quality**:",
        "> There is currently no independent relevance judge or rater rubric for audio captions.",
        "> As such, **no quality verdict** is rendered for audio retrieval.",
        "",
    ])

    if au:
        lines.extend([
            f"- Total queries: **{au.get('n_queries', 0)}** | Failed queries: **{au.get('n_failed', 0)}** | Collapsed results (len < k): **{au.get('n_collapsed', 0)}**",
            "",
            "| Metric | Mean [95% CI] (n) |",
            "|---|---|",
            f"| Top-1 Similarity | {_fmt_ci(au.get('top1_sim'))} |",
            f"| Average Similarity | {_fmt_ci(au.get('avg_sim'))} |",
            f"| Similarity Spread (Max - Min) | {_fmt_ci(au.get('sim_spread'))} |",
            "",
            "#### Retriever-Agreement Diagnostic (Audio)",
            "",
            "> **Diagnostic Label**: Self-consistency diagnostic based on the same retriever encoder; does not measure relevance.",
            "",
            f"- Agreement P@5: **{_fmt_ci(au.get('agreement_diagnostic', {}).get('agreement_p_at_5'))}**",
            f"- Agreement P@10: **{_fmt_ci(au.get('agreement_diagnostic', {}).get('agreement_p_at_10'))}**",
        ])
    else:
        lines.append("*Audio evaluation skipped or no data recorded.*")

    lines.append("")

    report_path = os.path.join(output_dir, "ir_eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
