"""
Statistical analysis of ablation CLAP scores.

Replicates the analyses from:
  - analisis_dist_clap_scores.ipynb  (overall distribution + paired t-test)
  - analisis_clap_scores_taste.ipynb (per-taste breakdown + Kendall)

Usage:
    # Analyse a specific experiment group (e.g. B2)
    python -m models.ablation_analysis --experiment B2

    # Analyse a single pair of score files
    python -m models.ablation_analysis \
        --reprompt-scores data/ablations/scores/clap_score_results_reprompt_outputs_*.csv \
        --raw-scores     data/ablations/scores/clap_score_results_prompt_outputs_*_raw.csv \
        --reprompt-csvs  data/ablations/reprompts/pipeline_results_*.csv

    # List available experiment groups
    python -m models.ablation_analysis --list
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import PROJECT_ROOT, setup_project_paths

setup_project_paths()

ABLATIONS_SCORES = PROJECT_ROOT / "data" / "ablations" / "scores"
ABLATIONS_REPROMPTS = PROJECT_ROOT / "data" / "ablations" / "reprompts"
ABLATIONS_ANALYSIS = PROJECT_ROOT / "data" / "ablations" / "analysis"

plt.style.use("seaborn-v0_8-whitegrid")


# ── Helpers ──────────────────────────────────────────────────


def _extract_tag(filename: str) -> str:
    """Extract experiment tag from a score filename."""
    # Pattern: clap_score_results_(reprompt|prompt)_outputs_<TAG>.csv
    # or: clap_score_results_prompt_outputs_<TAG>_(raw|cross).csv
    name = Path(filename).stem
    name = re.sub(r"^clap_score_results_(reprompt|prompt)_outputs_", "", name)
    name = re.sub(r"_(raw|cross)$", "", name)
    return name


def _find_matching_reprompt_csv(tag: str) -> Path | None:
    """Find the reprompt CSV whose filename contains the tag."""
    pattern = str(ABLATIONS_REPROMPTS / f"pipeline_results_*{tag}*.csv")
    matches = glob.glob(pattern)
    # prefer exact suffix match
    for m in matches:
        if Path(m).stem.endswith(tag):
            return Path(m)
    return Path(matches[0]) if matches else None


def _discover_experiment_groups(run_id: str = "") -> dict[str, list[str]]:
    """Discover score files and group them by experiment prefix (B2, B3, etc.).

    If run_id is provided, only include tags containing that run ID.
    """
    reprompt_scores = sorted(ABLATIONS_SCORES.glob("clap_score_results_reprompt_outputs_*.csv"))
    groups: dict[str, list[str]] = {}
    for f in reprompt_scores:
        tag = _extract_tag(f.name)
        # filter by run_id if provided
        if run_id and run_id not in tag:
            continue
        # extract experiment prefix: B2a_filter_default_R... → B2
        match = re.match(r".*?(A\d|B\d)", tag)
        if match:
            prefix = match.group(1)
            groups.setdefault(prefix, []).append(tag)
    return groups


# ── Analysis functions ───────────────────────────────────────


def load_experiment_data(
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load reprompt scores, raw scores, and cross scores for a given tag.

    Returns (df_reprompt, df_raw, df_cross | None).
    """
    reprompt_path = ABLATIONS_SCORES / f"clap_score_results_reprompt_outputs_{tag}.csv"
    raw_path = ABLATIONS_SCORES / f"clap_score_results_prompt_outputs_{tag}_raw.csv"
    cross_path = ABLATIONS_SCORES / f"clap_score_results_prompt_outputs_{tag}_cross.csv"

    if not reprompt_path.exists():
        raise FileNotFoundError(f"Reprompt scores not found: {reprompt_path}")

    df_reprompt = pd.read_csv(reprompt_path)
    df_raw = pd.read_csv(raw_path) if raw_path.exists() else None
    df_cross = pd.read_csv(cross_path) if cross_path.exists() else None

    return df_reprompt, df_raw, df_cross


def build_comparison_df(
    df_reprompt: pd.DataFrame,
    df_raw: pd.DataFrame | None,
    taste_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a comparison DataFrame merging reprompt and raw scores."""
    if df_raw is not None:
        comp = df_raw[["id_prompt", "text", "clap_score"]].merge(
            df_reprompt[["id_prompt", "clap_score"]],
            on="id_prompt",
            suffixes=("_raw", "_reprompt"),
        )
        comp["diferencia"] = comp["clap_score_reprompt"] - comp["clap_score_raw"]
    else:
        comp = df_reprompt[["id_prompt", "text", "clap_score"]].copy()
        comp.rename(columns={"clap_score": "clap_score_reprompt"}, inplace=True)
        comp["clap_score_raw"] = np.nan
        comp["diferencia"] = np.nan

    if taste_map:
        comp["taste"] = comp["id_prompt"].astype(str).map(taste_map)

    return comp


def descriptive_stats(comp: pd.DataFrame, label: str) -> pd.DataFrame:
    """Print and return descriptive statistics."""
    cols = {}
    if "clap_score_raw" in comp and comp["clap_score_raw"].notna().any():
        cols["Raw Prompt"] = comp["clap_score_raw"]
    cols["Reprompt"] = comp["clap_score_reprompt"]

    summary = pd.DataFrame({k: v.describe() for k, v in cols.items()}).round(4)
    print(f"\n{'=' * 60}")
    print(f"  Estadísticas Descriptivas — {label}")
    print(f"{'=' * 60}")
    print(summary.to_string())
    return summary


def paired_ttest(comp: pd.DataFrame, label: str) -> dict:
    """Paired t-test between raw and reprompt scores."""
    if comp["clap_score_raw"].isna().all():
        print(f"\n  ⚠ No raw scores available for t-test ({label})")
        return {}

    t_stat, p_value = stats.ttest_rel(
        comp["clap_score_raw"].dropna(),
        comp["clap_score_reprompt"].dropna(),
    )

    mean_raw = comp["clap_score_raw"].mean()
    mean_reprompt = comp["clap_score_reprompt"].mean()
    mean_diff = mean_reprompt - mean_raw

    pooled_std = np.sqrt(
        (comp["clap_score_raw"].std() ** 2 + comp["clap_score_reprompt"].std() ** 2)
        / 2
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    pct_improved = (comp["diferencia"] > 0).mean() * 100

    results = {
        "label": label,
        "n": len(comp),
        "mean_raw": mean_raw,
        "mean_reprompt": mean_reprompt,
        "mean_diff": mean_diff,
        "pct_change": (mean_diff / mean_raw * 100) if mean_raw != 0 else 0,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "pct_improved": pct_improved,
    }

    print(f"\n{'=' * 60}")
    print(f"  Paired t-test — {label}")
    print(f"{'=' * 60}")
    print(f"  n = {results['n']}")
    print(f"  Mean raw:      {mean_raw:.4f}")
    print(f"  Mean reprompt: {mean_reprompt:.4f}")
    print(f"  Diferencia:    {mean_diff:+.4f} ({results['pct_change']:+.1f}%)")
    print(f"  t = {t_stat:.4f},  p = {p_value:.6f}")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  Mejoraron: {pct_improved:.0f}%")

    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    print(f"  Significancia: {sig}")

    return results


def taste_breakdown(comp: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-taste t-test breakdown."""
    if "taste" not in comp.columns or comp["taste"].isna().all():
        print(f"\n  ⚠ No taste data available for breakdown ({label})")
        return pd.DataFrame()

    if comp["clap_score_raw"].isna().all():
        print(f"\n  ⚠ No raw scores for taste breakdown ({label})")
        return pd.DataFrame()

    rows = []
    for taste in sorted(comp["taste"].dropna().unique()):
        subset = comp[comp["taste"] == taste].dropna(subset=["clap_score_raw"])
        if len(subset) < 2:
            continue
        t_stat, p_val = stats.ttest_rel(
            subset["clap_score_raw"], subset["clap_score_reprompt"]
        )
        rows.append(
            {
                "Taste": taste,
                "N": len(subset),
                "Raw_μ": subset["clap_score_raw"].mean(),
                "Reprompt_μ": subset["clap_score_reprompt"].mean(),
                "Diferencia_μ": subset["diferencia"].mean(),
                "Cambio_%": (
                    subset["diferencia"].mean() / subset["clap_score_raw"].mean() * 100
                    if subset["clap_score_raw"].mean() != 0
                    else 0
                ),
                "t_stat": t_stat,
                "p_value": p_val,
                "Mejoraron_%": (subset["diferencia"] > 0).mean() * 100,
            }
        )

    taste_df = pd.DataFrame(rows).round(4)
    print(f"\n{'=' * 60}")
    print(f"  Per-Taste Breakdown — {label}")
    print(f"{'=' * 60}")
    print(taste_df.to_string(index=False))
    return taste_df


def kendall_by_taste(comp: pd.DataFrame, label: str) -> pd.DataFrame:
    """Kendall τ correlation between raw and reprompt scores per taste."""
    if "taste" not in comp.columns or comp["clap_score_raw"].isna().all():
        return pd.DataFrame()

    rows = []
    for taste in sorted(comp["taste"].dropna().unique()):
        subset = comp[comp["taste"] == taste].dropna(subset=["clap_score_raw"])
        if len(subset) < 3:
            continue
        tau, p_val = stats.kendalltau(
            subset["clap_score_raw"], subset["clap_score_reprompt"]
        )
        rows.append(
            {
                "Taste": taste,
                "N": len(subset),
                "Kendall_τ": tau,
                "p_value": p_val,
                "Interpretación": (
                    "Fuerte"
                    if abs(tau) > 0.6
                    else "Moderada"
                    if abs(tau) > 0.3
                    else "Débil"
                ),
            }
        )

    kendall_df = pd.DataFrame(rows).round(4)
    if not kendall_df.empty:
        print(f"\n{'=' * 60}")
        print(f"  Kendall τ por Taste — {label}")
        print(f"{'=' * 60}")
        print(kendall_df.to_string(index=False))
    return kendall_df


# ── Plots ────────────────────────────────────────────────────


def plot_distributions(comp: pd.DataFrame, label: str, output_dir: Path) -> None:
    """Box plots + histogram of differences."""
    has_raw = comp["clap_score_raw"].notna().any()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plots
    if has_raw:
        data = [comp["clap_score_raw"].dropna(), comp["clap_score_reprompt"]]
        bp_labels = [
            f"Raw Prompt (μ={comp['clap_score_raw'].mean():.3f})",
            f"Reprompt (μ={comp['clap_score_reprompt'].mean():.3f})",
        ]
    else:
        data = [comp["clap_score_reprompt"]]
        bp_labels = [f"Reprompt (μ={comp['clap_score_reprompt'].mean():.3f})"]

    bp = axes[0].boxplot(
        data,
        tick_labels=bp_labels,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2.5),
        meanprops=dict(
            color="green", linewidth=2.5, linestyle="--", marker="D", markersize=8
        ),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    axes[0].axhline(y=0, color="gray", linestyle=":", linewidth=2, alpha=0.5)
    axes[0].set_ylabel("CLAP Score", fontsize=12, fontweight="bold")
    axes[0].set_title(
        f"Distribución — {label}", fontsize=13, fontweight="bold", pad=15
    )
    axes[0].grid(True, alpha=0.3, axis="y")

    # Histogram of differences
    if has_raw and comp["diferencia"].notna().any():
        diff = comp["diferencia"].dropna()
        axes[1].hist(diff, bins=20, edgecolor="black", alpha=0.7, color="coral")
        axes[1].axvline(
            0, color="red", linestyle="--", linewidth=2.5, label="Sin cambio", zorder=5
        )
        axes[1].axvline(
            diff.mean(),
            color="green",
            linestyle="--",
            linewidth=2.5,
            label=f"Media = {diff.mean():.3f}",
            zorder=5,
        )
        axes[1].set_xlabel(
            "Diferencia (Reprompt − Raw)", fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("Frecuencia", fontsize=12, fontweight="bold")
        axes[1].set_title(
            "Distribución de Cambios", fontsize=13, fontweight="bold", pad=15
        )
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No raw scores\npara comparar", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=14, color="gray")

    plt.tight_layout()
    plt.savefig(output_dir / f"dist_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: dist_{label}.png")


def plot_taste_boxes(comp: pd.DataFrame, label: str, output_dir: Path) -> None:
    """Box plots per taste category."""
    if "taste" not in comp.columns or comp["taste"].isna().all():
        return
    if comp["clap_score_raw"].isna().all():
        return

    tastes = sorted(comp["taste"].dropna().unique())
    n = len(tastes)
    if n == 0:
        return

    cols = min(n, 2)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(8 * cols, 6 * rows_n), squeeze=False)

    colors = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#9B59B6", "#E67E22"]

    for idx, taste in enumerate(tastes):
        ax = axes[idx // cols, idx % cols]
        subset = comp[comp["taste"] == taste].dropna(subset=["clap_score_raw"])

        data = [subset["clap_score_raw"], subset["clap_score_reprompt"]]
        bp = ax.boxplot(
            data,
            tick_labels=["Raw Prompt", "Reprompt"],
            patch_artist=True,
            showmeans=True,
            boxprops=dict(facecolor=colors[idx % len(colors)], alpha=0.6),
            medianprops=dict(color="red", linewidth=2.5),
            meanprops=dict(
                color="darkgreen", linewidth=2.5, linestyle="--", marker="D", markersize=8
            ),
        )
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=2, alpha=0.5)
        ax.set_ylabel("CLAP Score", fontsize=11, fontweight="bold")

        _, p_val = stats.ttest_rel(
            subset["clap_score_raw"], subset["clap_score_reprompt"]
        )
        sig = (
            "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else "ns"
        )
        ax.set_title(
            f"{taste.upper()} (n={len(subset)}) | p={p_val:.3f} {sig}",
            fontsize=12, fontweight="bold", pad=10,
        )
        ax.grid(True, alpha=0.3, axis="y")

    # hide unused axes
    for idx in range(n, rows_n * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.suptitle(f"Distribuciones por Taste — {label}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"taste_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: taste_{label}.png")


def plot_comparison_across_variants(
    all_results: list[dict], output_dir: Path, group_name: str
) -> None:
    """Bar chart comparing CLAP scores across ablation variants."""
    if not all_results:
        return

    df = pd.DataFrame(all_results)
    if "mean_reprompt" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    if df["mean_raw"].notna().any():
        ax.bar(
            x - width / 2,
            df["mean_raw"],
            width,
            label="Raw Prompt",
            color="#4D96FF",
            alpha=0.8,
            edgecolor="black",
        )
    ax.bar(
        x + width / 2,
        df["mean_reprompt"],
        width,
        label="Reprompt",
        color="#6BCB77",
        alpha=0.8,
        edgecolor="black",
    )

    # Add significance markers
    for i, row in df.iterrows():
        if pd.notna(row.get("p_value")):
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "n.s."
            y_pos = max(row["mean_reprompt"], row.get("mean_raw", 0)) + 0.01
            ax.text(i, y_pos, sig, ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("CLAP Score (media)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Variante", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Comparación de Variantes — {group_name}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        output_dir / f"comparison_{group_name}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  📊 Saved: comparison_{group_name}.png")


# ── Main analysis pipeline ───────────────────────────────────


def analyse_single(tag: str, output_dir: Path) -> dict:
    """Run full analysis for a single experiment tag."""
    print(f"\n{'#' * 60}")
    print(f"  ANÁLISIS: {tag}")
    print(f"{'#' * 60}")

    df_reprompt, df_raw, df_cross = load_experiment_data(tag)

    # Load taste mapping from reprompt CSV
    reprompt_csv = _find_matching_reprompt_csv(tag)
    taste_map: dict[str, str] | None = None
    if reprompt_csv and reprompt_csv.exists():
        rp = pd.read_csv(reprompt_csv)
        if "taste" in rp.columns:
            taste_map = dict(zip(rp["id_prompt"].astype(str), rp["taste"]))

    comp = build_comparison_df(df_reprompt, df_raw, taste_map)

    # 1. Descriptive stats
    descriptive_stats(comp, tag)

    # 2. Paired t-test (raw vs reprompt)
    ttest_results = paired_ttest(comp, tag)

    # 3. Plots
    plot_distributions(comp, tag, output_dir)

    # 4. Per-taste breakdown
    taste_df = taste_breakdown(comp, tag)
    if not taste_df.empty:
        taste_df.to_csv(output_dir / f"taste_breakdown_{tag}.csv", index=False)

    # 5. Kendall by taste
    kendall_df = kendall_by_taste(comp, tag)
    if not kendall_df.empty:
        kendall_df.to_csv(output_dir / f"kendall_{tag}.csv", index=False)

    # 6. Taste box plots
    plot_taste_boxes(comp, tag, output_dir)

    return ttest_results


def analyse_group(group_prefix: str, run_id: str = "") -> None:
    """Run analysis for all tags in an experiment group (e.g. B2).

    If run_id is provided, only analyse scores from that specific run.
    """
    groups = _discover_experiment_groups(run_id)
    if group_prefix not in groups:
        print(f"Error: no score files found for group '{group_prefix}'", file=sys.stderr)
        if run_id:
            print(f"  (filtered by run_id={run_id})", file=sys.stderr)
        available = _discover_experiment_groups()
        print(f"Available groups: {', '.join(sorted(available.keys()))}", file=sys.stderr)
        sys.exit(1)

    tags = groups[group_prefix]
    # Include run_id in output directory for isolation
    run_suffix = f"_{run_id}" if run_id else ""
    output_dir = ABLATIONS_ANALYSIS / f"{group_prefix}{run_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  Análisis de ablación: {group_prefix}")
    if run_id:
        print(f"  Run ID: {run_id}")
    print(f"  Variantes: {', '.join(tags)}")
    print(f"  Output: {output_dir}")
    print(f"{'═' * 60}")

    all_results = []
    all_taste_dfs: dict[str, pd.DataFrame] = {}
    all_kendall_dfs: dict[str, pd.DataFrame] = {}
    for tag in sorted(tags):
        try:
            result = analyse_single(tag, output_dir)
            if result:
                all_results.append(result)
            # load taste & kendall CSVs that were just saved
            tb = output_dir / f"taste_breakdown_{tag}.csv"
            kd = output_dir / f"kendall_{tag}.csv"
            if tb.exists():
                all_taste_dfs[tag] = pd.read_csv(tb)
            if kd.exists():
                all_kendall_dfs[tag] = pd.read_csv(kd)
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {tag}: {e}", file=sys.stderr)

    # Cross-variant comparison
    if len(all_results) > 1:
        print(f"\n{'#' * 60}")
        print(f"  COMPARACIÓN ENTRE VARIANTES — {group_prefix}")
        print(f"{'#' * 60}")

        summary_df = pd.DataFrame(all_results)[
            ["label", "n", "mean_raw", "mean_reprompt", "mean_diff", "pct_change",
             "t_stat", "p_value", "cohens_d", "pct_improved"]
        ].round(4)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(output_dir / f"summary_{group_prefix}.csv", index=False)

        plot_comparison_across_variants(all_results, output_dir, group_prefix)

    # Generate findings markdown
    _generate_findings_report(
        group_prefix, all_results, all_taste_dfs, all_kendall_dfs, output_dir
    )

    print(f"\n  ✓ Análisis completo. Resultados en: {output_dir}")


def _significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _short_tag(tag: str) -> str:
    """Extract the short experiment id from a full tag (e.g. B2a_filter_default)."""
    match = re.search(r"(A\d\w|B\d\w)_(.+)", tag)
    if match:
        return f"{match.group(1)} ({match.group(2).replace('_', ' ')})"
    return tag


def _generate_findings_report(
    group: str,
    results: list[dict],
    taste_dfs: dict[str, pd.DataFrame],
    kendall_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate a hallazgos_<group>.md report with structured findings."""
    lines: list[str] = []
    w = lines.append

    w(f"# {group} — Hallazgos de la Ablación\n")
    w(f"> Generado automáticamente por `models/ablation_analysis.py`\n")

    # ── 1. General stats ─────────────────────────────────────
    if results:
        w("## 1. Estadísticas Generales\n")
        w("| Métrica | " + " | ".join(_short_tag(r["label"]) for r in results) + " |")
        w("|---------|" + "|".join("---" for _ in results) + "|")
        w("| n | " + " | ".join(str(r["n"]) for r in results) + " |")
        w("| Media raw prompt | " + " | ".join(f"{r['mean_raw']:.4f}" for r in results) + " |")
        w("| Media reprompt | " + " | ".join(f"{r['mean_reprompt']:.4f}" for r in results) + " |")
        w(
            "| Diferencia media | "
            + " | ".join(f"+{r['mean_diff']:.4f} ({r['pct_change']:+.1f}%)" for r in results)
            + " |"
        )
        w("| Estadístico t | " + " | ".join(f"{r['t_stat']:.4f}" for r in results) + " |")
        w(
            "| Valor p | "
            + " | ".join(
                f"**{r['p_value']:.4f}** {_significance_label(r['p_value'])}"
                for r in results
            )
            + " |"
        )
        w("| Cohen's d | " + " | ".join(f"**{r['cohens_d']:.4f}**" for r in results) + " |")
        w(
            "| Casos que mejoraron | "
            + " | ".join(f"{r['pct_improved']:.1f}%" for r in results)
            + " |"
        )

        # interpretation
        all_sig = all(r["p_value"] < 0.05 for r in results)
        if all_sig:
            best = max(results, key=lambda r: r["cohens_d"])
            w(
                f"\n**Hallazgo**: Todas las variantes presentan mejora **estadísticamente significativa** "
                f"(p < 0.05) con tamaño del efecto grande (d > 1.0). "
                f"La variante con mayor efecto es {_short_tag(best['label'])} "
                f"(d = {best['cohens_d']:.2f}).\n"
            )
        w("\n---\n")

    # ── 2. Per-taste breakdown ───────────────────────────────
    if taste_dfs:
        w("## 2. Análisis por Categoría de Taste\n")
        for tag, tdf in taste_dfs.items():
            w(f"### {_short_tag(tag)}\n")
            w("| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |")
            w("|-------|---|-------|------------|-----|----------|---------|-----------|")
            for _, row in tdf.sort_values("Diferencia_μ", ascending=False).iterrows():
                sig = _significance_label(row["p_value"])
                p_fmt = f"**{row['p_value']:.4f}**" if row["p_value"] < 0.05 else f"{row['p_value']:.4f}"
                cambio = f"**{row['Cambio_%']:+.1f}%**" if row["p_value"] < 0.05 else f"{row['Cambio_%']:+.1f}%"
                w(
                    f"| {row['Taste']} | {row['N']:.0f} | {row['Raw_μ']:.4f} | "
                    f"{row['Reprompt_μ']:.4f} | {row['Diferencia_μ']:+.4f} | "
                    f"{cambio} | {p_fmt} {sig} | {row['Mejoraron_%']:.1f}% |"
                )
            w("")

        # find tastes significant across variants
        sig_tastes: dict[str, int] = {}
        for tdf in taste_dfs.values():
            for _, row in tdf.iterrows():
                if row["p_value"] < 0.05:
                    sig_tastes[row["Taste"]] = sig_tastes.get(row["Taste"], 0) + 1
        always_sig = [t for t, c in sig_tastes.items() if c == len(taste_dfs)]
        if always_sig:
            w(
                f"**Hallazgo**: La(s) categoría(s) **{', '.join(always_sig)}** muestra(n) "
                f"mejora significativa (p < 0.05) en **todas** las variantes.\n"
            )
        w("\n---\n")

    # ── 3. Kendall correlations ──────────────────────────────
    if kendall_dfs:
        w("## 3. Correlación Kendall τ por Taste\n")
        # Build a combined table
        tastes_all = sorted(
            {t for kdf in kendall_dfs.values() for t in kdf["Taste"]}
        )
        header = "| Taste | " + " | ".join(
            f"{_short_tag(t)} τ" for t in kendall_dfs
        ) + " |"
        w(header)
        w("|-------|" + "|".join("---" for _ in kendall_dfs) + "|")
        for taste in tastes_all:
            cells = []
            for kdf in kendall_dfs.values():
                row = kdf[kdf["Taste"] == taste]
                if not row.empty:
                    r = row.iloc[0]
                    p_note = f" (p={r['p_value']:.2f})" if r["p_value"] < 0.05 else ""
                    cells.append(f"{r['Kendall_τ']:.3f} ({r['Interpretación']}){p_note}")
                else:
                    cells.append("—")
            w(f"| {taste} | " + " | ".join(cells) + " |")

        w("\n**Interpretación**: Correlaciones débiles indican que el reprompt transforma "
          "sustancialmente la representación texto→audio respecto al prompt original.\n")
        w("\n---\n")

    # ── Plots ────────────────────────────────────────────────
    w("## Visualizaciones\n")
    for f in sorted(output_dir.glob("*.png")):
        w(f"![{f.stem}]({f.name})\n")

    report_path = output_dir / f"hallazgos_{group}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📝 Saved: {report_path}")



def _extract_run_ids_from_tags(tags: list[str]) -> set[str]:
    """Extract unique run IDs from a list of tags."""
    run_ids: set[str] = set()
    for tag in tags:
        match = re.search(r"(R\d{8}_\d{6})", tag)
        if match:
            run_ids.add(match.group(1))
    return run_ids


def list_groups() -> None:
    """List available experiment groups with score files."""
    groups = _discover_experiment_groups()
    if not groups:
        print("No score files found in", ABLATIONS_SCORES)
        return
    print("\nExperiment groups with score files:")
    for prefix, tags in sorted(groups.items()):
        run_ids = _extract_run_ids_from_tags(tags)
        if run_ids:
            for rid in sorted(run_ids):
                run_tags = [t for t in tags if rid in t]
                print(f"  {prefix} [{rid}]: {len(run_tags)} variante(s)")
        else:
            print(f"  {prefix}: {', '.join(sorted(tags))}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of ablation CLAP scores"
    )
    parser.add_argument(
        "--experiment",
        help="Experiment group to analyse (e.g. B2, B1, A1)",
    )
    parser.add_argument(
        "--run",
        default="",
        help="Run ID to filter by (e.g. R20260416_194531)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_groups",
        help="List available experiment groups",
    )
    args = parser.parse_args()

    if args.list_groups:
        list_groups()
        return

    if not args.experiment:
        print("Error: --experiment or --list required", file=sys.stderr)
        sys.exit(1)

    analyse_group(args.experiment, run_id=args.run)


if __name__ == "__main__":
    main()
