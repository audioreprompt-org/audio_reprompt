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
import itertools
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import PROJECT_ROOT, setup_project_paths
from models.validate import compute_sample_hash

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
    """Discover score files and group them by experiment prefix (B1, B2, B3, etc.).

    If run_id is provided, only include tags containing that run ID.
    """
    score_files: list[Path] = []
    for pattern in (
        "clap_score_results_reprompt_outputs_*.csv",
        "clap_score_results_prompt_outputs_*_raw.csv",
        "clap_score_results_prompt_outputs_*_cross.csv",
    ):
        score_files.extend(ABLATIONS_SCORES.glob(pattern))

    groups: dict[str, set[str]] = {}
    for f in sorted(score_files):
        if f.name == "clap_score_results_raw_prompts_baseline.csv":
            continue
        tag = _extract_tag(f.name)
        if run_id and run_id not in tag:
            continue
        match = re.match(r".*?(B\d)", tag)
        if match:
            prefix = match.group(1)
            groups.setdefault(prefix, set()).add(tag)
    return {prefix: sorted(tags) for prefix, tags in groups.items()}


# ── Analysis functions ───────────────────────────────────────


def _is_true_raw_baseline_df(df: pd.DataFrame | None) -> bool:
    """Return True if a _raw.csv DataFrame explicitly comes from raw_ablations."""
    if df is None or df.empty or "source" not in df.columns:
        return False
    return bool((df["source"] == "raw_ablations").all())


def load_experiment_data(
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load reprompt, true raw baseline, and cross score CSVs for a given tag.

    Requires all three CSVs to exist and requires `_raw.csv` to carry
    ``source == "raw_ablations"`` without NaN baseline scores.
    """
    reprompt_path = ABLATIONS_SCORES / f"clap_score_results_reprompt_outputs_{tag}.csv"
    raw_path = ABLATIONS_SCORES / f"clap_score_results_prompt_outputs_{tag}_raw.csv"
    cross_path = ABLATIONS_SCORES / f"clap_score_results_prompt_outputs_{tag}_cross.csv"

    missing = [p for p in (reprompt_path, raw_path, cross_path) if not p.exists()]
    if missing:
        missing_names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Faltan archivos de score requeridos para '{tag}': {missing_names}"
        )

    df_reprompt = pd.read_csv(reprompt_path)
    df_raw = pd.read_csv(raw_path)
    df_cross = pd.read_csv(cross_path)

    if df_reprompt.empty or df_cross.empty:
        raise ValueError(f"CSVs de score vacíos para '{tag}'")
    if not _is_true_raw_baseline_df(df_raw):
        raise ValueError(
            f"_raw.csv para '{tag}' no tiene source=='raw_ablations' (no hay fallback)"
        )
    if df_raw["clap_score"].isna().any():
        raise ValueError(f"_raw.csv para '{tag}' contiene NaN en clap_score")

    return df_reprompt, df_raw, df_cross


def build_comparison_df(
    df_reprompt: pd.DataFrame,
    df_raw: pd.DataFrame | None,
    taste_map: dict[str, str] | None = None,
    df_cross: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a comparison DataFrame merging reprompt, raw baseline, and cross scores."""
    rep = df_reprompt.copy()
    rep["id_prompt"] = rep["id_prompt"].astype(str)

    if df_raw is not None:
        raw = df_raw.copy()
        raw["id_prompt"] = raw["id_prompt"].astype(str)
        comp = raw[["id_prompt", "text", "clap_score"]].merge(
            rep[["id_prompt", "clap_score"]],
            on="id_prompt",
            suffixes=("_raw", "_reprompt"),
        )
        comp["diferencia"] = comp["clap_score_reprompt"] - comp["clap_score_raw"]
        comp["has_true_raw_audio"] = _is_true_raw_baseline_df(df_raw)
    else:
        comp = rep[["id_prompt", "text", "clap_score"]].copy()
        comp.rename(columns={"clap_score": "clap_score_reprompt"}, inplace=True)
        comp["clap_score_raw"] = np.nan
        comp["diferencia"] = np.nan
        comp["has_true_raw_audio"] = False

    if df_cross is not None:
        cross = df_cross.copy()
        cross["id_prompt"] = cross["id_prompt"].astype(str)
        cross_sub = cross[["id_prompt", "clap_score"]].rename(
            columns={"clap_score": "clap_score_cross"}
        )
        comp = comp.merge(cross_sub, on="id_prompt", how="left")
    else:
        comp["clap_score_cross"] = np.nan

    if taste_map:
        comp["taste"] = comp["id_prompt"].astype(str).map(taste_map)

    return comp


def descriptive_stats(comp: pd.DataFrame, label: str) -> pd.DataFrame:
    """Print and return descriptive statistics."""
    cols = {}
    if "clap_score_raw" in comp and comp["clap_score_raw"].notna().any():
        cols["Raw Baseline"] = comp["clap_score_raw"]
    cols["Reprompt"] = comp["clap_score_reprompt"]
    if "clap_score_cross" in comp and comp["clap_score_cross"].notna().any():
        cols["Cross"] = comp["clap_score_cross"]

    summary = pd.DataFrame({k: v.describe() for k, v in cols.items()}).round(4)
    print(f"\n{'=' * 60}")
    print(f"  Estadísticas Descriptivas — {label}")
    print(f"{'=' * 60}")
    print(summary.to_string())
    return summary


def paired_ttest(comp: pd.DataFrame, label: str) -> dict:
    """Paired statistical tests within a variant:
    1. reprompt vs baseline raw (CLAP(reprompt, audio_rep) vs CLAP(prompt, audio_prompt))
    2. cross vs baseline raw (CLAP(prompt, audio_rep) vs CLAP(prompt, audio_prompt))
    """
    valid_rep = comp.dropna(subset=["clap_score_raw", "clap_score_reprompt"])
    if valid_rep.empty:
        print(f"\n  ⚠ No raw scores available for paired tests ({label})")
        return {}

    diff_rep = valid_rep["clap_score_reprompt"] - valid_rep["clap_score_raw"]
    mean_raw = float(valid_rep["clap_score_raw"].mean())
    mean_reprompt = float(valid_rep["clap_score_reprompt"].mean())
    delta_rep = float(diff_rep.mean())

    if len(valid_rep) >= 2:
        t_stat, p_rep = stats.ttest_rel(
            valid_rep["clap_score_reprompt"], valid_rep["clap_score_raw"]
        )
        t_stat = float(t_stat)
        p_rep = float(p_rep)
        try:
            _, p_w_rep = stats.wilcoxon(diff_rep)
            p_w_rep = float(p_w_rep)
        except Exception:
            p_w_rep = np.nan
        sd_diff_rep = float(diff_rep.std(ddof=1))
        dz_rep = delta_rep / sd_diff_rep if sd_diff_rep > 0 else 0.0
        pooled_std = float(
            np.sqrt(
                (valid_rep["clap_score_raw"].std() ** 2 + valid_rep["clap_score_reprompt"].std() ** 2)
                / 2
            )
        )
        cohens_d = delta_rep / pooled_std if pooled_std > 0 else 0.0
    else:
        t_stat, p_rep, p_w_rep, dz_rep, cohens_d = np.nan, np.nan, np.nan, 0.0, 0.0

    pct_improved = float((diff_rep > 0).mean() * 100)
    pct_change = (delta_rep / mean_raw * 100) if mean_raw != 0 else 0.0

    # Cross vs raw baseline (same text: original prompt)
    valid_cross = (
        comp.dropna(subset=["clap_score_raw", "clap_score_cross"])
        if "clap_score_cross" in comp.columns
        else pd.DataFrame()
    )
    if not valid_cross.empty:
        diff_cross = valid_cross["clap_score_cross"] - valid_cross["clap_score_raw"]
        mean_cross = float(valid_cross["clap_score_cross"].mean())
        delta_cross = float(diff_cross.mean())
        if len(valid_cross) >= 2:
            t_stat_cross, p_t_cross = stats.ttest_rel(
                valid_cross["clap_score_cross"], valid_cross["clap_score_raw"]
            )
            t_stat_cross = float(t_stat_cross)
            p_t_cross = float(p_t_cross)
            try:
                _, p_w_cross = stats.wilcoxon(diff_cross)
                p_w_cross = float(p_w_cross)
            except Exception:
                p_w_cross = np.nan
            sd_diff_cross = float(diff_cross.std(ddof=1))
            dz_cross = delta_cross / sd_diff_cross if sd_diff_cross > 0 else 0.0
        else:
            t_stat_cross, p_t_cross, p_w_cross, dz_cross = np.nan, np.nan, np.nan, 0.0
        p_cross = p_w_cross if pd.notna(p_w_cross) else p_t_cross
        pct_improved_cross = float((diff_cross > 0).mean() * 100)
    else:
        mean_cross = np.nan
        delta_cross = np.nan
        t_stat_cross = np.nan
        p_t_cross = np.nan
        p_w_cross = np.nan
        p_cross = np.nan
        dz_cross = np.nan
        pct_improved_cross = np.nan

    results = {
        "label": label,
        "n": int(len(valid_rep)),
        "mean_raw_baseline": mean_raw,
        "mean_raw": mean_raw,
        "mean_reprompt": mean_reprompt,
        "mean_cross": mean_cross,
        "delta_rep_vs_raw": delta_rep,
        "mean_diff": delta_rep,
        "pct_change": pct_change,
        "t_stat": t_stat,
        "p_rep_vs_raw": p_rep,
        "p_value": p_rep,
        "p_wilcoxon_rep_vs_raw": p_w_rep,
        "dz_rep_vs_raw": dz_rep,
        "cohens_d": cohens_d,
        "pct_improved": pct_improved,
        "delta_cross_vs_raw": delta_cross,
        "t_stat_cross": t_stat_cross,
        "p_cross_vs_raw": p_cross,
        "p_t_cross_vs_raw": p_t_cross,
        "p_wilcoxon_cross_vs_raw": p_w_cross,
        "dz_cross_vs_raw": dz_cross,
        "pct_improved_cross": pct_improved_cross,
    }

    print(f"\n{'=' * 60}")
    print(f"  Paired Tests — {label}")
    print(f"{'=' * 60}")
    print(f"  n = {results['n']}")
    print(f"  Mean raw baseline: {mean_raw:.4f}")
    print(f"  Mean reprompt:     {mean_reprompt:.4f}")
    print(f"  Δ rep vs raw:      {delta_rep:+.4f} ({pct_change:+.1f}%), t={t_stat:.4f}, p={p_rep:.6f}, d_z={dz_rep:.4f}")
    if pd.notna(mean_cross):
        print(f"  Mean cross:        {mean_cross:.4f}")
        print(f"  Δ cross vs raw:    {delta_cross:+.4f}, p_w={p_w_cross:.6f}, d_z={dz_cross:.4f}")
    print(f"  Mejoraron (rep):   {pct_improved:.0f}%")

    sig = _significance_label(p_rep)
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
        subset = comp[comp["taste"] == taste].dropna(
            subset=["clap_score_raw", "clap_score_reprompt"]
        )
        if len(subset) < 2:
            continue
        t_stat, p_val = stats.ttest_rel(
            subset["clap_score_reprompt"], subset["clap_score_raw"]
        )
        diff = subset["clap_score_reprompt"] - subset["clap_score_raw"]
        rows.append(
            {
                "Taste": taste,
                "N": len(subset),
                "Raw_μ": subset["clap_score_raw"].mean(),
                "Reprompt_μ": subset["clap_score_reprompt"].mean(),
                "Diferencia_μ": diff.mean(),
                "Cambio_%": (
                    diff.mean() / subset["clap_score_raw"].mean() * 100
                    if subset["clap_score_raw"].mean() != 0
                    else 0
                ),
                "t_stat": t_stat,
                "p_value": p_val,
                "Mejoraron_%": (diff > 0).mean() * 100,
            }
        )

    taste_df = pd.DataFrame(rows).round(4)
    if not taste_df.empty:
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
        subset = comp[comp["taste"] == taste].dropna(
            subset=["clap_score_raw", "clap_score_reprompt"]
        )
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
        subset = comp[comp["taste"] == taste].dropna(
            subset=["clap_score_raw", "clap_score_reprompt"]
        )

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

        if len(subset) >= 2:
            _, p_val = stats.ttest_rel(
                subset["clap_score_reprompt"], subset["clap_score_raw"]
            )
            sig = (
                "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else "ns"
            )
            p_txt = f"p={p_val:.3f} {sig}"
        else:
            p_txt = "p=—"
        ax.set_title(
            f"{taste.upper()} (n={len(subset)}) | {p_txt}",
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


def _holm_adjust(p_series: pd.Series) -> pd.Series:
    """Apply Holm correction over non-NaN p-values while preserving NaNs."""
    adjusted = pd.Series(np.nan, index=p_series.index, dtype=float)
    valid_mask = p_series.notna()
    if valid_mask.any():
        adjusted.loc[valid_mask] = multipletests(
            p_series.loc[valid_mask].astype(float), method="holm"
        )[1]
    return adjusted


def compare_variants_on_column(
    comp_by_tag: dict[str, pd.DataFrame],
    group_prefix: str,
    col: str = "clap_score_raw",
) -> tuple[dict, pd.DataFrame]:
    """Compare variants against each other on a paired per-prompt metric column."""
    series = {
        tag: df.set_index("id_prompt")[col]
        for tag, df in comp_by_tag.items()
        if col in df and df[col].notna().any()
    }
    if len(series) < 2:
        return {}, pd.DataFrame()

    wide = pd.DataFrame(series).dropna()
    tags = list(wide.columns)
    k = len(tags)

    omnibus: dict = {"n": len(wide), "k": k}
    if k > 2 and len(wide) >= 2:
        try:
            fr = stats.friedmanchisquare(*[wide[t] for t in tags])
            omnibus["friedman_chi2"] = float(fr.statistic)
            omnibus["friedman_p"] = float(fr.pvalue)
            omnibus["kendall_w"] = float(fr.statistic / (len(wide) * (k - 1)))
        except Exception:
            pass

    # Control variant for B3 (B3b) or all pairs for B1/B2
    ctrl = next((t for t in tags if re.search(r"B3b_", t)), None) if group_prefix == "B3" else None
    pairs = [(t, ctrl) for t in tags if t != ctrl] if ctrl else list(itertools.combinations(tags, 2))

    rows = []
    for a, b in pairs:
        diff = wide[a] - wide[b]
        t_stat, p_t = stats.ttest_rel(wide[a], wide[b])
        try:
            _, p_w = stats.wilcoxon(diff)
        except Exception:
            p_w = np.nan
        sd = diff.std(ddof=1)
        dz = diff.mean() / sd if sd > 0 else 0.0
        rows.append(
            {
                "pair": f"{_short_tag(a).split()[0]} vs {_short_tag(b).split()[0]}",
                "tag_a": a,
                "tag_b": b,
                "mean_a": wide[a].mean(),
                "mean_b": wide[b].mean(),
                "mean_diff": diff.mean(),
                "t_stat": t_stat,
                "p_t": p_t,
                "p_wilcoxon": p_w,
                "cohens_dz": dz,
            }
        )

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df["p_t_holm"] = _holm_adjust(pair_df["p_t"])
        pair_df["p_w_holm"] = _holm_adjust(pair_df["p_wilcoxon"])
        pair_df = pair_df.round(4)

    return omnibus, pair_df


def compare_variants_on_cross(
    comp_by_tag: dict[str, pd.DataFrame], group_prefix: str
) -> tuple[dict, pd.DataFrame]:
    """Compare variants against each other on CLAP(original_prompt, reprompt_audio)."""
    return compare_variants_on_column(comp_by_tag, group_prefix, col="clap_score_cross")


def _load_or_compute_nlp_for_tag(
    tag: str, reprompt_csv: Path | None, output_dir: Path
) -> tuple[dict | None, pd.DataFrame | None, str | None, dict[str, str]]:
    """Read-only loader for corpus-level and row-level NLP diversity metrics.

    Never invokes NLP computation on the fly. Returns
    ``(summary_dict, rows_df, error_reason, backends_dict)``.
    """
    csv_tag = (
        reprompt_csv.stem.replace("pipeline_results_", "")
        if reprompt_csv is not None
        else tag
    )

    candidate_dirs: list[Path] = [output_dir]
    m = re.search(r"(R\d{8}_\d{6})", tag)
    if m:
        candidate_dirs.append(ABLATIONS_ANALYSIS / f"nlp_{m.group(1)}")
    candidate_dirs.append(ABLATIONS_ANALYSIS / "nlp")

    status_backends: dict[str, str] = {}
    for d in candidate_dirs:
        status_path = d / "nlp_status.json"
        if not status_path.exists():
            continue
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(status_data, dict):
            continue
        entry = status_data.get(csv_tag) or status_data.get(tag)
        if isinstance(entry, dict):
            status_backends = entry.get("backends") or {}
            status_val = entry.get("status", "")
            if status_val != "ok":
                reason = entry.get("error") or status_val or "NLP falló"
                return None, None, str(reason), status_backends

    summary_csv: Path | None = None
    rows_csv: Path | None = None
    for d in candidate_dirs:
        cand_summary = d / f"nlp_eval_{csv_tag}.csv"
        cand_rows = d / f"nlp_rows_{csv_tag}.csv"
        if cand_summary.exists():
            summary_csv = cand_summary
            rows_csv = cand_rows
            break
        if reprompt_csv is None and d.is_dir():
            matches = sorted(d.glob(f"nlp_eval_*{tag}.csv"))
            if matches:
                summary_csv = matches[0]
                inferred_tag = summary_csv.stem.replace("nlp_eval_", "")
                rows_csv = d / f"nlp_rows_{inferred_tag}.csv"
                break

    if summary_csv is None or not summary_csv.exists():
        return None, None, "resultados NLP no encontrados (ejecutar --phase nlp)", status_backends

    summary_dict = pd.read_csv(summary_csv).iloc[0].to_dict()
    rows_df = pd.read_csv(rows_csv) if rows_csv is not None and rows_csv.exists() else None

    backends = dict(status_backends)
    if not backends:
        for key, col in (
            ("encoder", "backend_encoder"),
            ("tokenizer", "backend_tokenizer"),
            ("flesch", "backend_flesch"),
        ):
            val = summary_dict.get(col)
            if pd.notna(val) and str(val).strip():
                backends[key] = str(val)

    return summary_dict, rows_df, None, backends


def analyse_single(
    tag: str, output_dir: Path
) -> tuple[dict, pd.DataFrame, dict | None, str | None, dict[str, str]]:
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

    comp = build_comparison_df(df_reprompt, df_raw, taste_map, df_cross=df_cross)

    # Read-only NLP diversity loading and merge row-level metrics by id_prompt
    nlp_summary, nlp_rows, nlp_error, nlp_backends = _load_or_compute_nlp_for_tag(
        tag, reprompt_csv, output_dir
    )
    if nlp_rows is not None and not nlp_rows.empty:
        nlp_cols = [
            c
            for c in (
                "ttr",
                "mattr",
                "bigram_entropy",
                "flesch",
                "musical_vocab_count",
                "musical_vocab_density",
                "semantic_sim_raw_rep",
                "distinctiveness_cos_dist",
            )
            if c in nlp_rows.columns
        ]
        nlp_sub = nlp_rows[["id_prompt", *nlp_cols]].copy()
        nlp_sub["id_prompt"] = nlp_sub["id_prompt"].astype(str)
        comp = comp.merge(nlp_sub, on="id_prompt", how="left")

    # 1. Descriptive stats
    descriptive_stats(comp, tag)

    # 2. Paired tests (reprompt vs raw baseline, cross vs raw baseline)
    ttest_results = paired_ttest(comp, tag)
    if ttest_results:
        ttest_results["has_true_raw_audio"] = bool(comp["has_true_raw_audio"].any())

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

    return ttest_results, comp, nlp_summary, nlp_error, nlp_backends


SUMMARY_COLUMNS = [
    "label",
    "n",
    "mean_raw_baseline",
    "mean_reprompt",
    "mean_cross",
    "delta_rep_vs_raw",
    "p_rep_vs_raw",
    "p_wilcoxon_rep_vs_raw",
    "dz_rep_vs_raw",
    "delta_cross_vs_raw",
    "p_cross_vs_raw",
    "p_t_cross_vs_raw",
    "p_wilcoxon_cross_vs_raw",
    "dz_cross_vs_raw",
    "pct_change",
    "t_stat",
    "cohens_d",
    "pct_improved",
    "sample_hash",
]


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

    all_results: list[dict] = []
    comp_by_tag: dict[str, pd.DataFrame] = {}
    nlp_by_tag: dict[str, dict] = {}
    nlp_errors_by_tag: dict[str, str] = {}
    nlp_backends_by_tag: dict[str, dict[str, str]] = {}
    excluded_variants: dict[str, str] = {}
    all_taste_dfs: dict[str, pd.DataFrame] = {}
    all_kendall_dfs: dict[str, pd.DataFrame] = {}

    for tag in sorted(tags):
        try:
            result, comp_df, nlp_summary, nlp_error, nlp_backends = analyse_single(tag, output_dir)
            comp_by_tag[tag] = comp_df
            if nlp_summary is not None:
                nlp_by_tag[tag] = nlp_summary
            if nlp_error:
                nlp_errors_by_tag[tag] = nlp_error
            if nlp_backends:
                nlp_backends_by_tag[tag] = nlp_backends
            if result:
                all_results.append(result)
            # load taste & kendall CSVs that were just saved
            tb = output_dir / f"taste_breakdown_{tag}.csv"
            kd = output_dir / f"kendall_{tag}.csv"
            if tb.exists():
                all_taste_dfs[tag] = pd.read_csv(tb)
            if kd.exists():
                all_kendall_dfs[tag] = pd.read_csv(kd)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ Excluyendo {tag}: {e}", file=sys.stderr)
            excluded_variants[tag] = str(e)

    # Resolve sample_hash from sample_manifest.json or analysed prompt IDs
    sample_hash = ""
    manifest_path = output_dir / "sample_manifest.json"
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            sample_hash = str(manifest_data.get("sha1") or manifest_data.get("sample_hash") or "")
        except Exception:
            sample_hash = ""
    if not sample_hash and comp_by_tag:
        first_comp = next(iter(comp_by_tag.values()))
        sample_hash = compute_sample_hash(first_comp["id_prompt"].tolist())

    if sample_hash:
        print(f"  Sample hash (SHA-1): {sample_hash}")
    for r in all_results:
        r["sample_hash"] = sample_hash

    # Check backend consistency across variants
    backend_mismatch_msg = ""
    non_empty_backends = {t: b for t, b in nlp_backends_by_tag.items() if b}
    if len(non_empty_backends) > 1:
        unique_configs = {tuple(sorted(b.items())) for b in non_empty_backends.values()}
        if len(unique_configs) > 1:
            parts = [
                f"`{_short_tag(t).split()[0]}`={b}"
                for t, b in non_empty_backends.items()
            ]
            backend_mismatch_msg = "; ".join(parts)

    # Summary CSV (even for 1 valid variant)
    if all_results:
        summary_df = pd.DataFrame(all_results)
        for col in SUMMARY_COLUMNS:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        summary_df = summary_df[SUMMARY_COLUMNS].round(4)
        print(f"\n{'#' * 60}")
        print(f"  RESUMEN DE VARIANTES — {group_prefix}")
        print(f"{'#' * 60}")
        print(summary_df.to_string(index=False))
        summary_df.to_csv(output_dir / f"summary_{group_prefix}.csv", index=False)

    # Cross-variant comparison
    omnibus_cross: dict = {}
    pair_cross_df = pd.DataFrame()
    omnibus_rep: dict = {}
    pair_rep_df = pd.DataFrame()
    pair_nlp_df = pd.DataFrame()
    if len(all_results) > 1:
        omnibus_rep, pair_rep_df = compare_variants_on_column(
            comp_by_tag, group_prefix, col="clap_score_reprompt"
        )
        if not pair_rep_df.empty:
            print("\n  Contrastes entre variantes sobre CLAP(reprompt, audio_reprompt) [Holm]:")
            if "friedman_p" in omnibus_rep:
                print(
                    f"  Friedman χ²={omnibus_rep['friedman_chi2']:.3f}, "
                    f"p={omnibus_rep['friedman_p']:.4f}, W={omnibus_rep['kendall_w']:.4f}"
                )
            print(
                pair_rep_df[
                    ["pair", "mean_a", "mean_b", "mean_diff", "p_t", "p_t_holm", "p_wilcoxon", "p_w_holm", "cohens_dz"]
                ].to_string(index=False)
            )
            pair_rep_df.to_csv(output_dir / f"pairwise_reprompt_{group_prefix}.csv", index=False)

        omnibus_cross, pair_cross_df = compare_variants_on_cross(comp_by_tag, group_prefix)
        if not pair_cross_df.empty:
            print("\n  Contrastes entre variantes sobre CLAP(prompt_original, audio_reprompt) [Holm]:")
            if "friedman_p" in omnibus_cross:
                print(
                    f"  Friedman χ²={omnibus_cross['friedman_chi2']:.3f}, "
                    f"p={omnibus_cross['friedman_p']:.4f}, W={omnibus_cross['kendall_w']:.4f}"
                )
            print(
                pair_cross_df[
                    ["pair", "mean_a", "mean_b", "mean_diff", "p_t", "p_t_holm", "p_wilcoxon", "p_w_holm", "cohens_dz"]
                ].to_string(index=False)
            )
            pair_cross_df.to_csv(output_dir / f"pairwise_cross_{group_prefix}.csv", index=False)

        # Export combined CLAP + NLP summary and pairwise NLP diversity comparisons
        if nlp_by_tag:
            nlp_rows_summary = []
            for r in all_results:
                tag = r["label"]
                ns = nlp_by_tag.get(tag, {})
                nlp_rows_summary.append(
                    {
                        "variant": _short_tag(tag).split()[0],
                        "label": tag,
                        "clap_reprompt": r["mean_reprompt"],
                        "clap_raw_baseline": r["mean_raw_baseline"],
                        "clap_cross": r.get("mean_cross", np.nan),
                        "has_true_raw_audio": r.get("has_true_raw_audio", False),
                        "words_rep": ns.get("long_rep", np.nan),
                        "ttr": ns.get("ttr", np.nan),
                        "mattr": ns.get("mattr", np.nan),
                        "bigram_entropy": ns.get("entropia", np.nan),
                        "flesch": ns.get("flesch", np.nan),
                        "musical_vocab_density": ns.get("musical_vocab_density", np.nan),
                        "semantic_cos": ns.get("coseno", np.nan),
                        "corpus_distinct_1": ns.get("corpus_distinct_1", np.nan),
                        "corpus_distinct_2": ns.get("corpus_distinct_2", np.nan),
                        "inter_prompt_cos_dist": ns.get("inter_prompt_cos_dist", np.nan),
                        "self_bleu": ns.get("self_bleu", np.nan),
                    }
                )
            pd.DataFrame(nlp_rows_summary).round(4).to_csv(
                output_dir / f"summary_nlp_{group_prefix}.csv", index=False
            )

            nlp_pair_frames = []
            for metric_col in (
                "mattr",
                "distinctiveness_cos_dist",
                "musical_vocab_density",
                "semantic_sim_raw_rep",
            ):
                _, m_df = compare_variants_on_column(comp_by_tag, group_prefix, col=metric_col)
                if not m_df.empty:
                    m_df.insert(0, "metric", metric_col)
                    nlp_pair_frames.append(m_df)
            if nlp_pair_frames:
                pair_nlp_df = pd.concat(nlp_pair_frames, ignore_index=True)
                pair_nlp_df.to_csv(output_dir / f"pairwise_nlp_{group_prefix}.csv", index=False)

        plot_comparison_across_variants(all_results, output_dir, group_prefix)

    # Generate findings markdown
    _generate_findings_report(
        group_prefix,
        all_results,
        all_taste_dfs,
        all_kendall_dfs,
        output_dir,
        omnibus=omnibus_cross,
        pair_df=pair_cross_df,
        omnibus_rep=omnibus_rep,
        pair_rep_df=pair_rep_df,
        nlp_by_tag=nlp_by_tag,
        pair_nlp_df=pair_nlp_df,
        sample_hash=sample_hash,
        excluded_variants=excluded_variants,
        nlp_errors_by_tag=nlp_errors_by_tag,
        backend_mismatch_msg=backend_mismatch_msg,
    )

    print(f"\n  ✓ Análisis completo. Resultados en: {output_dir}")


def _significance_label(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _fmt_num(val: float | object, fmt: str = ".4f") -> str:
    if val is None or pd.isna(val):
        return "—"
    return format(float(val), fmt)


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
    omnibus: dict | None = None,
    pair_df: pd.DataFrame | None = None,
    omnibus_rep: dict | None = None,
    pair_rep_df: pd.DataFrame | None = None,
    nlp_by_tag: dict[str, dict] | None = None,
    pair_nlp_df: pd.DataFrame | None = None,
    sample_hash: str = "",
    excluded_variants: dict[str, str] | None = None,
    nlp_errors_by_tag: dict[str, str] | None = None,
    backend_mismatch_msg: str = "",
) -> None:
    """Generate a hallazgos_<group>.md report with structured findings."""
    lines: list[str] = []
    w = lines.append

    w(f"# {group} — Hallazgos de la Ablación (CLAP + Diversidad NLP)\n")
    w("> Generado automáticamente por `models/ablation_analysis.py`")
    if sample_hash:
        w(f"> Sample hash (SHA-1): `{sample_hash}`")
    if results:
        w(f"> Variantes analizadas: {', '.join(_short_tag(r['label']).split()[0] for r in results)}\n")
    else:
        w("> Variantes analizadas: *(ninguna)*\n")

    if nlp_errors_by_tag or backend_mismatch_msg:
        w("> [!WARNING]")
        if nlp_errors_by_tag:
            err_items = "; ".join(f"`{t}`: {msg}" for t, msg in nlp_errors_by_tag.items())
            w(f"> **Evaluación NLP incompleta o fallida**: {err_items}")
        if backend_mismatch_msg:
            w(f"> **Backends NLP inconsistentes entre variantes**: {backend_mismatch_msg}")
        w("")

    if excluded_variants:
        w("## Variantes excluidas\n")
        w("| Variante / Tag | Motivo |")
        w("|---|---|")
        for ex_tag, reason in excluded_variants.items():
            w(f"| `{ex_tag}` | {reason} |")
        w("\n---\n")

    # ── 0. Joint CLAP + NLP Decision Matrix ──────────────────
    if results:
        w("## 0. Matriz de Decisión Conjunta — CLAP + Diversidad NLP\n")
        w(
            "| Variante | CLAP(reprompt, audio_rep) | CLAP(prompt, audio_prompt) [_raw] | "
            "CLAP(prompt, audio_rep) [_cross] | MATTR-15 | Distinct-1 | Distinct-2 | Self-BLEU-4 ↓ | "
            "Dist. Coseno ↑ | Vocab. Musical | Cos(raw, rep) |"
        )
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in results:
            tag = r["label"]
            ns = (nlp_by_tag or {}).get(tag)
            nlp_err = (nlp_errors_by_tag or {}).get(tag)
            v_short = _short_tag(tag).split()[0]
            raw_cell = _fmt_num(r.get("mean_raw_baseline", r.get("mean_raw")))
            cross_cell = _fmt_num(r.get("mean_cross"))
            if ns is not None and not nlp_err:
                mattr_c = _fmt_num(ns.get("mattr"), ".3f")
                d1_c = _fmt_num(ns.get("corpus_distinct_1"), ".3f")
                d2_c = _fmt_num(ns.get("corpus_distinct_2"), ".3f")
                sb_c = _fmt_num(ns.get("self_bleu"), ".3f")
                cd_c = _fmt_num(ns.get("inter_prompt_cos_dist"), ".3f")
                mus_val = ns.get("musical_vocab_density", np.nan)
                mus_c = f"{float(mus_val) * 100:.1f}%" if pd.notna(mus_val) else "—"
                cos_c = _fmt_num(ns.get("coseno"), ".3f")
            else:
                reason = nlp_err or "sin datos NLP"
                fail_cell = f"— (NLP falló: {reason})"
                mattr_c = d1_c = d2_c = sb_c = cd_c = mus_c = cos_c = fail_cell
            w(
                f"| **{v_short}** | **{_fmt_num(r['mean_reprompt'])}** | {raw_cell} | "
                f"{cross_cell} | {mattr_c} | {d1_c} | {d2_c} | {sb_c} | "
                f"{cd_c} | {mus_c} | {cos_c} |"
            )
        w(
            "\n> **Regla de decisión conjunta**: Seleccionar la variante que maximice "
            "`CLAP(reprompt, audio_reprompt)` (fidelidad música↔reprompt superior al baseline "
            "`CLAP(prompt, audio_prompt)`) y mantenga alta diversidad inter-prompt "
            "(`Distinct-2`, `Dist. Coseno Inter-Prompt`, bajo `Self-BLEU-4`) sin degradar "
            "`CLAP(prompt, audio_reprompt)`.\n"
        )
        w("\n---\n")

    # ── 1A. Cross-variant comparison on CLAP(reprompt, reprompt_audio) ──
    if pair_rep_df is not None and not pair_rep_df.empty:
        w("## 1A. Comparación entre Variantes — CLAP(reprompt ↔ audio del reprompt)\n")
        if omnibus_rep and "friedman_p" in omnibus_rep:
            w(
                f"- **Test omnibus (Friedman, n={omnibus_rep['n']})**: "
                f"χ² = {omnibus_rep['friedman_chi2']:.3f}, p = {omnibus_rep['friedman_p']:.4f} "
                f"({_significance_label(omnibus_rep['friedman_p'])}), Kendall W = {omnibus_rep['kendall_w']:.4f}\n"
            )
        w("| Par | Media A | Media B | Δ (A − B) | t | p (t) | p_Holm (t) | p_Holm (Wilcoxon) | d_z |")
        w("|-----|---------|---------|-----------|---|-------|------------|-------------------|-----|")
        for _, r in pair_rep_df.iterrows():
            sig = _significance_label(r["p_t_holm"])
            w(
                f"| {r['pair']} | {_fmt_num(r['mean_a'])} | {_fmt_num(r['mean_b'])} | {_fmt_num(r['mean_diff'], '+.4f')} | "
                f"{_fmt_num(r['t_stat'])} | {_fmt_num(r['p_t'])} | **{_fmt_num(r['p_t_holm'])}** {sig} | "
                f"{_fmt_num(r['p_w_holm'])} | {_fmt_num(r['cohens_dz'])} |"
            )
        sig_rep_pairs = pair_rep_df[pair_rep_df["p_t_holm"] < 0.05]["pair"].tolist()
        if sig_rep_pairs:
            w(f"\n**Hallazgo en fidelidad al reprompt**: Diferencias significativas tras Holm en: {', '.join(sig_rep_pairs)}.\n")
        else:
            w("\n**Hallazgo en fidelidad al reprompt**: Ningún par difiere significativamente tras corrección de Holm (p_Holm ≥ 0.05).\n")
        w("\n---\n")

    # ── 1B. Cross-variant comparison on CLAP(original_prompt, reprompt_audio) ──
    if pair_df is not None and not pair_df.empty:
        w("## 1B. Comparación entre Variantes — CLAP(prompt original ↔ audio del reprompt)\n")
        if omnibus and "friedman_p" in omnibus:
            w(
                f"- **Test omnibus (Friedman, n={omnibus['n']})**: "
                f"χ² = {omnibus['friedman_chi2']:.3f}, p = {omnibus['friedman_p']:.4f} "
                f"({_significance_label(omnibus['friedman_p'])}), Kendall W = {omnibus['kendall_w']:.4f}\n"
            )
        w("| Par | Media A | Media B | Δ (A − B) | t | p (t) | p_Holm (t) | p_Holm (Wilcoxon) | d_z |")
        w("|-----|---------|---------|-----------|---|-------|------------|-------------------|-----|")
        for _, r in pair_df.iterrows():
            sig = _significance_label(r["p_t_holm"])
            w(
                f"| {r['pair']} | {_fmt_num(r['mean_a'])} | {_fmt_num(r['mean_b'])} | {_fmt_num(r['mean_diff'], '+.4f')} | "
                f"{_fmt_num(r['t_stat'])} | {_fmt_num(r['p_t'])} | **{_fmt_num(r['p_t_holm'])}** {sig} | "
                f"{_fmt_num(r['p_w_holm'])} | {_fmt_num(r['cohens_dz'])} |"
            )
        sig_pairs = pair_df[pair_df["p_t_holm"] < 0.05]["pair"].tolist()
        if sig_pairs:
            w(f"\n**Hallazgo entre variantes**: Diferencias significativas tras Holm en: {', '.join(sig_pairs)}.\n")
        else:
            w("\n**Hallazgo entre variantes**: Ningún par difiere significativamente tras corrección de Holm (p_Holm ≥ 0.05).\n")
        w("\n---\n")

    # ── 1C. Cross-variant comparison on Row-Level NLP Diversity Metrics ──
    if pair_nlp_df is not None and not pair_nlp_df.empty:
        w("## 1C. Contrastes Pareados en Diversidad y Estructura NLP [Holm]\n")
        w("| Métrica | Par | Media A | Media B | Δ (A − B) | p_Holm (t) | p_Holm (Wilcoxon) | d_z |")
        w("|---------|-----|---------|---------|-----------|------------|-------------------|-----|")
        for _, r in pair_nlp_df.iterrows():
            sig = _significance_label(r["p_t_holm"])
            w(
                f"| `{r['metric']}` | {r['pair']} | {_fmt_num(r['mean_a'])} | {_fmt_num(r['mean_b'])} | "
                f"{_fmt_num(r['mean_diff'], '+.4f')} | **{_fmt_num(r['p_t_holm'])}** {sig} | "
                f"{_fmt_num(r['p_w_holm'])} | {_fmt_num(r['cohens_dz'])} |"
            )
        w("\n---\n")

    # ── 2. General stats (within-variant vs baseline raw) ────────
    if results:
        w("## 2. Estadísticas por Variante\n")
        headers = " | ".join(_short_tag(r["label"]) for r in results)
        sep = "|".join("---" for _ in results)

        w("### 2A. reprompt vs baseline (texto y audio distintos)\n")
        w(f"| Métrica | {headers} |")
        w(f"|---------|{sep}|")
        w("| n | " + " | ".join(str(r["n"]) for r in results) + " |")
        w("| Media baseline raw | " + " | ".join(_fmt_num(r["mean_raw_baseline"]) for r in results) + " |")
        w("| Media reprompt | " + " | ".join(_fmt_num(r["mean_reprompt"]) for r in results) + " |")
        w(
            "| Δ (reprompt − raw) | "
            + " | ".join(f"{_fmt_num(r['delta_rep_vs_raw'], '+.4f')} ({r['pct_change']:+.1f}%)" for r in results)
            + " |"
        )
        w("| Estadístico t | " + " | ".join(_fmt_num(r["t_stat"]) for r in results) + " |")
        w(
            "| p (t) | "
            + " | ".join(
                f"**{_fmt_num(r['p_rep_vs_raw'])}** {_significance_label(r['p_rep_vs_raw'])}"
                for r in results
            )
            + " |"
        )
        w(
            "| p (Wilcoxon) | "
            + " | ".join(
                f"{_fmt_num(r['p_wilcoxon_rep_vs_raw'])} {_significance_label(r['p_wilcoxon_rep_vs_raw'])}"
                for r in results
            )
            + " |"
        )
        w("| d_z pareado | " + " | ".join(f"**{_fmt_num(r['dz_rep_vs_raw'])}**" for r in results) + " |")
        w("| Cohen's d | " + " | ".join(_fmt_num(r["cohens_d"]) for r in results) + " |")
        w(
            "| Casos que mejoraron | "
            + " | ".join(f"{r['pct_improved']:.1f}%" for r in results)
            + " |"
        )

        sig = [r for r in results if pd.notna(r["p_rep_vs_raw"]) and r["p_rep_vs_raw"] < 0.05]
        best = max(results, key=lambda r: r["dz_rep_vs_raw"])
        w(
            f"\n**Hallazgo (reprompt vs baseline)**: {len(sig)} de {len(results)} variantes con mejora significativa "
            f"(p < 0.05, sin corrección por comparaciones múltiples). "
            f"Mayor efecto: {_short_tag(best['label'])} (d_z = {best['dz_rep_vs_raw']:.2f}, Cohen's d = {best['cohens_d']:.2f}).\n"
        )

        w("\n### 2B. cross vs baseline (mismo texto)\n")
        w(f"| Métrica | {headers} |")
        w(f"|---------|{sep}|")
        w("| n | " + " | ".join(str(r["n"]) for r in results) + " |")
        w("| Media baseline raw | " + " | ".join(_fmt_num(r["mean_raw_baseline"]) for r in results) + " |")
        w("| Media cross | " + " | ".join(_fmt_num(r.get("mean_cross")) for r in results) + " |")
        w(
            "| Δ (cross − raw) | "
            + " | ".join(_fmt_num(r.get("delta_cross_vs_raw"), "+.4f") for r in results)
            + " |"
        )
        w("| Estadístico t | " + " | ".join(_fmt_num(r.get("t_stat_cross")) for r in results) + " |")
        w(
            "| p (t) | "
            + " | ".join(
                f"{_fmt_num(r.get('p_t_cross_vs_raw'))} {_significance_label(r.get('p_t_cross_vs_raw'))}"
                for r in results
            )
            + " |"
        )
        w(
            "| p (Wilcoxon) | "
            + " | ".join(
                f"**{_fmt_num(r.get('p_wilcoxon_cross_vs_raw'))}** {_significance_label(r.get('p_wilcoxon_cross_vs_raw'))}"
                for r in results
            )
            + " |"
        )
        w("| d_z pareado | " + " | ".join(f"**{_fmt_num(r.get('dz_cross_vs_raw'))}**" for r in results) + " |")
        w(
            "| Casos que mejoraron | "
            + " | ".join(
                f"{r['pct_improved_cross']:.1f}%" if pd.notna(r.get("pct_improved_cross")) else "—"
                for r in results
            )
            + " |"
        )
        w("\n---\n")

    # ── 3. Per-taste breakdown ───────────────────────────────
    if taste_dfs:
        w("## 3. Análisis por Categoría de Taste\n")
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

    # ── 4. Kendall correlations ──────────────────────────────
    if kendall_dfs:
        w("## 4. Correlación Kendall τ por Taste\n")
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
        help="Experiment group to analyse (e.g. B1, B2, B3)",
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
