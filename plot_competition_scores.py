#!/usr/bin/env python3
"""Plot competition scores, thresholds, and optional baselines from a grading report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt

# Add baselines by populating this list with dicts that at least contain
# "label" and "score". You can optionally specify "color".
BASELINES: List[Mapping[str, float | str]] = [
    # Example:
    # {"label": "Baseline Agent", "score": 0.42, "color": "#7f7f7f", "linestyle": "-"},
]

# Per-competition baselines keyed by the competition_id values from the reports.
# Example:
COMPETITION_BASELINES: Dict[str, List[Mapping[str, float | str]]] = {
    "random-acts-of-pizza": [
        {"label": "AB-MCTS", "score": 0.72},
        {"label": "Agent Laboratory", "score": 0.643},
        {"label": "AIDE (o1-preview)", "score": 0.655},
    ]
}

AGENT_LABEL = "Shinka"

THRESHOLD_STYLES = {
    "gold_threshold": {"color": "#d4af37", "label": "Gold Threshold"},
    "silver_threshold": {"color": "#c0c0c0", "label": "Silver Threshold"},
    "bronze_threshold": {"color": "#cd7f32", "label": "Bronze Threshold"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot competition scores from a grading report, including medal thresholds "
            "and optional baselines."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing a *grading_report.json file",
    )
    return parser.parse_args()


def find_report_files(run_dir: Path) -> Sequence[Path]:
    matches = sorted(run_dir.expanduser().resolve().glob("*grading_report.json"))
    if not matches:
        raise FileNotFoundError(
            f"No *grading_report.json file found inside {run_dir!s}."
        )
    return matches


def load_competition_reports(report_path: Path) -> Sequence[Mapping[str, float | str]]:
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    reports = payload.get("competition_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError(
            f"Field 'competition_reports' is missing or empty in {report_path.name}."
        )
    return reports


def validated_baselines() -> Iterable[Mapping[str, float | str]]:
    for baseline in BASELINES:
        label = baseline.get("label")
        score = baseline.get("score")
        if label is None or score is None:
            continue
        yield baseline


def validated_competition_baselines(
    comp_id: str,
) -> Iterable[Mapping[str, float | str]]:
    for baseline in COMPETITION_BASELINES.get(str(comp_id), []):
        label = baseline.get("label")
        score = baseline.get("score")
        if label is None or score is None:
            continue
        yield baseline


def baseline_offsets(count: int, width: float, gap_factor: float = 0.4) -> List[float]:
    if count <= 1:
        return [0.0]
    gap = width * gap_factor
    span = (count - 1) * (width + gap)
    start = -span / 2
    return [start + idx * (width + gap) for idx in range(count)]


def plot_competitions(
    reports: Sequence[Mapping[str, float | str]],
    title: str,
) -> plt.Figure:
    plt.style.use("seaborn-v0_8")

    competition_ids = [
        str(report.get("competition_id", f"Comp {idx + 1}"))
        for idx, report in enumerate(reports)
    ]
    scores = [float(report.get("score", 0.0)) for report in reports]

    per_comp_baselines = [
        list(validated_competition_baselines(comp_id)) for comp_id in competition_ids
    ]
    global_baselines = list(validated_baselines())

    max_per_comp = max((len(entries) for entries in per_comp_baselines), default=0)
    group_gap = 1.1 + 0.2 * max(0, max_per_comp)
    total_tick_groups = len(competition_ids) + len(global_baselines)
    fig_width = max(8, total_tick_groups * 1.2)

    fig, ax = plt.subplots(
        figsize=(fig_width, 6),
        constrained_layout=True,
    )

    x_positions = [idx * group_gap for idx in range(len(competition_ids))]
    tick_positions: List[float] = []
    tick_labels: List[str] = []

    def refresh_ticks() -> None:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    ax.set_ylabel("Score")
    ax.set_title(", ".join(competition_ids) if competition_ids else title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles: Dict[str, Any] = {}

    def register_handle(label: str, artist: Any) -> None:
        if label and label not in legend_handles:
            legend_handles[label] = artist

    per_bar_width = 0.18
    group_gap = max(group_gap, (per_bar_width + 0.08) * (max_per_comp + 1))
    final_width = max(
        8, (len(competition_ids) + len(global_baselines) + 1) * (group_gap * 0.6)
    )
    fig.set_size_inches(final_width, 6)
    x_positions = [idx * group_gap for idx in range(len(competition_ids))]
    tick_positions = []
    tick_labels = []

    per_comp_colors = plt.get_cmap("tab20")
    shinka_color = "#8fe0c4"
    for idx, (comp_id, entries) in enumerate(zip(competition_ids, per_comp_baselines)):
        group_entries = [
            {
                "label": AGENT_LABEL,
                "score": scores[idx],
                "color": shinka_color,
                "alpha": 0.95,
                "is_shinka": True,
            }
        ] + list(entries)

        offsets = baseline_offsets(len(group_entries), per_bar_width, gap_factor=0.2)
        for entry_idx, (offset, entry) in enumerate(zip(offsets, group_entries)):
            color = entry.get(
                "color", per_comp_colors((idx + entry_idx) % per_comp_colors.N)
            )
            score_value = float(entry["score"])
            bar = ax.bar(
                x_positions[idx] + offset,
                score_value,
                width=per_bar_width,
                color=color,
                edgecolor="white",
                alpha=float(entry.get("alpha", 0.85)),
                label=entry.get("label", ""),
            )
            patch = bar.patches[0] if bar.patches else None
            label_text = (
                f"{AGENT_LABEL}: {score_value:.3f}"
                if entry.get("is_shinka")
                else f"{score_value:.3f}"
            )
            ax.bar_label(bar, labels=[label_text], padding=3)
            if patch:
                patch.set_label(entry.get("label", ""))
                register_handle(entry.get("label", ""), patch)
            tick_positions.append(x_positions[idx] + offset)
            tick_labels.append(entry.get("label", ""))

    # Medal thresholds span full width for clarity.
    for threshold_key, style in THRESHOLD_STYLES.items():
        values: List[float] = []
        for report in reports:
            value = report.get(threshold_key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        unique_values: List[float] = []
        for value in values:
            if not any(abs(value - existing) < 1e-9 for existing in unique_values):
                unique_values.append(value)
        for value in unique_values:
            line = ax.axhline(
                value,
                linestyle=(0, (6, 4)),
                linewidth=2,
                color=style["color"],
                alpha=0.85,
                label=style["label"],
            )
            register_handle(style["label"], line)

    if global_baselines:
        baseline_color_cycle = plt.get_cmap("tab10")
        base_gap = group_gap if group_gap > 0 else 1.0
        next_x = (tick_positions[-1] + base_gap) if tick_positions else 0.0
        for color_idx, baseline in enumerate(global_baselines):
            label = baseline["label"]
            score = float(baseline["score"])
            color = baseline.get("color", baseline_color_cycle(color_idx % 10))
            bar = ax.bar(
                next_x,
                score,
                width=per_bar_width,
                color=color,
                edgecolor="white",
                alpha=0.85,
                label=label,
            )
            ax.bar_label(bar, fmt="{:.3f}", padding=3)
            patch = bar.patches[0] if bar.patches else None
            if patch:
                patch.set_label(label)
                register_handle(label, patch)
            tick_positions.append(next_x)
            tick_labels.append(label)
            next_x += base_gap

    refresh_ticks()

    if legend_handles:
        handles = list(legend_handles.values())
        labels = list(legend_handles.keys())
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=False,
        )

    return fig


def main() -> None:
    args = parse_args()
    try:
        report_paths = find_report_files(args.run_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    saved_plots: List[Path] = []
    for report_path in report_paths:
        try:
            reports = load_competition_reports(report_path)
        except ValueError as exc:  # Skip malformed reports but continue others.
            print(exc, file=sys.stderr)
            continue

        title = f"Competition Scores — {report_path.stem}"
        fig = plot_competitions(reports, title=title)
        output_path = report_path.with_suffix(".png")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        saved_plots.append(output_path)
        print(f"Saved plot to {output_path}")

    if not saved_plots:
        print("No valid competition reports were plotted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
