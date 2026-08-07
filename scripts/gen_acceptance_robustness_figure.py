#!/usr/bin/env python3
"""Generate the descriptive acceptance-robustness figure.

The two panels deliberately use different eligibility sets:

* Panel (a) uses the feature-time vintage-text universe.
* Panel (b) uses the fixed subset with four consecutive return windows plus
  the target window and feature-time vintage-text eligibility.

Accordingly, the figure supports within-panel method comparisons only.  It
contains no sampling intervals or other inferential decoration.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from plots.style import STYLES, apply_paper_style  # noqa: E402


ROBUST_RUN_ID = "run_20260730T032543Z"
ROBUST_RUN_DIR = (
    BASE / "results" / "acceptance_robustness" / ROBUST_RUN_ID
)
BLEND_INPUT = (
    ROBUST_RUN_DIR / "02_blend_historical_tuning_modern_frozen.json"
)
ROBUST_MANIFEST_INPUT = ROBUST_RUN_DIR / "artifact_manifest.json"
FOLLOWUP_RUN_ID = "run_20260730T065913Z"
FOLLOWUP_RUN_DIR = (
    BASE / "results" / "acceptance_followup" / FOLLOWUP_RUN_ID
)
FOLLOWUP_INPUT = FOLLOWUP_RUN_DIR / "adversarial_followup_census.json"
FOLLOWUP_MANIFEST_INPUT = FOLLOWUP_RUN_DIR / "artifact_manifest.json"
MEMORY_RUN_ID = "run_20260731T131959Z"
MEMORY_RUN_DIR = BASE / "results" / "memory_fusion_audit" / MEMORY_RUN_ID
MEMORY_INPUT = MEMORY_RUN_DIR / "memory_fusion_audit.json"
MEMORY_MANIFEST_INPUT = MEMORY_RUN_DIR / "artifact_manifest.json"

FIGURE_DIR = BASE / "paper" / "figures"
OUT_PDF = FIGURE_DIR / "fig_acceptance_robustness.pdf"
OUT_PNG = FIGURE_DIR / "fig_acceptance_robustness.png"
PROVENANCE_OUT = (
    MEMORY_RUN_DIR / "fig_acceptance_robustness.provenance.json"
)

TRANSITIONS = [
    "W1->W2",
    "W2->W3",
    "W3->W4",
    "W4->W5",
    "W5->W6",
    "W6->W7",
]
DISPLAY_TRANSITIONS = [
    label.replace("W", "").replace("->", "\N{RIGHTWARDS ARROW}")
    for label in TRANSITIONS
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """Return a repository-relative POSIX path for anonymous provenance."""
    return path.resolve().relative_to(BASE.resolve()).as_posix()


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_transition = {row["transition"]: row for row in rows}
    if len(by_transition) != len(rows):
        raise ValueError("Duplicate transition rows in figure input")
    missing = [transition for transition in TRANSITIONS if transition not in by_transition]
    extras = sorted(set(by_transition).difference(TRANSITIONS))
    if missing or extras:
        raise ValueError(f"Unexpected transition coverage: missing={missing}, extras={extras}")
    return [by_transition[transition] for transition in TRANSITIONS]


def _validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    expected = manifest["artifact_sha256"]
    recorded = expected.get(path.name)
    observed = _sha256(path)
    if recorded != observed:
        raise ValueError(
            f"Immutable input hash mismatch for {path.name}: "
            f"manifest={recorded}, observed={observed}"
        )


def _load_panel_a(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in data["per_transition"]
        if row.get("phase") == "modern_frozen_evaluation"
    ]
    rows = _ordered_rows(rows)
    for row in rows:
        if abs(float(row["alpha_return_path"]) - 0.8) > 1e-12:
            raise ValueError("Panel (a) is not the historically frozen 0.8 ReturnPath blend")
        if abs(float(row["alpha_tfidf"]) - 0.2) > 1e-12:
            raise ValueError("Panel (a) is not the historically frozen 0.2 TF-IDF blend")
        if row.get("tfidf_config_id") != "b":
            raise ValueError("Panel (a) does not use the historically selected TF-IDF config")
    return rows


def _load_panel_b(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = data["modern_frozen_alpha"]["per_transition"]
    grouped: dict[str, list[dict[str, Any]]] = {
        transition: [] for transition in TRANSITIONS
    }
    for row in raw_rows:
        if row["transition"] not in grouped:
            raise ValueError(f"Unexpected panel-b transition: {row['transition']}")
        grouped[row["transition"]].append(row)
    rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS:
        values = sorted(
            grouped[transition], key=lambda row: int(row["memory_windows"])
        )
        if [int(row["memory_windows"]) for row in values] != [1, 2, 3, 4]:
            raise ValueError("Panel (b) does not contain M=1,2,3,4")
        if len({int(row["n_universe"]) for row in values}) != 1:
            raise ValueError("Panel (b) universe varies across memory lengths")
        if any(row["n_queries"] != row["n_universe"] for row in values):
            raise ValueError("Panel (b) contains invalid query rows")
        if any(abs(float(row["alpha_return_path"]) - 0.8) > 1e-12 for row in values):
            raise ValueError("Panel (b) does not use the frozen 0.8 blend")
        rows.append(
            {
                "transition": transition,
                "n_universe": int(values[0]["n_universe"]),
                "return_by_memory": [
                    float(row["return_path_ndcg"]) for row in values
                ],
                "blend_by_memory": [
                    float(row["frozen_blend_ndcg"]) for row in values
                ],
            }
        )
    return rows


def _plot_line(
    ax: plt.Axes,
    xs: list[int],
    ys: list[float],
    *,
    color: str,
    linestyle: str,
    marker: str,
    zorder: int,
) -> None:
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.7,
        zorder=zorder,
    )


def _end_label(ax: plt.Axes, y: float, text: str, color: str, dy: float = 0.0) -> None:
    ax.text(
        5.16,
        y + dy,
        text,
        color=color,
        fontsize=6.1,
        ha="left",
        va="center",
        clip_on=False,
    )


def main() -> None:
    blend_data = _load_json(BLEND_INPUT)
    followup_data = _load_json(FOLLOWUP_INPUT)
    memory_data = _load_json(MEMORY_INPUT)
    robust_manifest = _load_json(ROBUST_MANIFEST_INPUT)
    followup_manifest = _load_json(FOLLOWUP_MANIFEST_INPUT)
    memory_manifest = _load_json(MEMORY_MANIFEST_INPUT)
    _validate_manifest(robust_manifest, BLEND_INPUT)
    _validate_manifest(followup_manifest, FOLLOWUP_INPUT)
    _validate_manifest(memory_manifest, MEMORY_INPUT)

    panel_a = _load_panel_a(blend_data)
    panel_b = _load_panel_b(memory_data)

    xs = list(range(len(TRANSITIONS)))
    tfidf = [float(row["tfidf_ndcg"]) for row in panel_a]
    one_a = [float(row["return_path_ndcg"]) for row in panel_a]
    blend = [float(row["blend_ndcg"]) for row in panel_a]
    n_a = [int(row["n_universe"]) for row in panel_a]

    memory_xs = [1, 2, 3, 4]
    memory_rows = [
        list(row["return_by_memory"])
        for row in panel_b
    ]
    memory_blend_rows = [list(row["blend_by_memory"]) for row in panel_b]
    memory_means = [
        sum(values) / len(values)
        for values in zip(*memory_rows)
    ]
    memory_blend_means = [
        sum(values) / len(values)
        for values in zip(*memory_blend_rows)
    ]
    n_b = [int(row["n_universe"]) for row in panel_b]

    apply_paper_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=(3.35, 3.50),
        sharey=True,
        gridspec_kw={"hspace": 0.50},
    )

    return_color, return_ls, return_marker, _ = STYLES["LW"]
    tfidf_color, tfidf_ls, tfidf_marker, _ = STYLES["TF-IDF"]
    blend_color, blend_ls, blend_marker, _ = STYLES["Blend"]

    _plot_line(
        ax_a,
        xs,
        tfidf,
        color=tfidf_color,
        linestyle=tfidf_ls,
        marker=tfidf_marker,
        zorder=2,
    )
    _plot_line(
        ax_a,
        xs,
        one_a,
        color=return_color,
        linestyle=return_ls,
        marker=return_marker,
        zorder=3,
    )
    _plot_line(
        ax_a,
        xs,
        blend,
        color=blend_color,
        linestyle=blend_ls,
        marker=blend_marker,
        zorder=4,
    )

    _end_label(ax_a, blend[-1], "Blend 0.8/0.2", blend_color)
    _end_label(ax_a, one_a[-1], "1-window RP", return_color)
    _end_label(ax_a, tfidf[-1], "TF-IDF", tfidf_color)

    for values in memory_rows:
        ax_b.plot(
            memory_xs,
            values,
            color="0.70",
            linewidth=0.6,
            alpha=0.8,
            zorder=1,
        )
    _plot_line(
        ax_b,
        memory_xs,
        memory_means,
        color=return_color,
        linestyle=return_ls,
        marker=return_marker,
        zorder=4,
    )
    _plot_line(
        ax_b,
        memory_xs,
        memory_blend_means,
        color=blend_color,
        linestyle=blend_ls,
        marker=blend_marker,
        zorder=5,
    )
    ax_b.text(
        4.08,
        memory_means[-1] + 0.003,
        "RP mean",
        color=return_color,
        fontsize=6.1,
        ha="left",
        va="center",
        clip_on=False,
    )
    ax_b.text(
        4.08,
        memory_blend_means[-1] - 0.003,
        "Frozen 80/20",
        color=blend_color,
        fontsize=6.1,
        ha="left",
        va="center",
        clip_on=False,
    )

    ax_a.set_title("(a) Feature-time vintage universe", loc="left", fontweight="bold", pad=3)
    ax_b.set_title(
        "(b) Fixed four-window availability subset",
        loc="left",
        fontweight="bold",
        pad=3,
    )

    ax_a.text(
        0.985,
        0.955,
        f"$N$={min(n_a):,}\N{EN DASH}{max(n_a):,}",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color="0.35",
    )
    ax_b.text(
        0.015,
        0.955,
        f"$N$={min(n_b):,}\N{EN DASH}{max(n_b):,}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color="0.35",
    )

    for ax in (ax_a, ax_b):
        ax.set_ylim(0.27, 0.45)
        ax.set_yticks([0.28, 0.32, 0.36, 0.40, 0.44])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", color="0.86", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    ax_a.set_xlim(-0.18, 6.28)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(DISPLAY_TRANSITIONS)
    ax_b.set_xlim(0.8, 4.35)
    ax_b.set_xticks(memory_xs)
    ax_b.set_xlabel("Consecutive path windows ($M$)")

    fig.text(
        0.025,
        0.52,
        "Census NDCG@5",
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.2,
    )
    fig.text(
        0.55,
        0.006,
        "Universes differ by panel; compare methods within panels.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="0.35",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.145, right=0.83, top=0.955, bottom=0.205)

    fixed_time = dt.datetime(2026, 7, 31, 13, 19, 59, tzinfo=dt.timezone.utc)
    pdf_metadata = {
        "Title": "Descriptive walk-forward robustness checks",
        "Author": "Anonymous",
        "Subject": "Within-universe census NDCG@5 comparisons",
        "Keywords": "return paths, retrieval, robustness",
        "Creator": "scripts/gen_acceptance_robustness_figure.py",
        "CreationDate": fixed_time,
        "ModDate": fixed_time,
    }
    png_metadata = {
        "Software": f"Matplotlib {matplotlib.__version__}",
        "Description": "Descriptive within-panel census NDCG@5 comparisons.",
    }

    fig.savefig(OUT_PDF, format="pdf", metadata=pdf_metadata)
    fig.savefig(OUT_PNG, format="png", dpi=300, metadata=png_metadata)
    plt.close(fig)

    provenance = {
        "schema_version": "acceptance-robustness-figure-v1",
        "run_ids": [ROBUST_RUN_ID, FOLLOWUP_RUN_ID, MEMORY_RUN_ID],
        "script": _relative(Path(__file__)),
        "inputs": {
            _relative(BLEND_INPUT): _sha256(BLEND_INPUT),
            _relative(ROBUST_MANIFEST_INPUT): _sha256(
                ROBUST_MANIFEST_INPUT
            ),
            _relative(FOLLOWUP_INPUT): _sha256(FOLLOWUP_INPUT),
            _relative(FOLLOWUP_MANIFEST_INPUT): _sha256(
                FOLLOWUP_MANIFEST_INPUT
            ),
            _relative(MEMORY_INPUT): _sha256(MEMORY_INPUT),
            _relative(MEMORY_MANIFEST_INPUT): _sha256(
                MEMORY_MANIFEST_INPUT
            ),
        },
        "outputs": {
            _relative(OUT_PDF): _sha256(OUT_PDF),
            _relative(OUT_PNG): _sha256(OUT_PNG),
        },
        "figure_design": {
            "metric": "census NDCG@5",
            "error_bars": False,
            "panel_a_universe": (
                "returns in feature and target windows plus non-empty vintage "
                "10-K Item 1 text at feature time"
            ),
            "panel_b_universe": (
                "identical M=1,2,3,4 subset with four consecutive return "
                "windows, target returns, and feature-time vintage text"
            ),
            "cross_panel_comparability": (
                "none asserted; method comparisons are within panel only"
            ),
            "transitions": TRANSITIONS,
            "panel_a_n": n_a,
            "panel_b_n": n_b,
            "panel_b_memory_windows": memory_xs,
            "panel_b_return_path_macro_means": memory_means,
            "panel_b_frozen_blend_macro_means": memory_blend_means,
        },
    }
    with PROVENANCE_OUT.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Saved {_relative(OUT_PDF)}")
    print(f"Saved {_relative(OUT_PNG)}")
    print(f"Saved {_relative(PROVENANCE_OUT)}")
    print(
        "Panel (a) macro means: "
        f"TF-IDF={sum(tfidf) / len(tfidf):.4f}, "
        f"one-window RP={sum(one_a) / len(one_a):.4f}, "
        f"frozen blend={sum(blend) / len(blend):.4f}"
    )
    print(
        "Panel (b) ReturnPath macro means: "
        + ", ".join(
            f"M={memory}:{value:.4f}"
            for memory, value in zip(memory_xs, memory_means)
        )
    )
    print(
        "Panel (b) frozen-blend macro means: "
        + ", ".join(
            f"M={memory}:{value:.4f}"
            for memory, value in zip(memory_xs, memory_blend_means)
        )
    )


if __name__ == "__main__":
    main()
