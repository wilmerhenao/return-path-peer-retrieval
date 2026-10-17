"""Verify published result checksums, partitions, and aggregate arithmetic.

Uses only the standard library and published summaries. This verifies internal
consistency, not the unshared scientific inputs or a rerun of the experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("return_path", "tfidf", "frozen_blend")
SELECTION = [f"H{i}->H{i+1}" for i in range(1, 13)]
MODERN = [f"W{i}->W{i+1}" for i in range(1, 7)]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equal(actual: float, expected: float) -> None:
    check(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-14),
          f"Numeric mismatch: {actual} != {expected}")


def check_aggregate(rows, summary, suffix, mean_key, sd_key):
    check(len(rows) == summary["n_transitions"], "Transition count mismatch")
    for method in METHODS:
        values = [row[method + suffix] for row in rows]
        equal(statistics.mean(values), summary[method][mean_key])
        equal(statistics.pstdev(values), summary[method][sd_key])
    for other in ("return_path", "tfidf"):
        wins = sum(r["frozen_blend" + suffix] > r[other + suffix] for r in rows)
        check(wins == summary["blend_wins_over_" + other], "Win count mismatch")


def verify(directory: Path) -> dict:
    names = {"full19_frozen_blend.json", "frozen_blend_depth_ticker.json",
             "learned_fusion.json", "cnn_historical_v1.json"}
    manifest = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        check(name not in manifest, "Duplicate manifest entry")
        manifest[name] = digest
    check(set(manifest) == names, "Unexpected result manifest entries")
    data = {}
    for name, digest in manifest.items():
        raw = (directory / name).read_bytes()
        check(hashlib.sha256(raw).hexdigest() == digest, f"Checksum mismatch: {name}")
        data[name] = json.loads(raw)
    full = data["full19_frozen_blend.json"]
    config = full["configuration"]
    check(config["selection_transitions"] == SELECTION, "Selection partition changed")
    check(config["held_out_transitions"] == ["H13->W1"] + MODERN, "Held-out partition changed")
    check(config["alpha_return_path"] == 0.8 and config["retuning_performed"] is False,
          "Frozen configuration changed")
    rows = full["per_transition"]
    check([r["transition"] for r in rows] == SELECTION + ["H13->W1"] + MODERN,
          "Full-panel transition sequence mismatch")
    for r in rows:
        check(r["used_for_configuration_selection"] == (r["transition"] in SELECTION),
              "Incorrect selection label")
        check(r["n_candidates_per_query"] == r["n_queries"] - 1, "Candidate count mismatch")
    groups = {"overall_19": rows,
              "selection_12": [r for r in rows if r["transition"] in SELECTION],
              "held_out_7": [r for r in rows if r["transition"] not in SELECTION],
              "modern_6": [r for r in rows if r["transition"] in MODERN],
              **{g: [r for r in rows if r["regime"] == g]
                 for g in ("HIGH_VOL", "MED_VOL", "LOW_VOL")}}
    for name, subset in groups.items():
        check_aggregate(subset, full["summary"][name], "_ndcg", "macro_mean_ndcg",
                        "population_std_across_transition_means")
    depth = data["frozen_blend_depth_ticker.json"]
    for k in (1, 5, 10, 20):
        subset = [r for r in depth["depth_per_transition"] if r["k"] == k]
        check([r["transition"] for r in subset] == MODERN, "Depth transitions changed")
        check_aggregate(subset, depth["summary"]["depth"][str(k)], "", "mean", "population_sd")
    subset = depth["ticker_exclusion_per_transition"]
    check([r["transition"] for r in subset] == MODERN, "Exclusion transitions changed")
    check_aggregate(subset, depth["summary"]["ticker_exclusion"], "", "mean", "population_sd")
    hgb = data["learned_fusion.json"]
    check(hgb["configuration"]["training_transitions"] == SELECTION, "HGB training partition changed")
    check(hgb["configuration"]["evaluation_transitions"] == MODERN, "HGB evaluation partition changed")
    evaluation = hgb["modern_evaluation"]
    for method, summary in evaluation["summary"]["aggregates"].items():
        values = [r[method + "_ndcg"] for r in evaluation["per_transition"]]
        equal(statistics.mean(values), summary["macro_mean_ndcg"])
        equal(statistics.pstdev(values), summary["population_std_across_transition_means"])
    cnn = data["cnn_historical_v1.json"]
    check(cnn["continuation_gate"]["passed"] is False, "Paper CNN must have stopped at historical gate")
    for width in (32, 64):
        slots = [s for s in cnn["validation_slots"] if s["width"] == width]
        check([s["transition"] for s in slots] == SELECTION[6:], "CNN continuation partition changed")
        for slot in slots:
            seeds = [s for s in cnn["seed_validation_results"]
                     if s["width"] == width and s["transition"] == slot["transition"]]
            check(sorted(s["seed"] for s in seeds) == [11, 29, 47], "CNN seeds changed")
            equal(statistics.mean(s["model_ndcg"] for s in seeds), slot["seed_mean_model_ndcg"])
            equal(slot["seed_mean_model_ndcg"] - slot["baseline_ndcg"], slot["advantage"])
        summary = next(s for s in cnn["width_summaries"] if s["width"] == width)
        equal(statistics.mean(s["advantage"] for s in slots), summary["mean_validation_advantage"])
    selected = [s for s in cnn["validation_slots"] if s["width"] == cnn["selected_width"]]
    return {"checksums": "passed", "aggregate_arithmetic": "passed",
            "selection_transitions": 12, "held_out_transitions": 7,
            "cnn_continuation_transitions_inside_selection": 6,
            "cnn_mean": statistics.mean(s["seed_mean_model_ndcg"] for s in selected),
            "cnn_comparison_blend_mean": statistics.mean(s["baseline_ndcg"] for s in selected)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "paper_results")
    args = parser.parse_args()
    print(json.dumps(verify(args.results_dir), indent=2))
