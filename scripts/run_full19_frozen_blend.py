"""Add the unchanged 80/20 blend to the existing 19-transition census.

Reuse the 12 selection-period and six modern fixed-weight scores after checking
their provenance and exact baseline agreement. Compute only the missing H13->W1
bridge. Preserve prior artifacts, configuration, scorer, and volatility labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts import run_acceptance_robustness as core  # noqa: E402

from scripts.frozen_blend_inputs import (  # noqa: E402
    ALPHA, CONFIG, configure, rel, verify_artifact, verify_inputs, authority_provenance,
)

OUTPUT_ROOT = PROJECT_ROOT / "results/full19_frozen_blend"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_transitions": len(rows)}
    for key in ("return_path", "tfidf", "frozen_blend"):
        values = np.array([row[f"{key}_ndcg"] for row in rows], dtype=np.float64)
        result[key] = {
            "macro_mean_ndcg": float(values.mean()),
            "population_std_across_transition_means": float(values.std(ddof=0)),
        }
    result["blend_wins_over_return_path"] = sum(r["blend_minus_return_path"] > 0 for r in rows)
    result["blend_wins_over_tfidf"] = sum(r["frozen_blend_ndcg"] > r["tfidf_ndcg"] for r in rows)
    result["return_path_wins_over_tfidf"] = sum(r["return_path_ndcg"] > r["tfidf_ndcg"] for r in rows)
    result["macro_mean_blend_minus_return_path"] = float(np.mean([r["blend_minus_return_path"] for r in rows]))
    result["n_query_transition_rows"] = sum(r["n_queries"] for r in rows)
    return result


def run(run_id: str | None = None, *, input_root=None, authority_root=None, output_root=None) -> Path:
    paths = configure(input_root, authority_root)
    BLEND_SOURCE = paths["02_blend_historical_tuning_modern_frozen.json"]
    FULL19_SOURCE = paths["full19_feature_time_census.json"]
    TFIDF_SOURCE = paths["01_tfidf_historical_tuning.json"]
    blend = verify_artifact(BLEND_SOURCE)
    full = verify_artifact(FULL19_SOURCE)
    tfidf_selection = verify_artifact(TFIDF_SOURCE)
    input_validation = verify_inputs(blend, full)
    config_json = core.json_safe(CONFIG)
    if blend["summary"]["selected_alpha_return_path"] != ALPHA:
        raise AssertionError("archived weight is not 0.8")
    if blend["summary"]["tfidf_config_frozen_from_historical"] != config_json:
        raise AssertionError("archived blend configuration differs from fixed b")
    if tfidf_selection["summary"]["selected_config_kwargs"] != config_json["kwargs"]:
        raise AssertionError("archived TF-IDF configuration differs from fixed b")
    if blend["summary"]["modern_data_used_for_selection"] is not False:
        raise AssertionError("selection partition not certified")

    selection = {f"{a}->{b}" for a, b in core.HISTORICAL_TUNING_TRANSITIONS}
    modern = {f"{a}->{b}" for a, b in core.MODERN_TRANSITIONS}
    fixed = [row for row in blend["per_transition"] if row["alpha_return_path"] == ALPHA]
    fixed_by_name = {row["transition"]: row for row in fixed}
    if len(fixed) != 18 or set(fixed_by_name) != selection | modern:
        raise AssertionError("expected exactly 12 selection and six modern fixed rows")
    labels = [row["transition"] for row in full["per_transition"]]
    if len(labels) != 19 or set(labels) != selection | modern | {"H13->W1"}:
        raise AssertionError("full19 transition partition differs")

    # Exact equality plus common input and code hashes validate reused rows.
    for row in full["per_transition"]:
        if row["transition"] not in fixed_by_name:
            continue
        saved = fixed_by_name[row["transition"]]
        if not (
            saved["return_path_ndcg"] == row["return_path"]["mean_ndcg"]
            and saved["tfidf_ndcg"] == row["tfidf"]["mean_ndcg"]
            and saved["n_universe"] == saved["n_queries"] == saved["n_valid"]
            == row["n_feature_time_eligible"] == row["return_path"]["n_valid"]
            == row["tfidf"]["n_valid"]
        ):
            raise AssertionError(f"saved baseline or universe mismatch: {row['transition']}")

    bridge_original = next(r for r in full["per_transition"] if r["transition"] == "H13->W1")
    bridge = core.build_frozen_transition("H13", "W1", CONFIG, oracle_check=True)
    td = bridge["transition_data"]
    for key, current in (("return_path", bridge["rp_eval"]), ("tfidf", bridge["tf_eval"])):
        if current["mean_ndcg"] != bridge_original[key]["mean_ndcg"]:
            raise AssertionError(f"bridge {key} baseline does not replay exactly")
        if current["n_queries"] != td.n or current["n_valid"] != td.n:
            raise AssertionError("bridge query counts disagree")
    if td.n != bridge_original["n_feature_time_eligible"]:
        raise AssertionError("bridge eligible universe differs")
    rp_rank = core.average_rank_matrix(bridge["rp_scores"])
    tf_rank = core.average_rank_matrix(bridge["tf_scores"])
    scores = ALPHA * rp_rank + (1.0 - ALPHA) * tf_rank
    bridge_eval = core.evaluate_scores(scores, td.realized, k=5)
    if bridge_eval["n_queries"] != td.n or bridge_eval["n_valid"] != td.n:
        raise AssertionError("blend queries differ from bridge baselines")

    rows = []
    for original in full["per_transition"]:
        label = original["transition"]
        is_bridge = label == "H13->W1"
        blend_value = bridge_eval["mean_ndcg"] if is_bridge else fixed_by_name[label]["blend_ndcg"]
        rp_value = original["return_path"]["mean_ndcg"]
        rows.append({
            "transition_index": original["transition_index"],
            "transition": label,
            "regime": original["regime"],
            "used_for_configuration_selection": label in selection,
            "phase": "selection_period" if label in selection else "held_out_from_configuration_selection",
            "bridge_transition": is_bridge,
            "source": "new_bridge_computation" if is_bridge else "reused_fixed_weight_score",
            "n_queries": original["n_feature_time_eligible"],
            "n_candidates_per_query": original["n_feature_time_eligible"] - 1,
            "return_path_ndcg": rp_value,
            "tfidf_ndcg": original["tfidf"]["mean_ndcg"],
            "frozen_blend_ndcg": blend_value,
            "blend_minus_return_path": blend_value - rp_value,
        })
    groups = {
        "overall_19": rows,
        **{regime: [r for r in rows if r["regime"] == regime] for regime in ("HIGH_VOL", "MED_VOL", "LOW_VOL")},
        "selection_12": [r for r in rows if r["used_for_configuration_selection"]],
        "held_out_7": [r for r in rows if not r["used_for_configuration_selection"]],
        "modern_6": [r for r in rows if r["transition"] in modern],
    }
    summary = {name: aggregate(subset) for name, subset in groups.items()}
    universe_bytes = ("\n".join(td.tickers) + "\n").encode()
    sources = (BLEND_SOURCE, FULL19_SOURCE, TFIDF_SOURCE)
    payload = {
        "schema_version": "full19-fixed-blend-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "alpha_return_path": ALPHA,
            "alpha_tfidf": 1.0 - ALPHA,
            "tfidf": config_json,
            "selection_transitions": sorted(selection, key=labels.index),
            "held_out_transitions": [x for x in labels if x not in selection],
            "retuning_performed": False,
            "metric": blend["configuration"]["metric"],
            "universe": blend["configuration"]["universe"],
            "rank_normalization": blend["configuration"]["rank_normalization"],
            "regime_definition": full["configuration"]["regime_definition"],
            "aggregation": "unweighted transition means and population SD (ddof=0)",
        },
        "per_transition": rows,
        "summary": summary,
        "bridge_evaluation": {
            "return_path": bridge["rp_eval"],
            "tfidf": bridge["tf_eval"],
            "frozen_blend": bridge_eval,
            "eligible_sorted_tickers": list(td.tickers),
            "eligible_sorted_tickers_sha256": hashlib.sha256(universe_bytes).hexdigest(),
            "vocab_size": bridge["vocab_size"],
        },
        "validations": {
            "public_extract_checksums_and_original_source_identities_pass": True,
            "reused_rows_exact_baseline_and_count_matches": 18,
            "bridge_baselines_replay_exactly": True,
            "identical_queries_and_candidate_sets": True,
            "matched_candidate_rule": "each sorted eligible ticker is a query; every other eligible ticker is a candidate; all methods use identical matrices and canonical self exclusion",
            "all_queries_valid": True,
            "selection_transition_count": len(groups["selection_12"]),
            "held_out_transition_count": len(groups["held_out_7"]),
        },
        "provenance": {
            "source_authorities": authority_provenance(sources),
            "source_manifest_sha256": {rel(p.parent / "SHA256SUMS.txt"): core.sha256_file(p.parent / "SHA256SUMS.txt") for p in sources},
            "script_sha256": {rel(p): core.sha256_file(p) for p in (Path(__file__).resolve(), PROJECT_ROOT / "scripts/frozen_blend_inputs.py")},
            "input_validation": input_validation,
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "polars": pl.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "original_environment": "Not included in the aggregate authority extracts; runtime versions above identify this replay.",
        },
        "interpretation": [
            "The 12 selection transitions selected both TF-IDF configuration and fusion weight; their scores and the mixed 19-transition/regime aggregates are descriptive, not held-out validation.",
            "Seven transitions were outside configuration selection: H13->W1 plus the six original modern transitions. The bridge is newly computed; this is not a preregistered seven-transition experiment.",
            "Historical cohorts, adjacent windows, shared securities, and full-panel volatility labels retain the limitations of the existing census. No IID tests are added.",
        ],
    }
    run_dir = core.create_run_directory(Path(output_root or OUTPUT_ROOT), run_id)
    result_path = run_dir / "full19_frozen_blend.json"
    core.write_json_new(result_path, payload)
    csv_path = run_dir / "per_transition.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "SHA256SUMS.txt").write_text("".join(f"{core.sha256_file(p)}  {p.name}\n" for p in (result_path, csv_path)))
    print(f"Saved: {result_path}")
    for name, result in summary.items():
        values = " | ".join(f"{result[key]['macro_mean_ndcg']:.6f} +/- {result[key]['population_std_across_transition_means']:.6f}" for key in ("return_path", "tfidf", "frozen_blend"))
        print(f"{name} ({result['n_transitions']}): {values}; blend wins {result['blend_wins_over_return_path']}")
    return result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--input-root", type=Path, help="Root containing the frozen results/ data tree (not distributed)")
    parser.add_argument("--authority-root", type=Path, help="Directory of public authority extracts and SHA256SUMS.txt")
    parser.add_argument("--output-root", type=Path, help="Parent for a new immutable output directory")
    args = parser.parse_args()
    run(args.run_id, input_root=args.input_root, authority_root=args.authority_root, output_root=args.output_root)
