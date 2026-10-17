"""Fill only the fixed-blend depth and ticker-exclusion robustness gaps.

Keep the archived TF-IDF configuration b and 80/20 rank weight. Reproduce all
existing comparator scores exactly before adding missing blend evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

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

OUTPUT_ROOT = PROJECT_ROOT / "results/frozen_blend_depth_ticker"
K_VALUES = (1, 5, 10, 20)


def ticker_hash(tickers):
    return hashlib.sha256(("\n".join(tickers) + "\n").encode()).hexdigest()


def blend_scores(rp, tf):
    return ALPHA * core.average_rank_matrix(rp) + (1.0 - ALPHA) * core.average_rank_matrix(tf)


def summarize(rows):
    result = {"n_transitions": len(rows)}
    for key in ("return_path", "tfidf", "frozen_blend"):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        result[key] = {"mean": float(values.mean()), "population_sd": float(values.std(ddof=0))}
    gains = [r["frozen_blend"] - r["return_path"] for r in rows]
    result.update({
        "blend_wins_over_return_path": sum(g > 0 for g in gains),
        "blend_wins_over_tfidf": sum(r["frozen_blend"] > r["tfidf"] for r in rows),
        "mean_blend_minus_return_path": float(np.mean(gains)),
        "minimum_blend_minus_return_path": min(gains),
        "maximum_blend_minus_return_path": max(gains),
    })
    return result


def run(run_id, *, input_root=None, authority_root=None, output_root=None):
    paths = configure(input_root, authority_root)
    BLEND_SOURCE = paths["02_blend_historical_tuning_modern_frozen.json"]
    DEPTH_SOURCE = paths["03_modern_k_sensitivity.json"]
    TICKER_SOURCE = paths["06_common_share_heuristic_exclusion.json"]
    authorities = {p: verify_artifact(p) for p in (BLEND_SOURCE, DEPTH_SOURCE, TICKER_SOURCE)}
    base, depth, ticker = (authorities[p] for p in (BLEND_SOURCE, DEPTH_SOURCE, TICKER_SOURCE))
    provenance = verify_inputs(base, depth)
    if ticker["provenance"]["sha256"] != depth["provenance"]["sha256"]:
        raise AssertionError("ticker and depth scientific provenance differ")
    if base["summary"]["tfidf_config_frozen_from_historical"] != core.json_safe(CONFIG):
        raise AssertionError("archived configuration is not frozen b")
    if base["summary"]["selected_alpha_return_path"] != ALPHA:
        raise AssertionError("archived weight is not 0.8")
    saved_base = {r["transition"]: r for r in base["per_transition"] if r["phase"] == "modern_frozen_evaluation"}
    saved_depth = {(r["transition"], r["k"], r["method"]): r for r in depth["per_transition"]}
    saved_ticker = {r["transition"]: r for r in ticker["per_transition"]}
    depth_rows, ticker_rows, universe_rows = [], [], []
    for wt, wt1 in core.MODERN_TRANSITIONS:
        frozen = core.build_frozen_transition(wt, wt1, CONFIG)
        td = frozen["transition_data"]
        label = td.label
        full_blend = blend_scores(frozen["rp_scores"], frozen["tf_scores"])
        for k in K_VALUES:
            row = {"transition": label, "k": k, "n_queries": td.n, "n_candidates_per_query": td.n - 1}
            for key, method, scores in (
                ("return_path", "ReturnPath", frozen["rp_scores"]),
                ("tfidf", "TF-IDF", frozen["tf_scores"]),
                ("frozen_blend", "FrozenBlend", full_blend),
            ):
                evaluation = core.evaluate_scores(scores, td.realized, k=k)
                if evaluation["n_queries"] != td.n or evaluation["n_valid"] != td.n:
                    raise AssertionError(f"{label}: undefined or unmatched query")
                row[key] = evaluation["mean_ndcg"]
                if method != "FrozenBlend":
                    old = saved_depth[(label, k, method)]
                    if row[key] != old["mean_ndcg"] or td.n != old["n_universe"]:
                        raise AssertionError(f"depth baseline mismatch: {label} k={k} {method}")
                if k == 5:
                    field = "blend_ndcg" if key == "frozen_blend" else f"{key}_ndcg"
                    if row[key] != saved_base[label][field] or td.n != saved_base[label]["n_universe"]:
                        raise AssertionError(f"primary baseline mismatch: {label} {method}")
            depth_rows.append(row)
        keep = tuple(t for t in td.tickers if not core.is_non_primary_share_heuristic(t))
        excluded = tuple(t for t in td.tickers if core.is_non_primary_share_heuristic(t))
        if len(excluded) != 118:
            raise AssertionError(f"{label}: exclusion count differs from archived 118")
        index = {t: i for i, t in enumerate(td.tickers)}
        idx = np.array([index[t] for t in keep], dtype=np.int64)
        rp = core.return_path_scores(td.ts_feature[idx])
        tf, vocabulary_size = core.fit_tfidf_scores(tuple(td.texts[index[t]] for t in keep), CONFIG["kwargs"])
        realized = td.realized[np.ix_(idx, idx)]
        restricted_blend = blend_scores(rp, tf)
        row = {"transition": label, "k": 5, "n_queries": len(keep), "n_candidates_per_query": len(keep) - 1, "n_excluded": len(excluded), "vocab_size": vocabulary_size}
        for key, scores in (("return_path", rp), ("tfidf", tf), ("frozen_blend", restricted_blend)):
            evaluation = core.evaluate_scores(scores, realized, k=5)
            if evaluation["n_queries"] != len(keep) or evaluation["n_valid"] != len(keep):
                raise AssertionError(f"{label}: restricted query mismatch")
            row[key] = evaluation["mean_ndcg"]
            if key != "frozen_blend":
                if row[key] != saved_ticker[label][f"{key}_after_ndcg"] or len(keep) != saved_ticker[label]["n_after"]:
                    raise AssertionError(f"ticker baseline mismatch: {label} {key}")
        ticker_rows.append(row)
        universe_rows.append({
            "transition": label,
            "primary_sorted_tickers_sha256": ticker_hash(td.tickers),
            "restricted_sorted_tickers_sha256": ticker_hash(keep),
            "excluded_sorted_tickers": list(excluded),
            "primary_n": td.n,
            "restricted_n": len(keep),
            "all_methods_share_queries_and_other_eligible_candidates": True,
        })
        print(f"{label}: all baseline scores replay exactly; ticker blend {row['frozen_blend']:.6f}; depth blend " + "/".join(f"{r['frozen_blend']:.6f}" for r in depth_rows[-4:]), flush=True)
        for r in depth_rows[-4:] + [row]:
            if r["frozen_blend"] <= r["return_path"]:
                print(f"OBSERVED NON-WIN: {r}", flush=True)
    summary = {
        "depth": {str(k): summarize([r for r in depth_rows if r["k"] == k]) for k in K_VALUES},
        "ticker_exclusion": summarize(ticker_rows),
    }
    scripts = (Path(__file__).resolve(), PROJECT_ROOT / "scripts/frozen_blend_inputs.py")
    payload = {
        "schema_version": "frozen-blend-depth-ticker-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "tfidf": core.json_safe(CONFIG), "alpha_return_path": ALPHA, "alpha_tfidf": 1.0 - ALPHA,
            "retuning_performed": False, "k_values": K_VALUES,
            "selection_transitions": base["configuration"]["historical_tuning_transitions"],
            "evaluation_transitions": base["configuration"]["modern_frozen_evaluation_transitions"],
            "metric": base["configuration"]["metric"], "rank_normalization": base["configuration"]["rank_normalization"],
            "universe": base["configuration"]["universe"], "ticker_exclusion_heuristic": ticker["configuration"]["heuristic"],
            "ticker_refit": "refit frozen-b TF-IDF on each restricted corpus; fully rerank all methods on the same restricted candidate set",
        },
        "depth_per_transition": depth_rows,
        "ticker_exclusion_per_transition": ticker_rows,
        "universe_identity": universe_rows,
        "summary": summary,
        "validations": {"primary_k5_all_18_method_transition_scores_replay_exactly": True, "depth_all_48_saved_comparator_scores_replay_exactly": True, "ticker_all_12_saved_comparator_scores_replay_exactly": True, "all_query_counts_match": True, "all_queries_valid": True, "public_extract_checksums_and_original_source_identities_pass": True},
        "provenance": {
            "source_authorities": authority_provenance(authorities),
            "source_manifest_sha256": {rel(BLEND_SOURCE.parent / "SHA256SUMS.txt"): core.sha256_file(BLEND_SOURCE.parent / "SHA256SUMS.txt")},
            "script_sha256": {rel(p): core.sha256_file(p) for p in scripts},
            "input_validation": provenance,
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "polars": pl.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "original_environment": "Not included in the aggregate authority extracts; runtime versions above identify this replay.",
        },
        "interpretation": "Descriptive robustness checks on the six original modern transitions with the previously selected configuration and weight. Ticker syntax is a heuristic, not a definitive common-stock classifier; no independent or IID inference is added.",
    }
    run_dir = core.create_run_directory(Path(output_root or OUTPUT_ROOT), run_id)
    result_path = run_dir / "frozen_blend_depth_ticker.json"
    core.write_json_new(result_path, payload)
    (run_dir / "SHA256SUMS.txt").write_text(f"{core.sha256_file(result_path)}  {result_path.name}\n")
    print(f"Saved {result_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--input-root", type=Path, help="Root containing the frozen results/ data tree (not distributed)")
    parser.add_argument("--authority-root", type=Path, help="Directory of public authority extracts and SHA256SUMS.txt")
    parser.add_argument("--output-root", type=Path, help="Parent for a new immutable output directory")
    args = parser.parse_args()
    run(args.run_id, input_root=args.input_root, authority_root=args.authority_root, output_root=args.output_root)
