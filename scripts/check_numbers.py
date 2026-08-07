"""Validate active manuscript claims against the frozen census artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "main.tex"
ABSTRACT = ROOT / "paper" / "abstract_submitted.tex"
WINDOWS = ROOT / "results" / "windows_v3" / "windows_summary.json"
HISTORICAL_WINDOWS = ROOT / "results" / "historical_windows_summary.json"
FULL_RETURN_RUN = (
    ROOT
    / "results"
    / "full_return_census"
    / "run_20260802T_full_return_authority_v2"
)
ROBUST_RUN = (
    ROOT
    / "results"
    / "acceptance_robustness"
    / "run_20260730T032543Z"
)
FOLLOWUP_RUN = (
    ROOT
    / "results"
    / "acceptance_followup"
    / "run_20260730T065913Z"
)
FULL19_RUN = (
    ROOT
    / "results"
    / "acceptance_full19_feature_time"
    / "run_20260730T032944Z"
)
UNCERTAINTY_RUN = (
    ROOT
    / "results"
    / "acceptance_target_date_uncertainty"
    / "run_20260731T014000Z"
)
TOKEN_RUN = (
    ROOT
    / "results"
    / "granite_vintage_token_audit"
    / "run_20260731T132708Z"
)
LONGCTX_RUN = (
    ROOT
    / "results"
    / "granite_vintage_longctx_census"
    / "granite-longctx-20260730-034718-3825"
)
MEMORY_FUSION_RUN = (
    ROOT
    / "results"
    / "memory_fusion_audit"
    / "run_20260731T131959Z"
)
APPLICATION_AUDIT_RUN = (
    ROOT
    / "results"
    / "strict_feature_time_application"
    / "run_20260801T233929Z"
)
OUTLIER_RUN = (
    ROOT
    / "results"
    / "outlier_sensitivity"
    / "run_20260801T234540Z"
)
LEARNED_FUSION_RUN = (
    ROOT
    / "results"
    / "learned_fusion_followup"
    / "run_20260731T012556Z"
)
GATED_TWO_TOWER_SELECTION = (
    ROOT
    / "experiments"
    / "multimodal_peer_encoder"
    / "artifacts"
    / "runs"
    / "run_20260802_v1"
    / "historical_selection.json"
)
SAME_WINDOW_RUN = (
    ROOT
    / "results"
    / "same_window_text_diagnostic"
    / "run_20260802T_same_window_authority_v2"
)
PAIRWISE_ALIGNMENT_RUN = (
    ROOT
    / "results"
    / "pairwise_alignment"
    / "run_20260806T_pairwise_authority_v3"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def active_tex(text: str) -> str:
    """Remove comments and simple ``\\iffalse ... \\fi`` legacy blocks."""
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\iffalse\b.*?\\fi\b", "", text, flags=re.DOTALL)
    return "\n".join(
        re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines()
    )


def fmt3(value: float) -> str:
    return format(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
        "f",
    )


def fmt1_pct(value: float) -> str:
    return format(
        Decimal(str(100.0 * value)).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        ),
        "f",
    )


def fmt2_pct(value: float) -> str:
    return format(
        Decimal(str(100.0 * value)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ),
        "f",
    )


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.pstdev(values)


def tex_int(value: int) -> str:
    """Format any integer with TeX-safe grouping separators."""
    return f"{value:,}".replace(",", "{,}")


def verify_sha256_manifest(directory: Path) -> list[str]:
    failures: list[str] = []
    manifest = directory / "SHA256SUMS.txt"
    if not manifest.is_file():
        return [f"checksum manifest missing: {manifest.relative_to(ROOT)}"]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            failures.append(
                f"malformed checksum line in {manifest.relative_to(ROOT)}: {line!r}"
            )
            continue
        expected, filename = match.groups()
        path = directory / filename
        if not path.is_file():
            failures.append(f"checksummed artifact missing: {path.relative_to(ROOT)}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"checksum mismatch: {path.relative_to(ROOT)}")
    return failures


def main() -> int:
    text = active_tex(PAPER.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: list[tuple[str, str]] = []

    for directory in (
        ROBUST_RUN,
        FOLLOWUP_RUN,
        FULL19_RUN,
        UNCERTAINTY_RUN,
        TOKEN_RUN,
        LONGCTX_RUN,
        MEMORY_FUSION_RUN,
        APPLICATION_AUDIT_RUN,
        OUTLIER_RUN,
        LEARNED_FUSION_RUN,
        SAME_WINDOW_RUN,
        PAIRWISE_ALIGNMENT_RUN,
        FULL_RETURN_RUN,
    ):
        failures.extend(verify_sha256_manifest(directory))

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        text,
        flags=re.DOTALL,
    )
    standalone_abstract = active_tex(ABSTRACT.read_text(encoding="utf-8"))
    normalize = lambda value: re.sub(r"\s+", " ", value).strip()
    if (
        abstract_match is None
        or normalize(abstract_match.group(1)) != normalize(standalone_abstract)
    ):
        failures.append("standalone abstract does not match main.tex")

    def require(label: str, literal: str) -> None:
        checks.append((label, literal))
        if literal not in text:
            failures.append(f"{label}: expected literal {literal!r}")

    def forbid(label: str, literal: str) -> None:
        if literal in text:
            failures.append(f"{label}: stale literal remains active: {literal!r}")

    # Window construction and full-return census.
    windows = load_json(WINDOWS)
    ordered = [windows[f"W{i}"] for i in range(1, 8)]
    for left, right in zip(ordered, ordered[1:]):
        if left["end"] >= right["start"]:
            failures.append(
                f"overlap: {left['window']} ends {left['end']} and "
                f"{right['window']} starts {right['start']}"
            )
    n_values = [row["N"] for row in ordered]
    t_values = [row["T"] for row in ordered]
    require("return-window N range", f"{min(n_values):,}--{max(n_values):,}")
    require("return-path dimension range", f"{min(t_values)}--{max(t_values)}")
    require("W7 start", ordered[-1]["start"])
    require("W7 end", ordered[-1]["end"])

    historical_windows = load_json(HISTORICAL_WINDOWS)
    historical_summary = historical_windows["summary"]
    if int(historical_summary["n_windows"]) != 13:
        failures.append("historical window summary does not contain H1--H13")
    require(
        "historical return-window N range",
        (
            f"N={tex_int(int(historical_summary['N_min']))}$--$"
            f"{tex_int(int(historical_summary['N_max']))}"
        ),
    )
    require(
        "historical return-path dimension range",
        (
            f"T={int(historical_summary['T_min'])}$--$"
            f"{int(historical_summary['T_max'])}"
        ),
    )
    require(
        "full-panel shrinkage range",
        (
            f"[{float(historical_summary['lw_shrinkage_min']):.2f},"
            f"{float(historical_summary['lw_shrinkage_max']):.2f}]"
        ),
    )

    full_return = load_json(FULL_RETURN_RUN / "full_return_census.json")
    full_return_rows = full_return["per_transition"]

    def full_return_values(method: str) -> list[dict]:
        rows = [row for row in full_return_rows if row["method"] == method]
        if len(rows) != 6:
            failures.append(
                f"full-return/{method}: expected 6 census rows, found {len(rows)}"
            )
        return rows

    full_query_n = [
        int(row["n_queries"])
        for row in full_return_values("ReturnPath")
    ]
    require(
        "full-return query range",
        f"{min(full_query_n):,}--{max(full_query_n):,}",
    )
    for label, method in (
        ("full return path", "ReturnPath"),
        ("spectral dot", "Spectral-dot-128"),
        ("SIC-2", "SIC-2"),
        ("random", "Random"),
    ):
        rows = full_return_values(method)
        mean, sd = mean_sd([float(row["mean_ndcg"]) for row in rows])
        require(label, f"{fmt3(mean)}\\pm{fmt3(sd)}")
    spectral_dot = full_return["summary"]["aggregates"]["Spectral-dot-128"]
    spectral_cos = full_return["summary"]["aggregates"]["Spectral-cos-128"]
    if abs(
        float(spectral_dot["macro_mean_ndcg"])
        - float(spectral_cos["macro_mean_ndcg"])
    ) > 0.002:
        failures.append("spectral cosine is no longer within 0.002 of spectral dot")

    # Historical-only selection and frozen modern feature-time evaluation.
    tfidf_selection = load_json(
        ROBUST_RUN / "01_tfidf_historical_tuning.json"
    )
    selection_summary = tfidf_selection["summary"]
    if selection_summary["selected_config_id"] != "b":
        failures.append("historical TF-IDF selection is not config b")
    if selection_summary["modern_data_used_for_selection"] is not False:
        failures.append("TF-IDF selection used modern data")
    require("selected TF-IDF size", "50{,}000")

    blend = load_json(
        ROBUST_RUN / "02_blend_historical_tuning_modern_frozen.json"
    )
    blend_summary = blend["summary"]
    if blend_summary["modern_data_used_for_selection"] is not False:
        failures.append("blend selection used modern data")
    if blend_summary["selected_alpha_return_path"] != 0.8:
        failures.append("frozen return-path blend weight is not 0.8")
    require("frozen blend weight", r"\alpha=0.8")

    modern = [
        row
        for row in blend["per_transition"]
        if row["phase"] == "modern_frozen_evaluation"
    ]
    if len(modern) != 6:
        failures.append(f"expected 6 frozen modern transitions, found {len(modern)}")
    modern_n = [int(row["n_universe"]) for row in modern]
    require(
        "feature-time N range",
        f"{min(modern_n):,}--{max(modern_n):,}",
    )
    for label, key in (
        ("feature-time return path", "return_path_ndcg"),
        ("feature-time TF-IDF", "tfidf_ndcg"),
        ("frozen blend", "blend_ndcg"),
    ):
        mean, sd = mean_sd([float(row[key]) for row in modern])
        require(label, f"{fmt3(mean)}\\pm{fmt3(sd)}")
    gains = [float(row["blend_minus_return_path"]) for row in modern]
    require("mean blend gain", f"{fmt3(statistics.fmean(gains))}")
    require("minimum blend gain", f"+{fmt3(min(gains))}")
    require("maximum blend gain", f"+{fmt3(max(gains))}")
    if not all(row["improves_return_path"] for row in modern):
        failures.append("frozen blend does not improve all modern transitions")

    followup = load_json(FOLLOWUP_RUN / "adversarial_followup_census.json")

    # All short-context dense encoders are rescored on the primary feature-time
    # universe. Cached embeddings are fixed; only eligibility and NDCG change.
    dense = followup["dense_feature_time_census"]
    if dense["summary"]["n_transitions"] != 6:
        failures.append("feature-time dense census does not contain six transitions")
    dense_aggregates = dense["summary"]["aggregates"]
    for label, method in (
        ("Granite-r2", "Granite-r2 (512 tok.)"),
        ("SEC-BERT", "SEC-BERT"),
        ("BGE", "BGE"),
        ("MiniLM", "MiniLM"),
    ):
        values = dense_aggregates[method]
        require(
            label,
            (
                f"{fmt3(values['macro_mean_ndcg'])}"
                f"\\pm{fmt3(values['population_std_across_transition_means'])}"
            ),
        )

    # Contemporaneous text is evaluated on the same feature-time universes and
    # frozen representations as the primary walk-forward comparison.  This
    # prevents a coverage or configuration change from masquerading as a
    # timing result.
    same_window = load_json(
        SAME_WINDOW_RUN / "same_window_text_diagnostic.json"
    )
    if not all(
        same_window["validations"][key]
        for key in (
            "same_security_universe_as_walk_forward",
            "all_eligible_securities_are_queries",
            "target_window_text_not_loaded",
            "return_path_same_window_identity",
        )
    ):
        failures.append("same-window diagnostic protocol validation failed")
    same_rows = same_window["per_transition"]
    same_methods = (
        "TF-IDF",
        "Granite-r2 (512 tok.)",
        "MiniLM",
    )
    modern_n_by_transition = {
        row["transition"]: int(row["n_universe"]) for row in modern
    }
    for method in same_methods:
        method_rows = [row for row in same_rows if row["method"] == method]
        if len(method_rows) != 6:
            failures.append(
                f"same-window {method}: expected six transitions, "
                f"found {len(method_rows)}"
            )
            continue
        if any(
            int(row["n_universe"])
            != modern_n_by_transition.get(row["transition"])
            for row in method_rows
        ):
            failures.append(
                f"same-window {method}: universe differs from walk-forward"
            )
        same_mean, same_sd = mean_sd(
            [float(row["mean_ndcg"]) for row in method_rows]
        )
        require(
            f"same-window {method}",
            f"{fmt3(same_mean)}\\pm{fmt3(same_sd)}",
        )
        summary_row = same_window["summary"][method]
        if abs(
            float(summary_row["same_window_macro_mean_ndcg"]) - same_mean
        ) > 1e-12:
            failures.append(f"same-window {method}: summary mean mismatch")
    largest_same_window_gap = max(
        abs(float(row["same_window_minus_walk_forward"]))
        for row in same_window["summary"].values()
    )
    require(
        "largest same-window/walk-forward gap",
        f"{largest_same_window_gap:.4f}",
    )

    # The pairwise coefficient is a complete upper-triangle descriptive
    # statistic.  Pair-level p-values are intentionally absent because pairs
    # share names and target observations.
    pairwise = load_json(PAIRWISE_ALIGNMENT_RUN / "pairwise_alignment.json")
    pair_n = int(pairwise["n_securities"])
    pair_count = int(pairwise["n_unordered_pairs"])
    if pair_count != pair_n * (pair_n - 1) // 2:
        failures.append("pairwise alignment count identity failed")
    if not all(
        pairwise["validations"][key]
        for key in (
            "feature_time_universe",
            "target_window_text_not_loaded",
            "complete_pair_census",
            "pair_count_identity",
            "same_window_pair_count_identity",
            "dense_cache_indexed_to_full_feature_universe",
        )
    ):
        failures.append("pairwise alignment protocol validation failed")
    require("pairwise alignment N", tex_int(pair_n))
    require("pairwise alignment pair count", tex_int(pair_count))
    require(
        "pairwise TF-IDF Spearman",
        f"{float(pairwise['tfidf']['spearman_rho']):.3f}",
    )
    require(
        "same-window TF-IDF Spearman",
        f"{float(pairwise['tfidf']['same_window_spearman_rho']):.3f}",
    )
    require(
        "pairwise return-path Spearman",
        f"{float(pairwise['return_path']['spearman_rho']):.3f}",
    )
    require(
        "same-window return-path Spearman",
        f"{float(pairwise['return_path']['same_window_spearman_rho']):.3f}",
    )
    dense_pairwise = pairwise["dense_text_encoders"]
    if set(dense_pairwise) != {
        "BGE",
        "Granite-r2 (512 tok.)",
        "MiniLM",
        "SEC-BERT",
    }:
        failures.append("pairwise dense-encoder set is not the frozen four")
    dense_same = [
        float(result["same_window_spearman_rho"])
        for result in dense_pairwise.values()
    ]
    dense_next = [
        float(result["spearman_rho"])
        for result in dense_pairwise.values()
    ]
    for label, value in (
        ("dense same-window Spearman minimum", min(dense_same)),
        ("dense same-window Spearman maximum", max(dense_same)),
        ("dense next-window Spearman minimum", min(dense_next)),
        ("dense next-window Spearman maximum", max(dense_next)),
    ):
        require(label, f"{value:.3f}")

    # Granite-r2 over the full stored excerpt on the primary feature-time universe.
    longctx = load_json(LONGCTX_RUN / "granite_longctx_census.json")
    longctx_rows = [
        row
        for row in longctx["per_transition"]
        if row["protocol"] == "feature_time_only"
    ]
    if len(longctx_rows) != 6:
        failures.append(
            f"expected 6 full-stored-excerpt Granite transitions, found {len(longctx_rows)}"
        )
    modern_by_transition = {row["transition"]: row for row in modern}
    for row in longctx_rows:
        transition = row["transition"]
        modern_row = modern_by_transition.get(transition)
        if modern_row is None:
            failures.append(f"full-stored-excerpt Granite transition absent: {transition}")
            continue
        if int(row["universe_n"]) != int(modern_row["n_universe"]):
            failures.append(
                f"full-stored-excerpt Granite universe mismatch for {transition}"
            )
        if float(row["granite_longctx_ndcg5"]) >= float(
            modern_row["tfidf_ndcg"]
        ):
            failures.append(
                f"full-stored-excerpt Granite no longer trails TF-IDF for {transition}"
            )
    longctx_mean, longctx_sd = mean_sd(
        [float(row["granite_longctx_ndcg5"]) for row in longctx_rows]
    )
    require(
        "full-stored-excerpt Granite-r2",
        f"{fmt3(longctx_mean)}\\pm{fmt3(longctx_sd)}",
    )
    if (
        longctx["summary"]["feature_time_only"]["return_path_transition_wins"]
        != 6
    ):
        failures.append("return path does not lead full-stored-excerpt Granite in 6/6")
    feature_time_granite = {
        row["transition"]: float(
            row["ndcg_by_method"]["Granite-r2 (512 tok.)"]
        )
        for row in dense["per_transition"]
    }
    if not all(
        float(row["granite_longctx_ndcg5"])
        > feature_time_granite[row["transition"]]
        for row in longctx_rows
    ):
        failures.append(
            "full-stored-excerpt Granite does not improve the 512-token result in all six"
        )

    # Depth, recency, memory, stale-path control, ticker syntax, and eligibility.
    k_artifact = load_json(ROBUST_RUN / "03_modern_k_sensitivity.json")
    aggregates = k_artifact["summary"]["aggregates"]
    by_method = {
        method: [
            row["macro_mean_ndcg"]
            for row in aggregates
            if row["method"] == method
        ]
        for method in ("ReturnPath", "TF-IDF")
    }
    require(
        "return-path depth sensitivity",
        "/".join(fmt3(value) for value in by_method["ReturnPath"]),
    )
    require(
        "TF-IDF depth sensitivity",
        "/".join(fmt3(value) for value in by_method["TF-IDF"]),
    )

    ewma = load_json(ROBUST_RUN / "04_fixed_ewma_return_path.json")["summary"]
    require("20-day half-life", f"{fmt3(ewma['mean_ewma_hl20_ndcg'])}")
    require("40-day half-life", f"{fmt3(ewma['mean_ewma_hl40_ndcg'])}")

    uncertainty = load_json(
        UNCERTAINTY_RUN / "target_date_block_bootstrap.json"
    )
    uncertainty_config = uncertainty["configuration"]
    n_replicates = int(uncertainty_config["replicates_per_block_length"])
    if n_replicates != 1000:
        failures.append(
            f"target-date sensitivity has {n_replicates} rather than 1000 replicates"
        )
    block_lengths = sorted(
        int(value)
        for value in uncertainty_config["block_lengths_trading_days"]
    )
    if block_lengths != [5, 10]:
        failures.append(
            f"unexpected target-date block lengths: {block_lengths}"
        )
    require("target-date replicates", tex_int(n_replicates))
    for gap, label in (
        ("return_path_minus_tfidf", "target-date return-path gap"),
        ("blend_minus_return_path", "target-date blend gap"),
    ):
        summaries = [
            uncertainty["block_bootstrap"][str(length)][
                "macro_gap_summaries"
            ][gap]
            for length in block_lengths
        ]
        lower = min(
            row[
                "conditional_percentile_interval_95_not_confidence_interval"
            ][0]
            for row in summaries
        )
        upper = max(
            row[
                "conditional_percentile_interval_95_not_confidence_interval"
            ][1]
            for row in summaries
        )
        require(label, f"[{fmt3(lower)},{fmt3(upper)}]")
        if any(
            uncertainty["block_bootstrap"][str(length)][
                "fraction_all_six_transition_gaps_positive"
            ][gap]
            != 1.0
            for length in block_lengths
        ):
            failures.append(
                f"{gap}: not every transition gap is positive in every replicate"
            )

    memory = followup["modern_memory_curve"]["summary"]
    memory_values = memory["aggregates"]
    require(
        "path-memory curve",
        "/".join(fmt3(row["macro_mean_ndcg"]) for row in memory_values),
    )
    memory_n_min = memory["n_universe_min"]
    memory_n_max = memory["n_universe_max"]
    require(
        "path-memory N range",
        f"{tex_int(memory_n_min)}$--${tex_int(memory_n_max)}",
    )
    require(
        "four-window path",
        (
            f"{fmt3(memory_values[-1]['macro_mean_ndcg'])}"
            f"\\pm{fmt3(memory_values[-1]['population_std_across_transition_means'])}"
        ),
    )
    if (
        not memory["identical_subset_across_memory_lengths_within_transition"]
        or any(row.get("wins_over_previous_memory") != 6 for row in memory_values[1:])
    ):
        failures.append("path-memory increments do not improve the fixed subset 6/6")

    memory_fusion = load_json(MEMORY_FUSION_RUN / "memory_fusion_audit.json")
    if memory_fusion["anti_leakage"][
        "modern_targets_used_for_alpha_or_memory_selection"
    ] is not False:
        failures.append("memory-fusion audit used modern targets for selection")
    memory_fusion_rows = memory_fusion["modern_frozen_alpha"]["summary"][
        "aggregates"
    ]
    require(
        "matched memory path curve",
        "/".join(
            fmt3(row["return_path_macro_mean_ndcg"])
            for row in memory_fusion_rows
        ),
    )
    require(
        "matched memory blend curve",
        "/".join(
            fmt3(row["frozen_blend_macro_mean_ndcg"])
            for row in memory_fusion_rows
        ),
    )
    require(
        "memory-fusion gain curve",
        "/".join(
            ("+" if row["macro_mean_blend_minus_return_path"] >= 0 else "")
            + fmt3(row["macro_mean_blend_minus_return_path"])
            for row in memory_fusion_rows
        ),
    )
    expected_memory_wins = [6, 6, 5, 0]
    observed_memory_wins = [
        int(row["blend_wins_over_return_path"])
        for row in memory_fusion_rows
    ]
    if observed_memory_wins != expected_memory_wins:
        failures.append(
            f"unexpected memory-fusion win curve: {observed_memory_wins}"
        )
    historical_memory = memory_fusion["historical_memory_specific_alpha"]
    selected_alphas = [
        float(row["selected_alpha_return_path"])
        for row in historical_memory["by_memory"]
    ]
    if selected_alphas != [0.8, 0.8, 1.0, 1.0]:
        failures.append(
            f"unexpected historical memory-specific alphas: {selected_alphas}"
        )
    if not historical_memory["common_four_window_availability_universe"]:
        failures.append("historical memory-specific alpha universes are not common")

    application = load_json(
        APPLICATION_AUDIT_RUN / "strict_application_census.json"
    )
    application_anti_leakage = application["anti_leakage"]
    if application_anti_leakage[
        "selection_universe_uses_target_availability"
    ] is not False:
        failures.append("strict application selection uses target availability")
    if application_anti_leakage[
        "missing_target_names_trigger_reranking"
    ] is not False:
        failures.append("strict application audit reranks around missing targets")
    application_summary = {
        row["method"]: row
        for row in application["macro_summary"]
    }
    application_methods = (
        "ReturnPath",
        "TF-IDF",
        "FrozenBlend",
        "SIC-2",
        "Random",
    )
    if set(application_summary) != set(application_methods):
        failures.append(
            f"unexpected strict application methods: {sorted(application_summary)}"
        )
    else:
        with gzip.open(
            APPLICATION_AUDIT_RUN / "strict_application_per_query.csv.gz",
            mode="rt",
            encoding="utf-8",
            newline="",
        ) as handle:
            application_query_rows = list(csv.DictReader(handle))
        application_transitions = tuple(
            f"W{i}->W{i + 1}" for i in range(1, 7)
        )
        positive_share_by_method: dict[str, float] = {}
        positive_share_by_transition: dict[tuple[str, str], float] = {}
        for method in application_methods:
            transition_shares = []
            for transition in application_transitions:
                values = [
                    float(row["query_basket_correlation"])
                    for row in application_query_rows
                    if row["method"] == method
                    and row["transition"] == transition
                    and row["evaluable"] == "True"
                ]
                if not values:
                    failures.append(
                        "strict application per-query artifact has no "
                        f"evaluable rows for {method} {transition}"
                    )
                    continue
                share = sum(value > 0.0 for value in values) / len(values)
                transition_shares.append(share)
                positive_share_by_transition[(transition, method)] = share
            if len(transition_shares) == len(application_transitions):
                positive_share_by_method[method] = statistics.fmean(
                    transition_shares
                )
        require(
            "application basket correlations",
            "/".join(
                fmt3(
                    application_summary[method][
                        "macro_equal_transition_mean_query_basket_correlation"
                    ]
                )
                for method in application_methods
            ),
        )
        coverage = [
            application_summary[method][
                "macro_equal_transition_evaluation_coverage"
            ]
            for method in application_methods
        ]
        require(
            "strict application coverage",
            f"{fmt1_pct(min(coverage))}--{fmt1_pct(max(coverage))}\\%",
        )
        display_names = {
            "ReturnPath": "Return path",
            "TF-IDF": "TF-IDF",
            "FrozenBlend": "Frozen blend",
            "SIC-2": "SIC-2",
            "Random": "Random",
        }
        for method in application_methods:
            summary = application_summary[method]
            basket = fmt3(
                summary[
                    "macro_equal_transition_mean_query_basket_correlation"
                ]
            )
            positive = fmt1_pct(positive_share_by_method[method])
            coverage_value = fmt1_pct(
                summary["macro_equal_transition_evaluation_coverage"]
            )
            if method == "FrozenBlend":
                basket = f"\\textbf{{{basket}}}"
                positive = f"\\textbf{{{positive}\\%}}"
            else:
                positive = f"{positive}\\%"
            require(
                f"strict application table row {method}",
                (
                    f"{display_names[method]} & {basket} & {positive} & "
                    f"{coverage_value}\\%"
                ),
            )
        if set(positive_share_by_method) == set(application_methods):
            if not all(
                positive_share_by_method["FrozenBlend"]
                > positive_share_by_method[standalone]
                for standalone in ("ReturnPath", "TF-IDF", "SIC-2", "Random")
            ):
                failures.append(
                    "frozen blend no longer has the highest positive-list share"
                )
            if not all(
                positive_share_by_transition[(transition, "FrozenBlend")]
                > positive_share_by_transition[(transition, standalone)]
                for transition in application_transitions
                for standalone in ("ReturnPath", "TF-IDF", "SIC-2", "Random")
            ):
                failures.append(
                    "frozen blend no longer has the highest positive-list share 6/6"
                )
    equal_weight_transition = application["per_transition"]
    application_by_transition = {
        (row["transition"], row["method"]): row
        for row in equal_weight_transition
    }
    if not all(
        application_by_transition[(transition, "FrozenBlend")][
            "mean_query_basket_correlation"
        ]
        > application_by_transition[(transition, standalone)][
            "mean_query_basket_correlation"
        ]
        for transition in (f"W{i}->W{i + 1}" for i in range(1, 7))
        for standalone in ("ReturnPath", "TF-IDF", "SIC-2", "Random")
    ):
        failures.append(
            "frozen blend no longer leads every basket comparator 6/6"
        )
    require(
        "strict application target timing",
        "Lists are fixed before target returns",
    )
    require(
        "conditional target-evaluability label",
        "Eval. is the share with observed target\nreturns, conditional on",
    )
    require(
        "deployment-coverage caveat",
        "not all firms a deployed system might encounter",
    )
    require(
        "independent-outcome caveat",
        "It contains no analyst decisions, independently labeled\n"
        "shared drivers, or economic outcomes",
    )
    require(
        "illustrative review-budget caveat",
        "illustrative analyst workload",
    )
    require(
        "complete-query aggregation hierarchy",
        "average over all eligible queries, then weight the\ntransition means equally",
    )
    require(
        "path-memory construction",
        "For $M$ periods of return history ending at $W_t$, we concatenate paths\n"
        "standardized separately within each period",
    )
    require(
        "application basket construction",
        "daily mean of its five peers' raw FF5 residual paths",
    )
    require(
        "feature-window regime assignment",
        "Each transition is labeled by the annualized volatility, during $W_t$",
    )
    require(
        "distinct Spearman notation",
        "correlation $\\rho_S$",
    )
    if "\\bar\\rho" in text or "\\overline\\rho" in text:
        failures.append("correlation notation again overloads rho with an overbar")

    outlier = load_json(
        OUTLIER_RUN / "outlier_sensitivity_census.json"
    )
    outlier_validations = outlier["validations"]
    if not outlier_validations["unscreened_reconciliation"]["passes"]:
        failures.append("outlier audit no longer reconciles to frozen baseline")
    if not all(
        outlier_validations[key]
        for key in (
            "all_eligible_securities_are_queries_after_each_screen",
            "all_transition_screen_cells_present",
            "five_unique_peers_per_query",
            "query_excluded_from_every_retrieved_peer_set",
            "stored_target_correlation_absent_from_inputs",
            "target_relevance_recomputed_after_every_screen",
            "tfidf_refit_after_every_screen",
        )
    ):
        failures.append("outlier audit validation flags no longer all pass")
    interpretation = outlier["anti_leakage_and_interpretation"]
    if (
        interpretation["either_window_screens_use_W_t1_values_for_eligibility"]
        is not True
        or interpretation["ranking_rebuilt_within_every_screen"] is not True
        or interpretation["target_return_availability_used_for_base_eligibility"]
        is not True
    ):
        failures.append("outlier audit timing or reranking interpretation changed")

    outlier_screens = outlier["screen_summaries"]
    expected_screens = {
        "unscreened",
        "feature_max_abs_le_1p0",
        "either_window_max_abs_le_1p0",
        "feature_max_abs_le_0p5",
        "either_window_max_abs_le_0p5",
    }
    if set(outlier_screens) != expected_screens:
        failures.append(
            f"unexpected outlier screens: {sorted(outlier_screens)}"
        )
    primary_outlier = outlier_screens["either_window_max_abs_le_1p0"]
    primary_universe = primary_outlier["universe"]
    unscreened_total = sum(
        int(row["n_universe"])
        for row in outlier["per_transition"]
        if row["screen_id"] == "unscreened"
    )
    excluded = int(primary_universe["total_transition_security_exclusions"])
    if abs(
        primary_universe["exclusion_share_across_transition_security_rows"]
        - excluded / unscreened_total
    ) > 1e-12:
        failures.append("outlier exclusion share does not match row counts")
    require(
        "outlier audit exclusions",
        f"{excluded} of {tex_int(unscreened_total)} transition--security rows",
    )
    require(
        "outlier audit share and N",
        (
            f"{fmt2_pct(primary_universe['exclusion_share_across_transition_security_rows'])}"
            f"\\%; $N={tex_int(primary_universe['minimum_n_after'])}$--"
            f"${tex_int(primary_universe['maximum_n_after'])}$"
        ),
    )
    outlier_methods = primary_outlier["methods"]
    require(
        "outlier audit NDCG",
        "NDCG@5 is $"
        + "/".join(
            fmt3(outlier_methods[method]["macro_mean_ndcg_at_5"])
            for method in ("ReturnPath", "TF-IDF", "FrozenBlend")
        )
        + "$",
    )

    def outlier_left_wins(
        screen: dict, left_method: str, right_method: str
    ) -> int | None:
        for comparison in screen["comparisons"]:
            if (
                comparison["left_method"] == left_method
                and comparison["right_method"] == right_method
            ):
                return int(comparison["ndcg_at_5"]["left_wins"])
        return None

    for screen_id in expected_screens - {"unscreened"}:
        screen = outlier_screens[screen_id]
        if (
            outlier_left_wins(screen, "ReturnPath", "TF-IDF") != 6
            or outlier_left_wins(screen, "FrozenBlend", "ReturnPath") != 6
        ):
            failures.append(
                f"outlier ordering no longer holds 6/6 under {screen_id}"
            )
    require(
        "outlier audit alternate screens",
        (
            "feature-period screens at thresholds of 1.0\n"
            "and 0.5 retain both method orderings"
        ),
    )
    require(
        "outlier audit retrospective caveat",
        "we use them only as retrospective data-quality checks",
    )

    learned = load_json(
        LEARNED_FUSION_RUN / "learned_fusion_followup.json"
    )["modern_evaluation"]["summary"]
    learned_values = learned["aggregates"][
        "historically_trained_nonlinear_fusion"
    ]
    require(
        "historically trained nonlinear fusion",
        (
            f"{fmt3(learned_values['macro_mean_ndcg'])}"
            f"\\pm{fmt3(learned_values['population_std_across_transition_means'])}"
        ),
    )
    if (
        learned["modern_data_used_for_training_or_configuration"] is not False
        or learned["learned_wins_over_return_path"] != 0
        or learned["learned_wins_over_frozen_rank_blend"] != 0
    ):
        failures.append("learned fusion audit no longer trails in all transitions")

    two_tower = load_json(GATED_TWO_TOWER_SELECTION)
    selected_width = int(two_tower["selected_width"])
    continuation_slots = [
        row
        for row in two_tower["validation_slots"]
        if int(row["width"]) == selected_width
    ]
    continuation_gate = two_tower["continuation_gate"]
    model_mean = statistics.fmean(
        float(row["seed_mean_model_ndcg"]) for row in continuation_slots
    )
    blend_mean = statistics.fmean(
        float(row["baseline_ndcg"]) for row in continuation_slots
    )
    wins = sum(bool(row["win"]) for row in continuation_slots)
    selected_seed_rows = [
        row
        for row in two_tower["seed_validation_results"]
        if int(row["width"]) == selected_width
    ]
    seed_values_by_transition = {
        str(slot["transition"]): [
            float(row["model_ndcg"])
            for row in selected_seed_rows
            if row["transition"] == slot["transition"]
        ]
        for slot in continuation_slots
    }
    if (
        selected_width != 32
        or len(continuation_slots) != 6
        or continuation_gate["passed"] is not False
        or int(continuation_gate["observed_validation_slot_wins"]) != wins
        or abs(
            float(continuation_gate["observed_mean_validation_advantage"])
            - (model_mean - blend_mean)
        )
        > 1e-12
        or any(row["win"] is not False for row in continuation_slots)
        or any(len(values) != 3 for values in seed_values_by_transition.values())
        or max(
            statistics.pstdev(values)
            for values in seed_values_by_transition.values()
        )
        >= 0.002
    ):
        failures.append("gated two-tower continuation record no longer matches")
    require(
        "gated two-tower continuation",
        (
            f"mean NDCG@5 is ${fmt3(model_mean)}$ versus "
            f"${fmt3(blend_mean)}$ for the frozen blend across the six\n"
            f"continuation tests, with {wins}/{len(continuation_slots)} wins "
            "and little variation across seeds"
        ),
    )

    two_extension = followup["two_window_full_extension"]["summary"]
    for label, key in (
        ("18-transition two-window result", "two_window"),
        ("18-transition one-window result", "one_window"),
    ):
        values = two_extension[key]
        require(
            label,
            (
                f"{fmt3(values['macro_mean_ndcg'])}"
                f"\\pm{fmt3(values['population_std_across_transition_means'])}"
            ),
        )
    require(
        "two-window extension win count",
        "improves on $M=1$ in all\n18",
    )
    if (
        two_extension["n_feasible_transitions"] != 18
        or two_extension["two_window_wins"] != 18
    ):
        failures.append("two-window extension does not improve all 18 transitions")

    stale = followup["stale_return_control"]["summary"]
    for label, key in (
        ("stale return path", "stale_return_path"),
        ("stale-control TF-IDF", "tfidf"),
        ("matched current return path", "current_return_path"),
    ):
        values = stale[key]
        require(
            label,
            (
                f"{fmt3(values['macro_mean_ndcg'])}"
                f"\\pm{fmt3(values['population_std_across_transition_means'])}"
            ),
        )
    require(
        "stale-control N range",
        f"{tex_int(stale['n_universe_min'])}$--${tex_int(stale['n_universe_max'])}",
    )
    if stale["stale_return_path_wins_over_tfidf"] != 6:
        failures.append("stale return path does not lead TF-IDF in all six transitions")

    universe = followup["universe_construction_disclosure"]
    if universe["point_in_time_security_master"] is not False:
        failures.append("candidate universe is unexpectedly point-in-time")
    require("candidate universe size", f"{universe['candidate_universe_n']:,}")
    require("candidate freeze date", "April~12, 2026")
    require("candidate news filter", "2025--2026")

    ticker_artifact = load_json(
        ROBUST_RUN / "06_common_share_heuristic_exclusion.json"
    )
    ticker_summary = ticker_artifact["summary"]
    ticker_rows = ticker_artifact["per_transition"]
    require(
        "ticker-syntax flagged count",
        str(ticker_summary["distinct_flagged_tickers_across_modern_feature_time_universes"]),
    )
    for label, key in (
        ("restricted return path", "return_path_after_ndcg"),
        ("restricted TF-IDF", "tfidf_after_ndcg"),
    ):
        mean, sd = mean_sd([float(row[key]) for row in ticker_rows])
        require(label, f"{fmt3(mean)}\\pm{fmt3(sd)}")
    restricted_gap = statistics.fmean(
        float(row["return_path_after_ndcg"])
        - float(row["tfidf_after_ndcg"])
        for row in ticker_rows
    )
    require("restricted ticker-syntax gap", f"{fmt3(restricted_gap)}")
    if any(int(row["n_flagged_excluded"]) != 118 for row in ticker_rows):
        failures.append("ticker-syntax flagged set does not persist at N=118")
    if ticker_summary["return_path_above_tfidf_after_in_transitions"] != 6:
        failures.append("restricted return path does not lead in all transitions")

    reconciliation = load_json(
        ROBUST_RUN / "07_symmetric_vs_feature_time_reconciliation.json"
    )
    reconciliation_added = [
        row["n_removed_by_target_text_coverage"]
        for row in reconciliation["per_transition"]
    ]
    if max(reconciliation_added) != 3:
        failures.append("feature-time reconciliation no longer has maximum +3 names")
    if min(reconciliation_added) != 0:
        failures.append("feature-time reconciliation no longer has minimum +0 names")
    require("eligibility reconciliation", "adds 0--3 names")

    # Frozen-config 19-transition extension.
    full19 = load_json(FULL19_RUN / "full19_feature_time_census.json")
    overall = full19["summary"]["overall"]
    for label, method in (
        ("19-transition return path", "ReturnPath"),
        ("19-transition TF-IDF", "TF-IDF"),
    ):
        values = overall[method]
        require(
            label,
            (
                f"{fmt3(values['macro_mean_ndcg'])}"
                f"\\pm{fmt3(values['population_std_across_transition_means'])}"
            ),
        )
    wins = overall["comparison"]["return_path_wins"]
    require("19-transition win count", f"{wins}/19")
    if wins != 18:
        failures.append(f"unexpected full-extension win count: {wins}")
    odd_rows = [
        row for row in full19["per_transition"]
        if int(row["transition_index"]) % 2 == 1
    ]
    even_rows = [
        row for row in full19["per_transition"]
        if int(row["transition_index"]) % 2 == 0
    ]
    odd_wins = sum(bool(row["return_path_wins"]) for row in odd_rows)
    even_wins = sum(bool(row["return_path_wins"]) for row in even_rows)
    require("odd-index nonadjacent wins", f"{odd_wins}/{len(odd_rows)}")
    require("even-index nonadjacent wins", f"{even_wins}/{len(even_rows)}")
    if (odd_wins, len(odd_rows), even_wins, len(even_rows)) != (9, 10, 9, 9):
        failures.append("unexpected alternating-transition win counts")

    for regime, manuscript_label in (
        ("HIGH_VOL", "high-volatility"),
        ("MED_VOL", "medium-volatility"),
        ("LOW_VOL", "low-volatility"),
    ):
        values = full19["summary"]["by_regime"][regime]
        for method in ("ReturnPath", "TF-IDF"):
            row = values[method]
            require(
                f"{manuscript_label} {method}",
                (
                    f"{fmt3(row['macro_mean_ndcg'])}"
                    f"\\pm{fmt3(row['population_std_across_transition_means'])}"
                ),
            )

    historical_rows = [
        row
        for row in full19["per_transition"]
        if row["feature_window"].startswith("H")
    ]
    modern19_rows = [
        row
        for row in full19["per_transition"]
        if row["feature_window"].startswith("W")
    ]
    historical_n = [row["n_feature_time_eligible"] for row in historical_rows]
    require(
        "historical feature-time N range",
        f"{tex_int(min(historical_n))}$--${tex_int(max(historical_n))}",
    )
    historical_coverage = [
        row["text_coverage_of_feature_return_universe"] for row in historical_rows
    ]
    modern_coverage = [
        row["text_coverage_of_feature_return_universe"] for row in modern19_rows
    ]
    require(
        "historical text coverage",
        (
            f"{fmt1_pct(min(historical_coverage))}$--$"
            f"{fmt1_pct(max(historical_coverage))}\\%"
        ),
    )
    require(
        "modern text coverage",
        (
            f"{fmt1_pct(min(modern_coverage))}$--$"
            f"{fmt1_pct(max(modern_coverage))}\\%"
        ),
    )

    # Granite tokenizer audit.
    token = load_json(TOKEN_RUN / "summary.json")["aggregate"]
    require("Granite median token length", f"{int(token['tokens_p50']):,}")
    require(
        "Granite fraction over 512",
        f"{fmt1_pct(token['fraction_tokens_gt_512'])}\\%",
    )
    require(
        "Item 1 collection-cap fraction",
        f"{fmt1_pct(token['fraction_chars_at_collection_cap'])}\\%",
    )
    if token["chars_p50"] != 8000.0:
        failures.append("median stored Item 1 excerpt no longer hits 8,000 chars")
    if token["n_tokens_gt_8192"] != 0:
        failures.append("stored vintage excerpt unexpectedly exceeds 8,192 tokens")

    # Regressions from superseded analyses.
    for stale in (
        "200 seeded queries",
        "symmetric evaluation contains",
        "0.325\\pm0.020",
        "0.336\\pm0.015",
        "paired bootstrap",
        "half-sample",
        "Sign-test",
        "sign test",
        "full-context",
        "vintage-aware benchmark",
        "modalities are complementary",
    ):
        forbid("claim hygiene", stale)

    print("Frozen artifact-to-manuscript checks")
    for label, literal in checks:
        status = "OK" if literal in text else "FAIL"
        print(f"  {status:4s} {label}: {literal}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"\nPASS: {len(checks)} checked claims; checksum manifests and "
        "standalone abstract verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
