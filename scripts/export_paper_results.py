"""Export aggregate-only paper results from the original local experiment tree.

No market arrays, filing excerpts, ticker lists, checkpoints, environment paths,
or private execution records are exported. The original artifacts are required
to regenerate these extracts; they are not distributed in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "full19_frozen_blend.json": (
        "results/full19_frozen_blend/run_20261016_fixed80_20/full19_frozen_blend.json",
        ("configuration", "per_transition", "summary", "validations", "interpretation"),
    ),
    "frozen_blend_depth_ticker.json": (
        "results/frozen_blend_depth_ticker/run_20261016_fixed80_20/frozen_blend_depth_ticker.json",
        ("configuration", "depth_per_transition", "ticker_exclusion_per_transition",
         "summary", "validations", "interpretation"),
    ),
    "learned_fusion.json": (
        "results/learned_fusion_followup/run_20260731T012556Z/learned_fusion_followup.json",
        ("configuration", "training_audit", "modern_evaluation"),
    ),
    "cnn_historical_v1.json": (
        "experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v1/historical_selection.json",
        ("baseline", "baseline_recomputed_from_historical_inputs", "config_sha256",
         "continuation_gate", "selected_width", "width_summaries", "validation_slots",
         "seed_validation_results"),
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export(source_root: Path, output_dir: Path, *, overwrite: bool = False) -> None:
    artifacts = {}
    for name, (relative, fields) in SOURCES.items():
        raw = (source_root / relative).read_bytes()
        original = json.loads(raw)
        extract = {
            "export_schema": "paper-aggregate-extract-v1",
            "source": {
                "artifact_path": relative,
                "artifact_sha256": sha256(raw),
                "schema_version": original["schema_version"],
                "selected_top_level_fields": list(fields),
            },
            "scope": "Selected aggregate fields copied exactly from the original result; not a new experimental run.",
            **{field: original[field] for field in fields},
        }
        artifacts[name] = (json.dumps(extract, indent=2, sort_keys=True) + "\n").encode()
    manifest = "".join(f"{sha256(data)}  {name}\n" for name, data in sorted(artifacts.items()))
    artifacts["SHA256SUMS.txt"] = manifest.encode()
    if not overwrite:
        existing = [name for name in artifacts if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite {existing}; use --overwrite explicitly")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (output_dir / name).write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_results")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export(args.source_root, args.output_dir, overwrite=args.overwrite)
    print(f"Exported four aggregate result extracts to {args.output_dir}")
