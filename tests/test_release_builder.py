"""Contracts for the history-free academic release allowlist."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_anonymous_release as release


class ReleaseBuilderTests(unittest.TestCase):
    def test_active_code_and_reader_documents_are_released(self) -> None:
        required = set(release.REQUIRED_FILES)
        expected = {
            "artifact_authorities.json",
            "artifact_release/.gitignore",
            "artifact_release/README.md",
            "artifact_release/DATA.md",
            "artifact_release/LICENSE",
            "artifact_release/LICENSE-CC-BY-4.0.txt",
            "artifact_release/REPRODUCTION.md",
            "artifact_release/CODE_MAP.md",
            "artifact_release/LICENSE_NOTICE.md",
            "environment/core.lock",
            "environment/models.lock",
            "environment/cloud-launch.lock",
            "environment/granite-sagemaker.requirements",
            "plots/style.py",
            "scripts/fetch_vintage_10k.py",
            "scripts/sec_filing_text.py",
            "scripts/build_dense_embedding_cache.py",
            "scripts/run_full_return_census.py",
            "scripts/run_same_window_text_diagnostic.py",
            "scripts/run_pairwise_alignment.py",
            "scripts/verify_artifact.py",
            "paper/main.pdf",
            "tests/test_sec_filing_text.py",
        }
        self.assertFalse(expected - required)

    def test_multimodal_v2_source_protocol_and_tests_are_released(self) -> None:
        required = set(release.REQUIRED_FILES)
        expected = {
            "experiments/multimodal_peer_encoder/.gitignore",
            "experiments/multimodal_peer_encoder/PROTOCOL_v2.md",
            "experiments/multimodal_peer_encoder/PROTOCOL_v2_CLARIFICATION_1.md",
            "experiments/multimodal_peer_encoder/README.md",
            "experiments/multimodal_peer_encoder/build_historical_embeddings.py",
            "experiments/multimodal_peer_encoder/config/v2_historical_protocol.json",
            "experiments/multimodal_peer_encoder/report_v2.py",
            "experiments/multimodal_peer_encoder/run_v2_historical.py",
            "experiments/multimodal_peer_encoder/v2_data.py",
            "experiments/multimodal_peer_encoder/v2_diagnostics.py",
            "experiments/multimodal_peer_encoder/v2_model.py",
            "experiments/multimodal_peer_encoder/v2_self_tests.py",
            "experiments/multimodal_peer_encoder/v2_training.py",
            "experiments/multimodal_peer_encoder/tests/test_v2_data.py",
            "experiments/multimodal_peer_encoder/tests/test_v2_model.py",
            "experiments/multimodal_peer_encoder/tests/test_v2_report.py",
            "experiments/multimodal_peer_encoder/tests/test_v2_runner.py",
            "experiments/multimodal_peer_encoder/tests/test_v2_training.py",
        }
        self.assertFalse(expected - required)

    def test_multimodal_release_excludes_restricted_inputs_and_checkpoints(self) -> None:
        selected = set(release.REQUIRED_FILES) | set(release.OPTIONAL_FILES)
        forbidden_fragments = {
            "results/ea_historical/windows/",
            "results/vintage_descriptions/",
            "artifacts/historical_text_cache/",
            "/checkpoints/",
            "models/granite-embedding-english-r2/",
        }
        for path in selected:
            self.assertFalse(
                any(fragment in path for fragment in forbidden_fragments), path
            )
            self.assertFalse(path.endswith((".npy", ".pt", ".safetensors")), path)

    def test_superseded_code_and_inactive_figures_are_not_released(self) -> None:
        required = set(release.REQUIRED_FILES)
        excluded = {
            "results/full_universe_ndcg_corrected.csv",
            "scripts/exp_vintage_eval.py",
            "scripts/exp_vintage_all_encoders.py",
            "scripts/exp_wf_ndcg.py",
            "scripts/run_top5_realized_correlation.py",
            "scripts/gen_fig_orthogonality.py",
            "paper/figures/fig_orthogonality.pdf",
            "paper/figures/provenance.log",
            "artifact_release/ANONYMOUS_UPLOAD.md",
            "artifact_release/CHECKLIST.md",
            ".gitignore",
        }
        self.assertFalse(excluded & required)
        self.assertFalse(
            any("top5_realized_correlation" in path for path in required)
        )

    def test_nested_integrity_files_are_in_the_allowlist(self) -> None:
        required = set(release.REQUIRED_FILES)
        for directory, filenames in release.RELEASE_CHECKSUM_GROUPS.items():
            self.assertIn(f"{directory}/SHA256SUMS.txt", required)
            for filename in filenames:
                self.assertIn(f"{directory}/{filename}", required)
        for manifest, filenames in release.RELEASE_RESULT_MANIFESTS.items():
            self.assertIn(manifest, required)
            directory = manifest.rsplit("/", 1)[0]
            for filename in filenames:
                self.assertIn(f"{directory}/{filename}", required)

    def test_destination_paths_are_unique(self) -> None:
        destinations = [
            release.DESTINATION_OVERRIDES.get(path, path)
            for path in release.REQUIRED_FILES
        ]
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertEqual(
            release.DESTINATION_OVERRIDES["artifact_release/LICENSE"],
            "LICENSE",
        )
        self.assertEqual(
            release.DESTINATION_OVERRIDES[
                "artifact_release/LICENSE-CC-BY-4.0.txt"
            ],
            "LICENSE-CC-BY-4.0.txt",
        )

    def test_short_git_commit_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.txt"
            linkable_reference = "sealed at " + "commit " + "6ea" + "1480\n"
            path.write_text(linkable_reference, encoding="utf-8")
            findings = release.scan_file(path, "source.txt")
        self.assertTrue(
            any(
                finding["kind"] == "source-control commit reference"
                for finding in findings
            )
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def _write_self_hashed_json(cls, path: Path, payload: dict) -> None:
        payload = dict(payload)
        payload["canonical_payload_sha256"] = release._canonical_json_sha256(
            payload
        )
        cls._write_json(path, payload)

    def _complete_v2_fixture(self, root: Path, decision: str = "STOP") -> Path:
        run = root / release.V2_RUN_DIRECTORY
        development = run / "training_history" / "development"
        gate = run / "training_history" / "gate"
        development.mkdir(parents=True)
        gate.mkdir(parents=True)
        sealed_commit = "d" * 40

        for architecture in ("a", "b"):
            for softness in ("raw_scale", "lw"):
                for application in ("global", "rank_band"):
                    for learning_rate in ("0p003", "0p03"):
                        for seed in (11, 29, 47):
                            name = (
                                f"dev-h{architecture}-{softness}-{application}-"
                                f"lr{learning_rate}-seed{seed}.json"
                            )
                            self._write_self_hashed_json(
                                development / name, {"code_commit": sealed_commit}
                            )

        winners = [
            {
                "architecture": "H-A",
                "target_softness": "raw_scale",
                "application": "global",
                "learning_rate": 0.003,
            },
            {
                "architecture": "H-B",
                "target_softness": "lw",
                "application": "rank_band",
                "learning_rate": 0.03,
            },
        ]
        for winner in winners:
            architecture = winner["architecture"].replace("-", "").lower()
            learning_rate = {0.003: "0p003", 0.03: "0p03"}[
                winner["learning_rate"]
            ]
            for fold in (1, 2, 3):
                for seed in (11, 29, 47):
                    name = (
                        f"gate-{architecture}-{winner['target_softness']}-"
                        f"{winner['application']}-lr{learning_rate}-fold{fold}-"
                        f"seed{seed}.json"
                    )
                    self._write_self_hashed_json(
                        gate / name, {"code_commit": sealed_commit}
                    )

        self._write_self_hashed_json(
            run / "RUN_START.json", {"code_commit": sealed_commit}
        )
        self_test_log = run / "SELF_TESTS.log"
        self_test_log.write_text("PASS\n", encoding="utf-8")
        self._write_self_hashed_json(
            run / "self_tests.json",
            {"tests": [], "log_sha256": release.sha256_file(self_test_log)},
        )
        self._write_json(run / "development_results.json", {"fits": []})
        selection = {
            "family_winners": winners,
            "development_results_sha256": release.sha256_file(
                run / "development_results.json"
            ),
        }
        selection_encoded = json.dumps(
            selection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        selection["selection_sha256"] = hashlib.sha256(selection_encoded).hexdigest()
        self._write_json(run / "development_selection.json", selection)

        prefinal_paths = [
            run / "RUN_START.json",
            run / "SELF_TESTS.log",
            run / "self_tests.json",
            run / "development_results.json",
            run / "development_selection.json",
            *sorted(development.glob("*.json")),
            *sorted(gate.glob("*.json")),
        ]
        prefinal_hashes = {
            path.relative_to(run).as_posix(): release.sha256_file(path)
            for path in prefinal_paths
        }
        self._write_self_hashed_json(
            run / "FINALIZATION_START.json",
            {
                "decision": decision,
                "prefinal_artifact_sha256": prefinal_hashes,
            },
        )

        runner = "experiments/multimodal_peer_encoder/run_v2_historical.py"
        release_builder = "scripts/build_anonymous_release.py"
        scientific_source = "experiments/multimodal_peer_encoder/v2_model.py"
        source_sha256 = {
            ".gitignore": release.sha256_file(
                release.PROJECT_ROOT / ".gitignore"
            ),
            runner: release.sha256_file(release.PROJECT_ROOT / runner),
            release_builder: release.sha256_file(
                release.PROJECT_ROOT / release_builder
            ),
            scientific_source: release.sha256_file(
                release.PROJECT_ROOT / scientific_source
            ),
        }

        manifest = {
            "decision": decision,
            "code_commit": sealed_commit,
            "base_protocol_commit": "e" * 40,
            "clarification_commit": "f" * 40,
            "fit_roster": {
                "development_completed": 48,
                "gate_completed": 18,
                "total_completed": 66,
            },
            "prefinal_artifact_sha256": prefinal_hashes,
            "finalization_state_sha256": release.sha256_file(
                run / "FINALIZATION_START.json"
            ),
            "source_sha256": source_sha256,
        }
        manifest["canonical_payload_sha256"] = release._canonical_json_sha256(
            manifest
        )
        self._write_json(run / "RUN_MANIFEST.json", manifest)
        self._write_json(
            run / "historical_results.json",
            {
                "run": {"status": decision},
                "gate": {"decision": decision},
                "provenance": {
                    "peeled_commit": sealed_commit,
                    "manifest_sha256": release.sha256_file(
                        run / "RUN_MANIFEST.json"
                    ),
                    "source_sha256": source_sha256,
                },
            },
        )
        (run / "RUN_REPORT.md").write_text(
            "\n".join(
                (
                    f"| peeled_commit | {sealed_commit} |",
                    f"| manifest_sha256 | {release.sha256_file(run / 'RUN_MANIFEST.json')} |",
                    f"| .gitignore | {release.sha256_file(release.PROJECT_ROOT / '.gitignore')} |",
                    f"| {runner} | {release.sha256_file(release.PROJECT_ROOT / runner)} |",
                    f"| {release_builder} | {release.sha256_file(release.PROJECT_ROOT / release_builder)} |",
                    f"| {scientific_source} | {release.sha256_file(release.PROJECT_ROOT / scientific_source)} |",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (run / decision).write_text(f"{decision}\n", encoding="utf-8")
        receipt = {
            "decision": decision,
            "artifact_sha256": {
                name: release.sha256_file(run / name)
                for name in (
                    "RUN_MANIFEST.json",
                    "historical_results.json",
                    "RUN_REPORT.md",
                    decision,
                )
            },
        }
        receipt["canonical_payload_sha256"] = release._canonical_json_sha256(
            receipt
        )
        self._write_json(run / "RUN_RECEIPT.json", receipt)
        return run

    def test_v2_result_discovery_is_all_or_nothing_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / release.V2_RUN_DIRECTORY
            run.mkdir(parents=True)
            self.assertEqual(release.discover_complete_v2_run(root), [])

            run.rmdir()
            run = self._complete_v2_fixture(root)
            checkpoint = run / "checkpoints" / "private.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"not released")
            selected = release.discover_complete_v2_run(root)
            self.assertEqual(len(selected), 77)
            self.assertFalse(any("checkpoint" in path for path in selected))
            self.assertFalse(any("historical_text_cache" in path for path in selected))

    def test_v2_result_discovery_rejects_an_incomplete_decided_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / release.V2_RUN_DIRECTORY
            run.mkdir(parents=True)
            (run / "STOP").write_text("STOP\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                release.discover_complete_v2_run(root)

    def test_v2_result_discovery_ignores_hidden_resume_remnants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._complete_v2_fixture(root)
            development_remnant = (
                run
                / "training_history"
                / "development"
                / ".dev-ha-lw-global-lr0p003-seed11.json.incomplete-20260802T120000Z"
            )
            gate_remnant = (
                run
                / "training_history"
                / "gate"
                / ".gate-hb-lw-rank_band-lr0p03-fold3-seed47.json.partial-123-456"
            )
            development_remnant.write_text("retired partial fit\n", encoding="utf-8")
            gate_remnant.write_text("interrupted atomic write\n", encoding="utf-8")

            selected = release.discover_complete_v2_run(root)

            self.assertEqual(len(selected), 77)
            self.assertFalse(any(".incomplete-" in path for path in selected))
            self.assertFalse(any(".partial-" in path for path in selected))
            self.assertNotIn(
                development_remnant.relative_to(root).as_posix(), selected
            )
            self.assertNotIn(gate_remnant.relative_to(root).as_posix(), selected)

    def test_v2_git_identities_are_anonymized_without_removing_fields(self) -> None:
        commit = "a" * 40
        cleaned, count = release.redact_json_provenance(
            {
                "code_commit": commit,
                "provenance": {"peeled_commit": commit},
                "configuration_sha256": "b" * 64,
            }
        )
        self.assertEqual(count, 2)
        self.assertEqual(cleaned["code_commit"], release.ANONYMOUS_GIT_IDENTITY)
        self.assertEqual(
            cleaned["provenance"]["peeled_commit"],
            release.ANONYMOUS_GIT_IDENTITY,
        )
        self.assertEqual(cleaned["configuration_sha256"], "b" * 64)

        audit_path = "experiments/multimodal_peer_encoder/AUDIT_v2.md"
        text, replacements = release.anonymize_text_provenance(
            f"commit {commit}; sha {'c' * 64}\n", audit_path
        )
        self.assertEqual(replacements, 1)
        self.assertIn(release.ANONYMOUS_GIT_IDENTITY, text)
        self.assertIn("c" * 64, text)

        log_path = f"{release.V2_RUN_DIRECTORY}/SELF_TESTS.log"
        mac_home = "/" + "Users/reviewer/"
        linux_home = "/" + "home/reviewer/"
        text, replacements = release.anonymize_text_provenance(
            f"{mac_home}Library/Python/pytest.py: warning\n"
            f"54 passed; cache={linux_home}.cache; again={mac_home}tmp\n",
            log_path,
        )
        self.assertEqual(replacements, 3)
        self.assertNotIn(mac_home, text)
        self.assertNotIn(linux_home, text)
        self.assertIn(release.ANONYMOUS_USER_HOME, text)
        self.assertIn("pytest.py: warning", text)
        self.assertIn("54 passed", text)
        unrelated = f"warning from {mac_home}Library/Python/pytest.py\n"
        untouched, replacements = release.anonymize_text_provenance(
            unrelated, "logs/unrelated.log"
        )
        self.assertEqual(replacements, 0)
        self.assertEqual(untouched, unrelated)

    def test_anonymous_v2_runner_removes_git_ids_but_keeps_sha256_ids(self) -> None:
        runner = "experiments/multimodal_peer_encoder/run_v2_historical.py"
        expected_config_sha256 = (
            "d094b78050ecfd25760dc1304b7a5214649a13928dfb0cc8aaa14a6e4d76c0f8"
        )
        source = (release.PROJECT_ROOT / runner).read_text(encoding="utf-8")
        source_git_ids = release.GIT_IDENTITY_PATTERN.findall(source)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            transformations = release.copy_allowlisted([runner], destination)
            copied = (destination / runner).read_text(encoding="utf-8")

        # The same test is shipped in the anonymous artifact.  It therefore has
        # to cover both the private source (two Git identities) and a second pass
        # over the already-redacted release (no remaining identities).
        self.assertIn(len(source_git_ids), {0, 2})
        self.assertEqual(transformations.get(runner, 0), len(source_git_ids))
        self.assertIsNone(release.GIT_IDENTITY_PATTERN.search(copied))
        self.assertEqual(copied.count(release.ANONYMOUS_GIT_IDENTITY), 2)
        self.assertIn(expected_config_sha256, copied)

    def test_v2_anonymous_redaction_refreshes_nested_integrity(self) -> None:
        release_ignore = (
            "artifact_release/.gitignore"
            if (release.PROJECT_ROOT / "artifact_release/.gitignore").is_file()
            else ".gitignore"
        )
        runner = "experiments/multimodal_peer_encoder/run_v2_historical.py"
        release_builder = "scripts/build_anonymous_release.py"
        scientific_source = "experiments/multimodal_peer_encoder/v2_model.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._complete_v2_fixture(root)
            release.copy_allowlisted(
                [release_ignore, runner, release_builder, scientific_source], root
            )

            log_path = run / "SELF_TESTS.log"
            fixture_home = "/" + "Users/reviewer/"
            log_path.write_text(
                f"{fixture_home}Library/Python/pytest.py: warning\n",
                encoding="utf-8",
            )
            log, replacements = release.anonymize_text_provenance(
                log_path.read_text(encoding="utf-8"),
                f"{release.V2_RUN_DIRECTORY}/SELF_TESTS.log",
            )
            self.assertEqual(replacements, 1)
            log_path.write_text(log, encoding="utf-8")

            for path in sorted(run.rglob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                cleaned, _ = release.redact_json_provenance(payload)
                self._write_json(path, cleaned)
            report_path = run / "RUN_REPORT.md"
            report, _ = release.anonymize_text_provenance(
                report_path.read_text(encoding="utf-8"),
                f"{release.V2_RUN_DIRECTORY}/RUN_REPORT.md",
            )
            report_path.write_text(report, encoding="utf-8")

            refreshed = release.refresh_v2_run_integrity(root)
            self.assertIn(
                f"{release.V2_RUN_DIRECTORY}/RUN_START.json", refreshed
            )
            for path in (
                run / "RUN_START.json",
                run / "self_tests.json",
                run / "FINALIZATION_START.json",
                run / "RUN_MANIFEST.json",
                run / "RUN_RECEIPT.json",
                *sorted((run / "training_history").rglob("*.json")),
            ):
                release._validate_self_hashed_json(path)

            manifest = release._validate_self_hashed_json(
                run / "RUN_MANIFEST.json"
            )
            finalization = release._validate_self_hashed_json(
                run / "FINALIZATION_START.json"
            )
            for name, digest in manifest["prefinal_artifact_sha256"].items():
                self.assertEqual(release.sha256_file(run / name), digest)
            self.assertEqual(
                manifest["prefinal_artifact_sha256"],
                finalization["prefinal_artifact_sha256"],
            )
            self.assertEqual(
                manifest["finalization_state_sha256"],
                release.sha256_file(run / "FINALIZATION_START.json"),
            )
            self.assertEqual(
                manifest["source_sha256"][".gitignore"],
                release.sha256_file(root / ".gitignore"),
            )
            if release_ignore != ".gitignore":
                self.assertNotEqual(
                    manifest["source_sha256"][".gitignore"],
                    release.sha256_file(release.PROJECT_ROOT / ".gitignore"),
                )
            self.assertEqual(
                manifest["source_sha256"][runner],
                release.sha256_file(root / runner),
            )
            self.assertEqual(
                manifest["source_sha256"][release_builder],
                release.sha256_file(root / release_builder),
            )
            self.assertEqual(
                manifest["source_sha256"][scientific_source],
                release.sha256_file(root / scientific_source),
            )
            self_tests = release._validate_self_hashed_json(
                run / "self_tests.json"
            )
            self.assertEqual(
                self_tests["log_sha256"], release.sha256_file(log_path)
            )
            self.assertNotIn(
                "/" + "Users/", log_path.read_text(encoding="utf-8")
            )

            result = json.loads(
                (run / "historical_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                result["provenance"]["source_sha256"], manifest["source_sha256"]
            )
            self.assertEqual(
                result["provenance"]["manifest_sha256"],
                release.sha256_file(run / "RUN_MANIFEST.json"),
            )
            self.assertEqual(
                result["provenance"]["source_sha256"][runner],
                release.sha256_file(root / runner),
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIsNone(release.GIT_IDENTITY_PATTERN.search(report))
            for name, digest in manifest["source_sha256"].items():
                self.assertIn(f"| {name} | {digest} |", report)

            receipt = release._validate_self_hashed_json(run / "RUN_RECEIPT.json")
            for name, digest in receipt["artifact_sha256"].items():
                self.assertEqual(release.sha256_file(run / name), digest)

    def test_v2_integrity_refresh_rejects_scientific_source_drift(self) -> None:
        release_ignore = (
            "artifact_release/.gitignore"
            if (release.PROJECT_ROOT / "artifact_release/.gitignore").is_file()
            else ".gitignore"
        )
        runner = "experiments/multimodal_peer_encoder/run_v2_historical.py"
        release_builder = "scripts/build_anonymous_release.py"
        scientific_source = "experiments/multimodal_peer_encoder/v2_model.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_v2_fixture(root)
            release.copy_allowlisted(
                [release_ignore, runner, release_builder, scientific_source], root
            )
            source_path = root / scientific_source
            source_path.write_text(
                source_path.read_text(encoding="utf-8") + "\n# altered after seal\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "scientific source differs"):
                release.refresh_v2_run_integrity(root)


if __name__ == "__main__":
    unittest.main()
