#!/usr/bin/env python3
"""Verify release integrity, manuscript claims, and focused unit contracts.

These checks answer three different questions and are reported separately:

1. Did any released file change after the archive manifest was created?
2. Do the frozen result authorities still support the active manuscript text?
3. Do the metric and return-window implementations satisfy their unit tests?

Passing this command verifies the submitted artifact boundary. It does not
recompute private input arrays or establish statistical validity by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUTHORITY_MAP = ROOT / "artifact_authorities.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release_manifest(
    path: Path,
    *,
    strict_file_set: bool,
) -> list[str]:
    """Return integrity failures for one release-level manifest."""
    if not path.is_file():
        return [f"release manifest is absent: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    recorded: set[str] = set()
    for row in payload.get("files", []):
        relative = str(row.get("path", ""))
        candidate = Path(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            failures.append(f"unsafe manifest path: {relative!r}")
            continue
        recorded.add(candidate.as_posix())
        target = ROOT / candidate
        if not target.is_file():
            failures.append(f"manifest file is missing: {relative}")
            continue
        if target.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {relative}")
        if sha256_file(target) != row["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")

    if strict_file_set:
        permitted_unlisted = {
            path.resolve().relative_to(ROOT.resolve()).as_posix(),
        }
        observed = {
            candidate.relative_to(ROOT).as_posix()
            for candidate in ROOT.rglob("*")
            if candidate.is_file()
            and "__pycache__" not in candidate.parts
            and not candidate.name.endswith(".pyc")
        }
        extras = sorted(observed - recorded - permitted_unlisted)
        if extras:
            failures.append(
                "unlisted files are present: " + ", ".join(extras)
            )
    return failures


def verify_authority_map(path: Path) -> list[str]:
    """Check that every indexed authority exists and is checksummed as stated."""
    if not path.is_file():
        return [f"authority map is absent: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    scopes: set[str] = set()
    for row in payload.get("authorities", []):
        scope = str(row.get("scope", ""))
        relative = str(row.get("path", ""))
        candidate = Path(relative)
        if not scope or scope in scopes:
            failures.append(f"missing or duplicate authority scope: {scope!r}")
        scopes.add(scope)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            failures.append(f"unsafe authority path for {scope}: {relative!r}")
            continue
        authority = ROOT / candidate
        if not authority.is_file():
            failures.append(f"authority is missing for {scope}: {relative}")
            continue
        manifest_relative = row.get("checksum_manifest")
        if manifest_relative is None:
            continue
        manifest_candidate = Path(str(manifest_relative))
        if (
            manifest_candidate.is_absolute()
            or ".." in manifest_candidate.parts
        ):
            failures.append(
                f"unsafe checksum-manifest path for {scope}: "
                f"{manifest_relative!r}"
            )
            continue
        manifest = ROOT / manifest_candidate
        if not manifest.is_file():
            failures.append(
                f"checksum manifest is missing for {scope}: {manifest_relative}"
            )
            continue
        expected_line = (
            f"{sha256_file(authority)}  {authority.name}"
        )
        if expected_line not in manifest.read_text(encoding="utf-8").splitlines():
            failures.append(
                f"authority is not correctly checksummed for {scope}: {relative}"
            )
    if not scopes:
        failures.append("authority map contains no authorities")
    return failures


def run_check(label: str, command: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.rstrip()
    print(f"\n[{label}]")
    print(output or "(no output)")
    return completed.returncode == 0, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the research artifact.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "ARTIFACT_MANIFEST.json",
    )
    parser.add_argument("--skip-release-manifest", action="store_true")
    parser.add_argument("--skip-number-check", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-authority-map", action="store_true")
    parser.add_argument(
        "--strict-file-set",
        action="store_true",
        help="fail if the extracted release contains an unlisted file",
    )
    args = parser.parse_args()

    failures: list[str] = []
    if not args.skip_authority_map:
        authority_failures = verify_authority_map(AUTHORITY_MAP)
        print("[authority index]")
        if authority_failures:
            for failure in authority_failures:
                print(f"FAIL: {failure}")
            failures.extend(authority_failures)
        else:
            print("PASS: every indexed authority exists and has stated integrity")

    if not args.skip_release_manifest:
        integrity = verify_release_manifest(
            args.manifest.resolve(), strict_file_set=args.strict_file_set
        )
        print("[release integrity]")
        if integrity:
            for failure in integrity:
                print(f"FAIL: {failure}")
            failures.extend(integrity)
        else:
            print("PASS: every recorded release file matches its size and SHA-256")

    if not args.skip_number_check:
        passed, _ = run_check(
            "manuscript claim authorities",
            [sys.executable, "scripts/check_numbers.py"],
        )
        if not passed:
            failures.append("manuscript claim-authority check failed")

    if not args.skip_tests:
        passed, _ = run_check(
            "unit contracts",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
        )
        if not passed:
            failures.append("unit tests failed")

    print("\n[artifact verification]")
    if failures:
        print(f"FAIL: {len(failures)} check group or integrity failure(s)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
