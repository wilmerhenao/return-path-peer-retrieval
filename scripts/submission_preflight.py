"""Run local ICAIF/ACM submission checks on the compiled manuscript."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
TEX = PAPER_DIR / "main.tex"
PDF = PAPER_DIR / "main.pdf"
LOG = PAPER_DIR / "main.log"
ABSTRACT = PAPER_DIR / "abstract_submitted.tex"
BIB = PAPER_DIR / "references.bib"


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []

    if not PDF.is_file():
        print("FAIL: paper/main.pdf does not exist")
        return 1

    tex = TEX.read_text(encoding="utf-8")
    if r"\documentclass[sigconf,anonymous]{acmart}" not in tex:
        failures.append("manuscript is not using anonymous ACM sigconf mode")
    for expected, label in (
        (r"\author{Anonymous}", "anonymous author"),
        (r"\institution{Anonymous Institution}", "anonymous affiliation"),
        (r"\email{anonymous@example.com}", "anonymous email"),
    ):
        if expected not in tex:
            failures.append(f"{label} placeholder is missing")
    if "Generative-AI disclosure" not in tex:
        failures.append("visible generative-AI disclosure is missing")
    if "ANONYMIZED-REPOSITORY-URL" in tex:
        failures.append("repository URL placeholder is unresolved")
    if "anonymous.4open.science" in tex:
        failures.append(
            "external anonymous-repository link remains in the self-contained review PDF"
        )
    if re.search(r"\\appendix\b|\\section\*?\{Appendix", tex):
        failures.append("appendix detected; the eight-page submission must be self-contained")

    source_identity_patterns = (
        (r"/Users/[^/\s]+/", "macOS home-directory path"),
        (r"/home/[^/\s]+/", "Unix home-directory path"),
        (r"[A-Za-z]:\\Users\\[^\\\s]+\\", "Windows home-directory path"),
    )
    for source_path in (TEX, ABSTRACT, BIB):
        source = source_path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in source_identity_patterns:
            if re.search(pattern, source):
                failures.append(
                    f"{label} leaked in {source_path.relative_to(ROOT)}"
                )

    info = command("pdfinfo", str(PDF))
    creator_match = re.search(r"^Creator:\s*(.*)$", info, re.MULTILINE)
    creator = creator_match.group(1).strip() if creator_match else ""
    if "LaTeX with acmart 2026/06/27 v2.19" not in creator:
        failures.append(
            "PDF was not built with the validated ACM class "
            "(acmart 2026/06/27 v2.19)"
        )
    producer_match = re.search(r"^Producer:\s*(.*)$", info, re.MULTILINE)
    producer = producer_match.group(1).strip() if producer_match else ""
    if "pdfTeX" not in producer:
        failures.append(f"PDF producer is not pdfTeX: {producer or 'unknown'}")
    page_match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    pages = int(page_match.group(1)) if page_match else -1
    if not 1 <= pages <= 8:
        failures.append(f"page count is {pages}; ICAIF limit is 8 total pages")
    elif pages < 7:
        notes.append(
            f"manuscript is {pages} pages; the project target is 7--8 total pages"
        )
    size_match = re.search(r"^Page size:\s+(.+)$", info, re.MULTILINE)
    page_size = size_match.group(1).strip() if size_match else "unknown"
    if "612 x 792 pts (letter)" not in page_size:
        failures.append(f"unexpected page size: {page_size}")
    author_match = re.search(r"^Author:\s*(.*)$", info, re.MULTILINE)
    if author_match and author_match.group(1).strip() not in {"", "Anonymous"}:
        failures.append(f"non-anonymous PDF author metadata: {author_match.group(1)}")

    pdf_text = command("pdftotext", str(PDF), "-")
    for pattern, label in source_identity_patterns:
        if re.search(pattern, pdf_text):
            failures.append(f"{label} leaked in rendered PDF")
    if "ANONYMIZED-REPOSITORY-URL" in pdf_text:
        failures.append("repository URL placeholder is visible in rendered PDF")
    if "anonymous.4open.science" in pdf_text:
        failures.append(
            "external anonymous-repository link is visible in the review PDF"
        )

    fonts = command("pdffonts", str(PDF))
    font_rows = [
        line
        for line in fonts.splitlines()[2:]
        if line.strip() and not set(line.strip()) <= {"-"}
    ]
    for row in font_rows:
        flags = re.search(
            r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row
        )
        if not flags or flags.group(1) != "yes":
            failures.append(f"font is not embedded or could not be parsed: {row}")

    if LOG.is_file():
        log = LOG.read_text(encoding="utf-8", errors="replace")
        for pattern, label in (
            (r"LaTeX Warning: Citation .* undefined", "undefined citation"),
            (r"LaTeX Warning: Reference .* undefined", "undefined reference"),
            (r"There were undefined references", "undefined references"),
            (r"Label\(s\) may have changed", "labels need another LaTeX pass"),
        ):
            if re.search(pattern, log):
                failures.append(label)
        overfull = re.findall(r"Overfull \\[hv]box \(([^)]+)\)", log)
        if overfull:
            failures.append(
                f"{len(overfull)} overfull box warning(s): {', '.join(overfull)}"
            )
    else:
        failures.append("paper/main.log is missing")

    number_check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_numbers.py")],
        text=True,
        capture_output=True,
    )
    if number_check.returncode:
        failures.append("artifact-to-manuscript number check failed")
        details = "\n".join(
            part.strip()
            for part in (number_check.stdout, number_check.stderr)
            if part.strip()
        )
        if details:
            notes.append(details)

    print("ICAIF submission preflight")
    print(f"  Creator: {creator or 'unknown'}")
    print(f"  Producer: {producer or 'unknown'}")
    print(f"  Pages: {pages}/8")
    print(f"  Page size: {page_size}")
    print(f"  Embedded fonts checked: {len(font_rows)}")
    for note in notes:
        print(f"  NOTE: {note}")
    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
