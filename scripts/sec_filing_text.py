"""SEC EDGAR access and Item 1 extraction for the vintage-text pipeline.

The client follows the SEC request limit and uses the researcher's User-Agent.
Extraction can be tested without network requests. It uses a deterministic
heuristic, not a general-purpose filing parser.
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests


SEC_SUBMISSIONS = "https://data.sec.gov/submissions"
SEC_ARCHIVES = "https://www.sec.gov/Archives"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"
SUPPORTED_FORMS = {"10-K", "10-K405", "10-KSB", "10-KT"}
MIN_ITEM1_CHARS = 100
MAX_ITEM1_CHARS = 8000
MIN_SECTION_GAP = 500

ITEM1_START = re.compile(
    r"\bItem\s*1[.\s]+(?:Business|General|Overview)\b",
    re.IGNORECASE,
)
ITEM1_END = re.compile(r"\bItem\s*1A\b|\bItem\s*2\b", re.IGNORECASE)


@dataclass(frozen=True)
class FilingReference:
    cik: int
    form: str
    filing_date: date
    accession_number: str
    primary_document: str

    @property
    def accession_without_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def filing_url(self) -> str:
        return (
            f"{SEC_ARCHIVES}/edgar/data/{self.cik}/"
            f"{self.accession_without_dashes}/{self.primary_document}"
        )


class SecClient:
    """Rate-limited SEC client with bounded retries.

    Parameters
    ----------
    user_agent:
        A researcher or project name and a monitored contact address, as
        requested by SEC fair-access guidance.
    max_requests_per_second:
        Local request limit. The SEC maximum is ten; the default is lower to
        allow other requests from the same host.
    """

    def __init__(
        self,
        user_agent: str,
        *,
        max_requests_per_second: float = 2.0,
        timeout_seconds: float = 45.0,
        max_attempts: int = 4,
    ) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise ValueError(
                "SEC User-Agent must identify the project and include a contact address"
            )
        if not 0.1 <= max_requests_per_second <= 10.0:
            raise ValueError("SEC request rate must be between 0.1 and 10 per second")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.minimum_interval = 1.0 / max_requests_per_second
        self.last_request = 0.0
        # Cache each company's filings to avoid fetching its metadata again
        # for every window cutoff.
        self._annual_filing_cache: dict[int, tuple[FilingReference, ...]] = {}
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent.strip(),
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/html,text/plain,*/*",
            }
        )

    def get(self, url: str) -> requests.Response:
        """Space requests and retry temporary HTTP errors when fetching from SEC."""
        for attempt in range(1, self.max_attempts + 1):
            elapsed = time.monotonic() - self.last_request
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            response = self.session.get(url, timeout=self.timeout_seconds)
            self.last_request = time.monotonic()
            if response.status_code == 200:
                return response
            if response.status_code not in {403, 429, 500, 502, 503, 504}:
                response.raise_for_status()
            if attempt == self.max_attempts:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else float(2 ** attempt)
            time.sleep(min(delay, 30.0))
        raise RuntimeError("unreachable SEC retry state")

    def ticker_cik_map(self) -> dict[str, int]:
        """Download the official SEC ticker/exchange mapping."""
        payload = self.get(SEC_TICKERS).json()
        fields = payload["fields"]
        ticker_idx = fields.index("ticker")
        cik_idx = fields.index("cik")
        result: dict[str, int] = {}
        for row in payload["data"]:
            ticker = str(row[ticker_idx]).strip().upper()
            if ticker:
                result[ticker] = int(row[cik_idx])
        return result

    def _scan_filing_table(
        self,
        table: dict[str, Any],
        cutoff: date,
    ) -> list[FilingReference]:
        references: list[FilingReference] = []
        rows = zip(
            table.get("form", []),
            table.get("filingDate", []),
            table.get("accessionNumber", []),
            table.get("primaryDocument", []),
        )
        for form, filing_date, accession, primary_document in rows:
            # Skip amendments because their Item 1 coverage can differ from
            # the complete annual filing.
            if form not in SUPPORTED_FORMS:
                continue
            try:
                filed = date.fromisoformat(str(filing_date))
            except ValueError:
                continue
            if filed >= cutoff or not accession or not primary_document:
                continue
            references.append(
                FilingReference(
                    cik=0,
                    form=str(form),
                    filing_date=filed,
                    accession_number=str(accession),
                    primary_document=str(primary_document),
                )
            )
        return references

    def most_recent_annual_filing(
        self,
        cik: int,
        cutoff: date,
    ) -> FilingReference | None:
        """Return the most recent supported annual filing strictly before cutoff."""
        if cik not in self._annual_filing_cache:
            payload = self.get(
                f"{SEC_SUBMISSIONS}/CIK{cik:010d}.json"
            ).json()
            filings = payload.get("filings", {})
            references = self._scan_filing_table(
                filings.get("recent", {}), date.max
            )
            for page in filings.get("files", []):
                name = str(page.get("name", ""))
                if not name:
                    continue
                page_payload = self.get(f"{SEC_SUBMISSIONS}/{name}").json()
                references.extend(
                    self._scan_filing_table(page_payload, date.max)
                )
            self._annual_filing_cache[cik] = tuple(
                FilingReference(
                    cik=cik,
                    form=row.form,
                    filing_date=row.filing_date,
                    accession_number=row.accession_number,
                    primary_document=row.primary_document,
                )
                for row in references
            )
        eligible = [
            row
            for row in self._annual_filing_cache[cik]
            if row.filing_date < cutoff
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda row: row.filing_date)

    def filing_text(self, reference: FilingReference) -> str:
        return self.get(reference.filing_url).text


def html_to_text(raw: str) -> str:
    """Remove markup and normalize whitespace without executing filing content."""
    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1\s*>",
        " ",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]{0,4000}>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_primary_document(sgml: str) -> str:
    """Return the annual-report document from a complete-submission SGML file."""
    pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>([^\n<]+).*?<TEXT>(.*?)</TEXT>",
        re.IGNORECASE | re.DOTALL,
    )
    fallback = ""
    for match in pattern.finditer(sgml):
        form = match.group(1).strip()
        document = match.group(2).strip()
        if not fallback:
            fallback = document
        if form in SUPPORTED_FORMS:
            return document
    return fallback or sgml


def extract_item1(raw: str, *, character_cap: int = MAX_ITEM1_CHARS) -> str | None:
    """Extract the longest plausible Item 1 body and apply the collection cap.

    Multiple Item 1 headers are common because filings contain a table of
    contents. Candidates ending fewer than 500 characters after the header are
    treated as table entries. The longest remaining candidate is retained.
    """
    if "<DOCUMENT>" in raw[:10000].upper():
        raw = extract_primary_document(raw)
    text = html_to_text(raw) if "<" in raw[:2000] else re.sub(r"\s+", " ", raw).strip()
    candidates: list[str] = []
    for start in ITEM1_START.finditer(text):
        end = ITEM1_END.search(text, start.end() + MIN_SECTION_GAP)
        end_position = end.start() if end else min(len(text), start.start() + 12000)
        candidate = text[start.start() : end_position].strip()
        if len(candidate) >= MIN_ITEM1_CHARS:
            candidates.append(candidate[:character_cap])
    if not candidates:
        return None
    return max(candidates, key=len)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
