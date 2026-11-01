"""Scanning: from sources to a ranked list of findings about enrolled users.

A scan takes one or more (source, query) targets, fetches every media item they
lead to, analyses each through the same image or video pipeline a single upload
goes through, and returns one report ordered by urgency.

Three rules keep a scan from turning into a copy of the internet:

Only what concerns an enrolled user is kept
    Items whose verdict is ``unrelated`` or ``inconclusive`` are counted and
    discarded; their bytes are never stored. Flagged items keep their evidence
    record and, unless disabled, a copy of the exact bytes analysed, named by
    their SHA-256 so the record can be checked against it.
The same bytes are analysed once
    Media served at several URLs is recognised by hash and reported once, with
    every URL it was found at.
A failing item does not end the scan
    Blocked, unreachable, oversized or undecodable items are listed with their
    reason, and the scan continues. A link that looked like media but served a
    page is only counted: following it was a guess, not a failure.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepshield.config import DeepShieldConfig
from deepshield.exceptions import InvalidMediaError, NotMediaError, SourceError
from deepshield.logging_utils import get_logger
from deepshield.sources.base import ContentKind, ContentSource, FetchedContent
from deepshield.storage import build_evidence_repository
from deepshield.types import EvidenceRecord, RiskLevel, Verdict, utc_now

logger = get_logger(__name__)

SCAN_SUBDIR = "scans"
UNFLAGGED = frozenset({Verdict.UNRELATED, Verdict.INCONCLUSIVE})
LEVEL_ORDER = {level: rank for rank, level in enumerate(RiskLevel)}


@dataclass
class ScanFinding:
    """One analysed item and what the verdict engine concluded about it."""

    uri: str
    kind: str
    sha256: str
    verdict: str
    risk_level: str
    subject_user_id: str | None
    face_similarity: float | None
    summary: str
    found_on: str | None = None
    also_at: list[str] = field(default_factory=list)
    analysis_id: str | None = None
    saved_copy: str | None = None

    @property
    def flagged(self) -> bool:
        """Return whether the finding concerns an enrolled user."""
        return Verdict(self.verdict) not in UNFLAGGED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "uri": self.uri,
            "found_on": self.found_on,
            "also_at": list(self.also_at),
            "kind": self.kind,
            "sha256": self.sha256,
            "verdict": self.verdict,
            "risk_level": self.risk_level,
            "subject_user_id": self.subject_user_id,
            "face_similarity": self.face_similarity,
            "summary": self.summary,
            "analysis_id": self.analysis_id,
            "saved_copy": self.saved_copy,
        }


@dataclass
class ScanReport:
    """Everything one scan found, flagged findings first."""

    scan_id: str
    targets: list[dict[str, str]]
    user_id: str | None
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    discovered: int = 0
    duplicates: int = 0
    not_media: int = 0
    findings: list[ScanFinding] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    def flagged(self) -> list[ScanFinding]:
        """Return flagged findings, most urgent first, then most similar."""
        return sorted(
            (f for f in self.findings if f.flagged),
            key=lambda f: (-LEVEL_ORDER[RiskLevel(f.risk_level)], -(f.face_similarity or 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON report."""
        flagged = self.flagged()
        return {
            "scan_id": self.scan_id,
            "targets": list(self.targets),
            "user_id": self.user_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "counts": {
                "discovered": self.discovered,
                "analysed": len(self.findings),
                "duplicates": self.duplicates,
                "not_media": self.not_media,
                "flagged": len(flagged),
                "failures": len(self.failures),
            },
            "flagged": [f.to_dict() for f in flagged],
            "failures": list(self.failures),
            "limitations": [
                "Only content reachable from the given targets was examined; this is not "
                "a search of the web.",
                "Unflagged items were analysed and discarded; their bytes were not kept.",
                "A flagged face match does not establish that the content is synthetic, "
                "nor that it was made from the user's photographs.",
            ],
        }


class ScanRunner:
    """Fetches, analyses and reports on every item a set of targets leads to."""

    def __init__(self, config: DeepShieldConfig, analysis: Any = None) -> None:
        """Build or accept the analysis pipeline shared by every item."""
        from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline

        self.config = config
        self.analysis = analysis or DefaultAnalysisPipeline(config)
        self.evidence = build_evidence_repository(config)
        self.results_dir = Path(config.runtime.results_dir) / SCAN_SUBDIR
        self.media_dir = Path(config.runtime.data_dir) / SCAN_SUBDIR

    def _analyse(self, fetched: FetchedContent, user_id: str | None) -> EvidenceRecord:
        """Run the image or video pipeline on one fetched item."""
        if fetched.item.kind is ContentKind.VIDEO:
            record = self.analysis.analyze_video(fetched.path, user_id)
        else:
            record = self.analysis.analyze_image(fetched.path, user_id)
        record.source_id = fetched.final_uri
        return record

    def _keep(self, report: ScanReport, fetched: FetchedContent) -> str:
        """Copy the analysed bytes of a flagged item into the scan's evidence folder."""
        directory = self.media_dir / report.scan_id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{fetched.sha256}{fetched.path.suffix.lower()}"
        shutil.copyfile(fetched.path, destination)
        return str(destination)

    def run(
        self,
        targets: list[tuple[ContentSource, str]],
        user_id: str | None = None,
        keep_flagged: bool = True,
    ) -> ScanReport:
        """Scan every target and return the report, also written to disk.

        Raises:
            IdentityNotFoundError: If ``user_id`` names an unenrolled user.

        """
        report = ScanReport(
            scan_id=uuid.uuid4().hex[:16],
            targets=[{"source": source.name, "query": query} for source, query in targets],
            user_id=user_id,
        )
        if user_id is not None:
            self.analysis.identities.require(user_id)
        by_hash: dict[str, ScanFinding] = {}

        with tempfile.TemporaryDirectory(prefix="deepshield-scan-") as scratch:
            workspace = Path(scratch)
            for source, query in targets:
                already_skipped = len(getattr(source, "skipped", []))
                try:
                    items = list(source.discover(query))
                except SourceError as exc:
                    report.failures.append({"uri": query, "error": str(exc)})
                    continue
                for url, reason in getattr(source, "skipped", [])[already_skipped:]:
                    report.failures.append({"uri": url, "error": reason})
                report.discovered += len(items)

                for item in items:
                    try:
                        fetched = source.fetch(item, workspace)
                    except NotMediaError:
                        report.not_media += 1
                        continue
                    except SourceError as exc:
                        report.failures.append({"uri": item.uri, "error": str(exc)})
                        continue
                    if fetched.sha256 in by_hash:
                        by_hash[fetched.sha256].also_at.append(item.uri)
                        report.duplicates += 1
                        continue
                    try:
                        record = self._analyse(fetched, user_id)
                    except InvalidMediaError as exc:
                        report.failures.append({"uri": item.uri, "error": str(exc)})
                        continue
                    assert record.risk is not None
                    finding = ScanFinding(
                        uri=fetched.final_uri,
                        kind=fetched.item.kind.value,
                        sha256=fetched.sha256,
                        verdict=record.risk.verdict.value,
                        risk_level=record.risk.risk_level.value,
                        subject_user_id=record.risk.subject_user_id,
                        face_similarity=record.face_similarity,
                        summary=record.summary,
                        found_on=item.found_on,
                    )
                    if finding.flagged:
                        finding.analysis_id = self.evidence.save(record)
                        if keep_flagged:
                            finding.saved_copy = self._keep(report, fetched)
                    by_hash[fetched.sha256] = finding
                    report.findings.append(finding)

        report.finished_at = utc_now()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / f"{report.scan_id}.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "scan %s: %d discovered, %d analysed, %d flagged, %d failures",
            report.scan_id,
            report.discovered,
            len(report.findings),
            len(report.flagged()),
            len(report.failures),
        )
        return report
