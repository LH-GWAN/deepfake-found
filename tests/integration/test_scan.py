"""Phase 16 end to end: scanning folders and URLs into ranked findings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from deepshield.config import DeepShieldConfig, default_config
from deepshield.exceptions import IdentityNotFoundError, InvalidMediaError
from deepshield.media import save_image
from deepshield.pipeline.scan_pipeline import ScanRunner
from deepshield.sources import FolderSource, WebSource
from deepshield.types import EvidenceRecord, MediaType, RiskAssessment, RiskLevel, Verdict
from tests.conftest import synthetic_photo

pytestmark = pytest.mark.integration

VERDICTS = {
    "match": (Verdict.IDENTITY_MATCH, RiskLevel.MEDIUM, 0.71),
    "altered": (Verdict.OWN_ALTERED, RiskLevel.HIGH, 0.12),
    "stranger": (Verdict.UNRELATED, RiskLevel.LOW, 0.05),
}


class Identities:
    def require(self, user_id: str) -> None:
        if user_id != "u1":
            raise IdentityNotFoundError(f"no enrolled identity for user '{user_id}'")


class ScriptedAnalysis:
    """Returns the verdict a file is labelled with, or rejects 'broken' files.

    Local files are labelled by their name; downloaded ones, which are stored
    under a hash of their URL, by the SHA-256 of their bytes in ``by_hash``.
    """

    def __init__(self, by_hash: dict[str, str] | None = None) -> None:
        """Start with no files analysed."""
        self.identities = Identities()
        self.by_hash = by_hash or {}
        self.analysed: list[str] = []

    def _record(self, path: Path, media: MediaType) -> EvidenceRecord:
        self.analysed.append(path.name)
        if path.name.startswith("broken"):
            raise InvalidMediaError(f"unsupported or corrupt image: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        label = self.by_hash.get(digest, path.stem.split("_")[0])
        verdict, level, similarity = VERDICTS[label]
        return EvidenceRecord(
            source_id=path.name,
            media_type=media,
            face_similarity=similarity,
            risk=RiskAssessment(verdict=verdict, risk_level=level, subject_user_id="u1"),
            summary=f"{verdict.value} summary",
        )

    def analyze_image(self, path: Path, user_id: str | None = None) -> EvidenceRecord:
        return self._record(Path(path), MediaType.IMAGE)

    def analyze_video(self, path: Path, user_id: str | None = None) -> EvidenceRecord:
        return self._record(Path(path), MediaType.VIDEO)


@pytest.fixture
def config(tmp_path: Path) -> DeepShieldConfig:
    base = default_config()
    return base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={"data_dir": tmp_path / "data", "results_dir": tmp_path / "results"}
            ),
            "sources": base.sources.model_copy(
                update={"allow_private_networks": True, "crawl_delay_seconds": 0.0}
            ),
        }
    )


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "inbox"
    for index, name in enumerate(["match_a", "altered_b", "stranger_c", "stranger_d"]):
        save_image(synthetic_photo(seed=index + 1, size=64), root / f"{name}.png")
    save_image(synthetic_photo(seed=1, size=64), root / "copies" / "match_again.png")
    (root / "broken_e.png").write_bytes(b"not an image")
    return root


def run(config: DeepShieldConfig, folder: Path, **kwargs: Any) -> tuple[Any, ScriptedAnalysis]:
    analysis = ScriptedAnalysis()
    report = ScanRunner(config, analysis=analysis).run(
        [(FolderSource(), str(folder))], "u1", **kwargs
    )
    return report, analysis


def test_flagged_findings_come_first_by_urgency(config, folder: Path) -> None:
    report, _ = run(config, folder)
    flagged = report.to_dict()["flagged"]
    assert [f["verdict"] for f in flagged] == ["own_altered", "identity_match"]
    assert report.to_dict()["counts"] == {
        "discovered": 6, "analysed": 4, "duplicates": 1, "not_media": 0, "flagged": 2,
        "failures": 1,
    }


def test_identical_bytes_are_analysed_once(config, folder: Path) -> None:
    report, analysis = run(config, folder)
    copies = {"match_a.png", "match_again.png"}
    assert len(copies & set(analysis.analysed)) == 1
    match = next(f for f in report.findings if f.verdict == "identity_match")
    assert {Path(match.uri).name, Path(match.also_at[0]).name} == copies


def test_only_flagged_items_are_kept(config, folder: Path) -> None:
    report, _ = run(config, folder)
    kept = sorted(Path(f.saved_copy).name for f in report.flagged() if f.saved_copy)
    stored = sorted(p.name for p in (Path(config.runtime.data_dir) / "scans").rglob("*.png"))
    assert kept == stored
    assert len(stored) == 2
    assert all(f.analysis_id for f in report.flagged())
    assert all(f.analysis_id is None for f in report.findings if not f.flagged)


def test_no_keep_stores_evidence_but_not_media(config, folder: Path) -> None:
    report, _ = run(config, folder, keep_flagged=False)
    assert not (Path(config.runtime.data_dir) / "scans").exists()
    assert all(f.analysis_id for f in report.flagged())


def test_a_bad_item_is_reported_and_the_scan_continues(config, folder: Path) -> None:
    report, _ = run(config, folder)
    assert report.failures[0]["uri"].endswith("broken_e.png")
    assert "corrupt" in report.failures[0]["error"]


def test_the_report_is_written_to_disk(config, folder: Path) -> None:
    report, _ = run(config, folder)
    stored = json.loads(
        (Path(config.runtime.results_dir) / "scans" / f"{report.scan_id}.json").read_text()
    )
    assert stored["counts"]["flagged"] == 2
    assert any("not a search of the web" in line for line in stored["limitations"])


def test_an_unknown_user_stops_the_scan_before_any_fetch(config, folder: Path) -> None:
    analysis = ScriptedAnalysis()
    with pytest.raises(IdentityNotFoundError):
        ScanRunner(config, analysis=analysis).run([(FolderSource(), str(folder))], "nobody")
    assert analysis.analysed == []


def test_an_unreachable_target_is_a_failure_not_a_crash(config) -> None:
    report = ScanRunner(config, analysis=ScriptedAnalysis()).run(
        [(WebSource(config.sources), "ftp://example.com/x")], "u1"
    )
    assert report.failures[0]["uri"] == "ftp://example.com/x"
    assert report.findings == []


def test_a_crawled_page_is_scanned(config, tmp_path: Path) -> None:
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    root = tmp_path / "www"
    labels = {}
    for seed, name in ((3, "match_x.png"), (4, "stranger_y.png")):
        written = save_image(synthetic_photo(seed=seed, size=64), root / name)
        labels[hashlib.sha256(written.read_bytes()).hexdigest()] = name.split("_")[0]
    (root / "index.html").write_text('<img src="match_x.png"><img src="stranger_y.png">')

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
        report = ScanRunner(config, analysis=ScriptedAnalysis(labels)).run(
            [(WebSource(config.sources, max_depth=0), url)], "u1"
        )
    finally:
        server.shutdown()
        server.server_close()
    flagged = report.flagged()
    assert [f.verdict for f in flagged] == ["identity_match"]
    assert flagged[0].uri.endswith("match_x.png")
    assert flagged[0].found_on == url


def test_skips_are_reported_once_when_a_source_serves_several_targets(config) -> None:
    from collections.abc import Iterator

    from deepshield.sources import ContentItem, ContentSource, FetchedContent

    class Skipping(ContentSource):
        name = "skipping"

        def __init__(self) -> None:
            """Start with nothing skipped."""
            self.skipped: list[tuple[str, str]] = []

        def discover(self, query: str) -> Iterator[ContentItem]:
            self.skipped.append((f"{query}/page", "disallowed"))
            return iter([])

        def fetch(self, item: ContentItem, directory: Path) -> FetchedContent:
            raise AssertionError("nothing to fetch")

    source = Skipping()
    report = ScanRunner(config, analysis=ScriptedAnalysis()).run(
        [(source, "a"), (source, "b")], "u1"
    )
    assert [f["uri"] for f in report.failures] == ["a/page", "b/page"]


def test_a_link_that_serves_a_page_is_counted_not_failed(config, tmp_path: Path) -> None:
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    root = tmp_path / "www"
    root.mkdir()
    (root / "wiki").mkdir()
    (root / "wiki" / "File:photo.jpg").write_text("<html>a description page</html>")
    (root / "index.html").write_text('<a href="/wiki/File:photo.jpg">photo</a>')

    class HtmlEverywhere(SimpleHTTPRequestHandler):
        def guess_type(self, path: str | Path) -> str:
            return "text/html"

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(HtmlEverywhere, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
        report = ScanRunner(config, analysis=ScriptedAnalysis()).run(
            [(WebSource(config.sources, max_depth=0), url)], "u1"
        )
    finally:
        server.shutdown()
        server.server_close()
    counts = report.to_dict()["counts"]
    assert (counts["discovered"], counts["not_media"], counts["failures"]) == (1, 1, 0)
