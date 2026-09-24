"""Phase 16: content sources, the fetch policy and the bounded crawler.

Every web test runs against an HTTP server started on the loopback interface
for the test, so nothing leaves the machine. Because loopback is a private
address, those tests opt in with ``allow_private_networks``; the tests of the
default policy check that the same server is refused without it.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from deepshield.config import SourcesConfig
from deepshield.exceptions import BlockedSourceError, SourceError
from deepshield.sources import ContentItem, ContentKind, FolderSource, HttpFetcher, WebSource
from deepshield.sources.web import extract_links, media_type, normalise


def png(color: int = 120, size: int = 32) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (color, 40, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


class Site:
    """A throwaway website: files on disk plus scripted responses."""

    def __init__(self, root: Path) -> None:
        """Serve ``root`` once a server is attached."""
        self.root = root
        self.routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
        self.requests: list[str] = []
        self.server: ThreadingHTTPServer | None = None

    @property
    def base(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def other_host_base(self) -> str:
        assert self.server is not None
        return f"http://localhost:{self.server.server_address[1]}"

    def write(self, name: str, content: bytes | str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)


@pytest.fixture
def site(tmp_path: Path) -> Iterator[Site]:
    state = Site(tmp_path / "www")
    state.root.mkdir()

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            return

        def do_GET(self) -> None:
            state.requests.append(self.path)
            if self.path in state.routes:
                status, headers, body = state.routes[self.path]
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(state.root)))
    state.server = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()


def local_config(**overrides: object) -> SourcesConfig:
    values: dict[str, object] = {
        "allow_private_networks": True,
        "crawl_delay_seconds": 0.0,
        "timeout_seconds": 5.0,
    }
    values.update(overrides)
    return SourcesConfig(**values)  # type: ignore[arg-type]


def discover(source: WebSource, url: str) -> list[ContentItem]:
    return list(source.discover(url))


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.com/a.jpg", "javascript:alert(1)", "http://"]
)
def test_only_http_urls_with_a_host_are_accepted(url: str) -> None:
    with pytest.raises(BlockedSourceError):
        normalise(url)


def test_fragments_are_dropped() -> None:
    assert normalise("https://example.com/a.jpg#top") == "https://example.com/a.jpg"


def test_media_type_prefers_the_header_and_falls_back_to_the_suffix() -> None:
    assert media_type("image/png; charset=binary", "https://x/y") == (ContentKind.IMAGE, ".png")
    assert media_type("application/octet-stream", "https://x/clip.mp4") == (
        ContentKind.VIDEO,
        ".mp4",
    )
    assert media_type(None, "https://x/photo.jpeg") == (ContentKind.IMAGE, ".jpeg")
    assert media_type("text/html", "https://x/photo.jpg") is None


def test_links_and_media_are_extracted_from_html() -> None:
    html = """
    <html><head>
      <base href="https://site.test/gallery/">
      <meta property="og:image" content="/cover.jpg">
      <meta property="og:video" content="https://cdn.test/v/intro">
    </head><body>
      <a href="next.html#part">next</a>
      <a href="/files/clip.mp4">clip</a>
      <a href="javascript:void(0)">nothing</a>
      <img src="a.png" srcset="a-small.png 1x, a-large.png 2x">
      <img src="data:image/png;base64,AAAA">
      <video src="talk.webm" poster="poster.jpg"><source src="talk.mp4" type="video/mp4"></video>
      <picture><source srcset="b.webp" type="image/webp"></picture>
    </body></html>
    """
    pages, media = extract_links(html, "https://site.test/index.html")
    assert pages == ["https://site.test/gallery/next.html"]
    found = dict(media)
    assert found["https://site.test/cover.jpg"] is ContentKind.IMAGE
    assert found["https://cdn.test/v/intro"] is ContentKind.VIDEO
    assert found["https://site.test/files/clip.mp4"] is ContentKind.VIDEO
    assert found["https://site.test/gallery/a-large.png"] is ContentKind.IMAGE
    assert found["https://site.test/gallery/talk.webm"] is ContentKind.VIDEO
    assert found["https://site.test/gallery/poster.jpg"] is ContentKind.IMAGE
    assert found["https://site.test/gallery/b.webp"] is ContentKind.IMAGE
    assert not any(url.startswith("data:") for url in found)


def test_a_page_yields_its_media_without_following_links(site: Site) -> None:
    site.write("index.html", '<img src="a.png"><img src="b.png"><a href="more.html">more</a>')
    site.write("more.html", '<img src="c.png">')
    for name in ("a.png", "b.png", "c.png"):
        site.write(name, png())
    items = discover(WebSource(local_config(), max_depth=0), f"{site.base}/index.html")
    assert [item.uri.rsplit("/", 1)[1] for item in items] == ["a.png", "b.png"]
    assert all(item.found_on == f"{site.base}/index.html" for item in items)


def test_depth_follows_same_host_links_only(site: Site) -> None:
    site.write(
        "index.html",
        f'<a href="more.html">more</a><a href="{site.other_host_base}/away.html">away</a>',
    )
    site.write("more.html", '<img src="c.png">')
    site.write("away.html", '<img src="d.png">')
    items = discover(WebSource(local_config(), max_depth=1), f"{site.base}/index.html")
    assert [item.uri.rsplit("/", 1)[1] for item in items] == ["c.png"]
    assert "/away.html" not in site.requests


def test_other_hosts_are_followed_when_asked(site: Site) -> None:
    site.write("index.html", f'<a href="{site.other_host_base}/away.html">away</a>')
    site.write("away.html", '<img src="d.png">')
    source = WebSource(local_config(), max_depth=1, same_host=False)
    items = discover(source, f"{site.base}/index.html")
    assert [item.uri.rsplit("/", 1)[1] for item in items] == ["d.png"]


def test_robots_txt_is_honoured(site: Site) -> None:
    site.write("robots.txt", "User-agent: *\nDisallow: /private/\n")
    site.write("index.html", '<a href="private/secret.html">s</a><img src="private/x.png">')
    site.write("private/secret.html", '<img src="y.png">')
    site.write("private/x.png", png())
    source = WebSource(local_config(), max_depth=1)
    items = discover(source, f"{site.base}/index.html")
    assert "/private/secret.html" not in site.requests
    assert any("disallows" in reason for _, reason in source.skipped)
    blocked = [item for item in items if item.uri.endswith("private/x.png")]
    with pytest.raises(BlockedSourceError, match="disallows"):
        source.fetch(blocked[0], site.root.parent / "out")
    assert "/private/x.png" not in site.requests


def test_an_unreadable_robots_txt_blocks_the_site(site: Site) -> None:
    site.routes["/robots.txt"] = (503, {"Content-Type": "text/plain"}, b"down")
    site.write("index.html", '<img src="a.png">')
    with pytest.raises(BlockedSourceError, match="could not be read"):
        discover(WebSource(local_config(), max_depth=0), f"{site.base}/index.html")
    assert "/index.html" not in site.requests


def test_a_missing_robots_txt_allows_everything(site: Site) -> None:
    site.write("a.png", png())
    items = discover(WebSource(local_config()), f"{site.base}/a.png")
    assert len(items) == 1


def test_private_networks_are_refused_by_default(site: Site) -> None:
    site.write("a.png", png())
    with pytest.raises(BlockedSourceError, match="non-public address"):
        discover(WebSource(SourcesConfig(crawl_delay_seconds=0.0)), f"{site.base}/a.png")
    assert site.requests == []


@pytest.mark.parametrize(
    ("location", "error"),
    [("file:///etc/passwd", SourceError), ("ftp://example.com/a.jpg", BlockedSourceError)],
)
def test_a_redirect_to_another_scheme_is_refused(
    site: Site, location: str, error: type[Exception]
) -> None:
    site.routes["/go.jpg"] = (302, {"Location": location}, b"")
    source = WebSource(local_config())
    item = ContentItem(uri=f"{site.base}/go.jpg", kind=ContentKind.IMAGE, source="web")
    with pytest.raises(error):
        source.fetch(item, site.root.parent / "out")
    assert not list((site.root.parent / "out").glob("*"))


def test_downloads_are_hashed_and_typed(site: Site) -> None:
    site.write("a.png", png(10))
    source = WebSource(local_config())
    item = discover(source, f"{site.base}/a.png")[0]
    fetched = source.fetch(item, site.root.parent / "out")
    assert fetched.path.suffix == ".png"
    assert fetched.size_bytes == len(png(10))
    assert len(fetched.sha256) == 64
    assert fetched.item.kind is ContentKind.IMAGE


def test_oversized_downloads_are_refused(site: Site) -> None:
    site.write("big.png", b"\0" * (1024 * 1024 + 1))
    source = WebSource(local_config(max_download_mb=1))
    item = ContentItem(uri=f"{site.base}/big.png", kind=ContentKind.IMAGE, source="web")
    with pytest.raises(SourceError, match="limit"):
        source.fetch(item, site.root.parent / "out")
    assert not list((site.root.parent / "out").glob("*"))


def test_a_non_media_response_is_refused(site: Site) -> None:
    site.routes["/fake.jpg"] = (200, {"Content-Type": "text/html"}, b"<html></html>")
    item = ContentItem(uri=f"{site.base}/fake.jpg", kind=ContentKind.IMAGE, source="web")
    with pytest.raises(SourceError, match="not a supported image or video"):
        WebSource(local_config()).fetch(item, site.root.parent / "out")


def test_media_and_page_caps_bound_the_crawl(site: Site) -> None:
    site.write("index.html", "".join(f'<img src="m{i}.png">' for i in range(10)))
    items = discover(WebSource(local_config(), max_media=3), f"{site.base}/index.html")
    assert len(items) == 3
    site.write("p0.html", '<a href="p1.html">1</a><a href="p2.html">2</a><a href="p3.html">3</a>')
    for i in range(1, 4):
        site.write(f"p{i}.html", f'<img src="q{i}.png">')
    source = WebSource(local_config(), max_depth=1, max_pages=2)
    discover(source, f"{site.base}/p0.html")
    pages = [path for path in site.requests if path.startswith("/p")]
    assert len(pages) == 2


def test_requests_to_one_host_are_spaced(site: Site) -> None:
    site.write("a.png", png())
    site.write("b.png", png(20))
    waits: list[float] = []
    now = [100.0]
    fetcher = HttpFetcher(
        local_config(crawl_delay_seconds=2.0), sleep=waits.append, clock=lambda: now[0]
    )
    source = WebSource(local_config(crawl_delay_seconds=2.0), fetcher=fetcher)
    out = site.root.parent / "out"
    for name in ("a.png", "b.png"):
        item = ContentItem(uri=f"{site.base}/{name}", kind=ContentKind.IMAGE, source="web")
        source.fetch(item, out)
    assert waits and all(wait == pytest.approx(2.0) for wait in waits)


def test_a_longer_crawl_delay_in_robots_txt_wins(site: Site) -> None:
    site.write("robots.txt", "User-agent: *\nCrawl-delay: 5\n")
    fetcher = HttpFetcher(local_config(crawl_delay_seconds=1.0))
    assert fetcher.check_allowed(f"{site.base}/a.png") == pytest.approx(5.0)


def test_folder_source_finds_media_recursively(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.jpg").write_bytes(png())
    (tmp_path / "nested" / "b.mp4").write_bytes(b"\0")
    (tmp_path / "notes.txt").write_text("x")
    items = list(FolderSource().discover(str(tmp_path)))
    assert [(Path(i.uri).name, i.kind) for i in items] == [
        ("a.jpg", ContentKind.IMAGE),
        ("b.mp4", ContentKind.VIDEO),
    ]
    fetched = FolderSource().fetch(items[0], tmp_path / "unused")
    assert fetched.path == tmp_path / "a.jpg"
    assert len(fetched.sha256) == 64


def test_folder_source_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not a directory"):
        list(FolderSource().discover(str(tmp_path / "missing")))
