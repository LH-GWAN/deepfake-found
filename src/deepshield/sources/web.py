"""The web as a content source: single URLs and a bounded, polite crawler.

Fetching arbitrary URLs is the riskiest thing this system does, so the rules
are enforced here rather than left to callers.

Only ``http`` and ``https``
    Other schemes are refused before a request is made and on every redirect.
No private networks
    Every address a host resolves to must be globally routable, and the address
    actually connected to is checked again after the connection opens, so a
    DNS answer that changes between the check and the connection cannot slip
    through. Proxies from the environment are ignored for the same reason.
    Tests and deployments that scan their own intranet opt in explicitly.
robots.txt is always honoured
    It is not a setting. A site whose robots.txt cannot be read because of a
    server or network error is treated as disallowing everything; a missing
    robots.txt allows everything, as the convention says. A longer
    ``Crawl-delay`` than the configured one wins.
Bounded work
    Pages, depth, media items and bytes per download are all capped, and
    requests to one host are spaced by the configured delay.
Pages are followed only on the starting host by default
    Media embedded in a visited page is fetched wherever it is served from,
    since that is what the page shows; links to other sites are not followed
    unless asked.

Nothing here logs in, solves challenges or evades blocking. A page that refuses
the request stays refused.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from deepshield.config import SourcesConfig
from deepshield.exceptions import BlockedSourceError, NotMediaError, SourceError
from deepshield.logging_utils import get_logger
from deepshield.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from deepshield.sources.base import ContentItem, ContentKind, ContentSource, FetchedContent

logger = get_logger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_PAGE_BYTES = 5 * 1024 * 1024
MAX_ROBOTS_BYTES = 512 * 1024
CHUNK_BYTES = 1 << 16
WEB_IMAGE_SUFFIXES = IMAGE_SUFFIXES | {".gif"}
# Vector art and icons: never a photograph, and not decodable by the pipeline.
NON_MEDIA_SUFFIXES = frozenset({".svg", ".svgz", ".ico"})
# An image declared smaller than this cannot hold a face the detector would find.
MIN_DECLARED_SIDE = 48
MEDIA_TYPES: dict[str, tuple[ContentKind, str]] = {
    "image/jpeg": (ContentKind.IMAGE, ".jpg"),
    "image/png": (ContentKind.IMAGE, ".png"),
    "image/webp": (ContentKind.IMAGE, ".webp"),
    "image/bmp": (ContentKind.IMAGE, ".bmp"),
    "image/gif": (ContentKind.IMAGE, ".gif"),
    "image/tiff": (ContentKind.IMAGE, ".tif"),
    "video/mp4": (ContentKind.VIDEO, ".mp4"),
    "video/webm": (ContentKind.VIDEO, ".webm"),
    "video/quicktime": (ContentKind.VIDEO, ".mov"),
    "video/x-matroska": (ContentKind.VIDEO, ".mkv"),
    "video/x-msvideo": (ContentKind.VIDEO, ".avi"),
    "video/x-m4v": (ContentKind.VIDEO, ".m4v"),
}
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
MEDIA_META = frozenset(
    {
        "og:image", "og:image:url", "og:image:secure_url", "twitter:image",
        "og:video", "og:video:url", "og:video:secure_url", "twitter:player:stream",
    }
)


def kind_from_suffix(url: str) -> ContentKind | None:
    """Guess the media kind from a URL's path suffix."""
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    if suffix in WEB_IMAGE_SUFFIXES:
        return ContentKind.IMAGE
    if suffix in VIDEO_SUFFIXES:
        return ContentKind.VIDEO
    return None


def media_type(content_type: str | None, url: str) -> tuple[ContentKind, str] | None:
    """Return the kind and file suffix a response should be stored as, if it is media."""
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in MEDIA_TYPES:
        return MEDIA_TYPES[mime]
    if mime and mime not in {"application/octet-stream", "binary/octet-stream"}:
        return None
    kind = kind_from_suffix(url)
    if kind is None:
        return None
    return kind, PurePosixPath(urlsplit(url).path).suffix.lower()


def normalise(url: str) -> str:
    """Return ``url`` without its fragment, refusing schemes other than http(s).

    Raises:
        BlockedSourceError: If the scheme is not http or https, or there is no host.

    """
    clean, _ = urldefrag(url.strip())
    parts = urlsplit(clean)
    if parts.scheme.lower() not in ALLOWED_SCHEMES or not parts.hostname:
        raise BlockedSourceError(f"only http(s) URLs with a host can be fetched: {url}")
    return clean


def origin(url: str) -> str:
    """Return scheme, host and port: the unit robots.txt and politeness apply to."""
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _check_address(address: str, allow_private: bool, host: str) -> None:
    """Refuse an address that is not globally routable unless explicitly allowed."""
    if allow_private:
        return
    ip = ipaddress.ip_address(address.split("%")[0])
    if not ip.is_global:
        raise BlockedSourceError(
            f"{host} resolves to {ip}, a non-public address; private networks are not "
            "fetched unless sources.allow_private_networks is enabled"
        )


class _GuardedHTTPConnection(http.client.HTTPConnection):
    """An HTTP connection that verifies the address it actually connected to."""

    def __init__(self, *args: Any, allow_private: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_private = allow_private

    def connect(self) -> None:
        super().connect()
        _check_address(self.sock.getpeername()[0], self.allow_private, self.host)


class _GuardedHTTPSConnection(http.client.HTTPSConnection):
    """An HTTPS connection that verifies the address it actually connected to."""

    def __init__(self, *args: Any, allow_private: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_private = allow_private

    def connect(self) -> None:
        super().connect()
        _check_address(self.sock.getpeername()[0], self.allow_private, self.host)


class _GuardedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self.allow_private = allow_private

    def http_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(
            partial(_GuardedHTTPConnection, allow_private=self.allow_private), req
        )


class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self.allow_private = allow_private

    def https_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(
            partial(_GuardedHTTPSConnection, allow_private=self.allow_private),
            req,
            context=self._context,  # type: ignore[attr-defined]
        )


class _SchemeCheckingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        normalise(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class Response:
    """What a completed request produced."""

    final_url: str
    content_type: str | None
    body: bytes


class HttpFetcher:
    """Policy-enforcing HTTP client shared by every web source."""

    def __init__(
        self,
        config: SourcesConfig,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build an opener with no proxy, no file or ftp handlers, and guarded sockets."""
        self.config = config
        self.sleep = sleep
        self.clock = clock
        self.max_download_bytes = config.max_download_mb * 1024 * 1024
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._opener = urllib.request.OpenerDirector()
        for handler in (
            urllib.request.ProxyHandler({}),
            urllib.request.UnknownHandler(),
            urllib.request.HTTPDefaultErrorHandler(),
            _SchemeCheckingRedirectHandler(),
            urllib.request.HTTPErrorProcessor(),
            _GuardedHTTPHandler(config.allow_private_networks),
            _GuardedHTTPSHandler(config.allow_private_networks),
        ):
            self._opener.add_handler(handler)

    def _resolve_check(self, url: str) -> None:
        """Refuse a host whose resolved addresses include a non-public one."""
        host = urlsplit(url).hostname or ""
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError as exc:
            raise SourceError(f"cannot resolve {host}: {exc}") from exc
        for info in infos:
            _check_address(str(info[4][0]), self.config.allow_private_networks, host)

    def _wait_turn(self, url: str, delay: float) -> None:
        """Space requests to one origin by at least ``delay`` seconds."""
        key = origin(url)
        last = self._last_request.get(key)
        if last is not None:
            remaining = delay - (self.clock() - last)
            if remaining > 0:
                self.sleep(remaining)
        self._last_request[key] = self.clock()

    def _open(self, url: str, delay: float) -> Any:
        """Open ``url`` under every policy check, returning the live response."""
        target = normalise(url)
        self._resolve_check(target)
        self._wait_turn(target, delay)
        request = urllib.request.Request(target, headers={"User-Agent": self.config.user_agent})
        try:
            return self._opener.open(request, timeout=self.config.timeout_seconds)
        except BlockedSourceError:
            raise
        except urllib.error.HTTPError as exc:
            raise SourceError(f"{target} answered HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, BlockedSourceError):
                raise reason from exc
            raise SourceError(f"could not fetch {target}: {reason}") from exc

    def _read_limited(self, response: Any, limit: int, url: str) -> bytes:
        """Read a response body, refusing one larger than ``limit`` bytes."""
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise SourceError(f"{url} is {int(declared)} bytes, over the {limit}-byte limit")
        chunks, total = [], 0
        while chunk := response.read(CHUNK_BYTES):
            total += len(chunk)
            if total > limit:
                raise SourceError(f"{url} exceeded the {limit}-byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def get(self, url: str, limit: int, delay: float | None = None) -> Response:
        """Fetch ``url`` into memory, subject to every policy and the byte limit."""
        pause = self.config.crawl_delay_seconds if delay is None else delay
        with self._open(url, pause) as response:
            body = self._read_limited(response, limit, url)
            return Response(
                final_url=response.geturl(),
                content_type=response.headers.get("Content-Type"),
                body=body,
            )

    def robots_for(self, url: str) -> RobotFileParser | None:
        """Return the parsed robots.txt of ``url``'s origin, cached; ``None`` means unreadable."""
        key = origin(url)
        if key in self._robots:
            return self._robots[key]
        parser: RobotFileParser | None = RobotFileParser()
        robots_url = f"{key}/robots.txt"
        try:
            response = self.get(robots_url, MAX_ROBOTS_BYTES, delay=0.0)
            assert parser is not None
            parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
        except BlockedSourceError:
            raise
        except SourceError as exc:
            cause = exc.__cause__
            if isinstance(cause, urllib.error.HTTPError) and 400 <= cause.code < 500:
                assert parser is not None
                parser.parse([])
            else:
                logger.info("robots.txt unreadable at %s (%s); treating as disallow", key, exc)
                parser = None
        self._robots[key] = parser
        return parser

    def check_allowed(self, url: str) -> float:
        """Refuse a URL robots.txt disallows; return the delay to keep for its origin.

        Raises:
            BlockedSourceError: If robots.txt disallows the URL or cannot be read.

        """
        parser = self.robots_for(url)
        if parser is None:
            raise BlockedSourceError(
                f"robots.txt of {origin(url)} could not be read, so nothing there is fetched"
            )
        if not parser.can_fetch(self.config.user_agent, url):
            raise BlockedSourceError(f"robots.txt of {origin(url)} disallows {url}")
        requested = parser.crawl_delay(self.config.user_agent)
        return max(self.config.crawl_delay_seconds, float(requested or 0.0))

    def download(self, url: str, directory: Path, stem: str) -> tuple[Path, Response, ContentKind]:
        """Stream a media URL to ``directory`` and return the file, response and kind.

        Raises:
            NotMediaError: If the response is not a supported image or video.
            BlockedSourceError: If policy forbids the request.
            SourceError: If the download fails or exceeds the size limit.

        """
        delay = self.check_allowed(url)
        with self._open(url, delay) as response:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type")
            detected = media_type(content_type, final_url)
            if detected is None:
                raise NotMediaError(f"{url} is not a supported image or video ({content_type})")
            kind, suffix = detected
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{stem}{suffix}"
            limit = self.max_download_bytes
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise SourceError(f"{url} is {int(declared)} bytes, over the {limit}-byte limit")
            total = 0
            with destination.open("wb") as handle:
                while chunk := response.read(CHUNK_BYTES):
                    total += len(chunk)
                    if total > limit:
                        handle.close()
                        destination.unlink(missing_ok=True)
                        raise SourceError(f"{url} exceeded the {limit}-byte limit")
                    handle.write(chunk)
        return destination, Response(final_url, content_type, b""), kind


def _largest_candidate(src: str | None, srcset: str | None) -> str | None:
    """Return the highest-resolution URL an image offers: srcset's largest, else src.

    Width descriptors (``500w``) outrank density ones (``2x``), since a width
    says how large the file is; ``src`` counts as ``1x``.
    """
    widths: list[tuple[float, str]] = []
    densities: list[tuple[float, str]] = [(1.0, src)] if src else []
    for candidate in (srcset or "").split(","):
        parts = candidate.strip().split()
        if not parts:
            continue
        descriptor = parts[1].lower() if len(parts) > 1 else "1x"
        try:
            value = float(descriptor[:-1])
        except ValueError:
            continue
        if descriptor.endswith("w"):
            widths.append((value, parts[0]))
        elif descriptor.endswith("x"):
            densities.append((value, parts[0]))
    best = max(widths or densities, default=None, key=lambda pair: pair[0])
    return best[1] if best else None


def _too_small(values: dict[str, str | None]) -> bool:
    """Return whether an element declares a side too small to hold a face."""
    declared = [
        int(value) for value in (values.get("width"), values.get("height"))
        if value and value.strip().isdigit()
    ]
    return bool(declared) and min(declared) < MIN_DECLARED_SIDE


class _LinkParser(HTMLParser):
    """Collects followable page links and embedded media from one HTML document.

    Each image element contributes one URL, its largest rendition, so a photo
    offered at several sizes is fetched once. Media a page only links to comes
    after media it shows, since such links often lead to a description page.
    """

    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.pages: list[str] = []
        self.media: list[tuple[str, ContentKind]] = []
        self.linked: list[tuple[str, ContentKind]] = []
        self._picture: list[str] | None = None
        self._picture_too_small = False

    def _absolute(self, value: str | None) -> str | None:
        if not value or value.strip().lower().startswith(("javascript:", "data:", "mailto:")):
            return None
        url, _ = urldefrag(urljoin(self.base, value.strip()))
        return url if urlsplit(url).scheme.lower() in ALLOWED_SCHEMES else None

    def _add_media(self, value: str | None, default: ContentKind | None) -> None:
        url = self._absolute(value)
        if url is None or PurePosixPath(urlsplit(url).path).suffix.lower() in NON_MEDIA_SUFFIXES:
            return
        kind = kind_from_suffix(url) or default
        if kind is not None:
            self.media.append((url, kind))

    def _close_picture(self) -> None:
        if self._picture is not None and not self._picture_too_small:
            self._add_media(_largest_candidate(None, ", ".join(self._picture)), ContentKind.IMAGE)
        self._picture = None
        self._picture_too_small = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value for name, value in attrs}
        if tag == "base":
            base = self._absolute(values.get("href"))
            if base:
                self.base = base
        elif tag == "a":
            url = self._absolute(values.get("href"))
            if url is None:
                return
            kind = kind_from_suffix(url)
            if kind is None:
                self.pages.append(url)
            else:
                self.linked.append((url, kind))
        elif tag == "picture":
            self._close_picture()
            self._picture = []
        elif tag == "img":
            src = values.get("src") or values.get("data-src")
            srcset = values.get("srcset") or values.get("data-srcset")
            if self._picture is not None:
                self._picture_too_small |= _too_small(values)
                self._picture.extend(part for part in (srcset, f"{src} 1x" if src else "") if part)
            elif not _too_small(values):
                self._add_media(_largest_candidate(src, srcset), ContentKind.IMAGE)
        elif tag == "video":
            self._add_media(values.get("src"), ContentKind.VIDEO)
            self._add_media(values.get("poster"), ContentKind.IMAGE)
        elif tag == "source":
            declared = (values.get("type") or "").lower()
            if self._picture is not None:
                if values.get("srcset"):
                    self._picture.append(values["srcset"] or "")
                return
            default = (
                ContentKind.VIDEO if declared.startswith("video/")
                else ContentKind.IMAGE if declared.startswith("image/")
                else None
            )
            self._add_media(values.get("src"), default)
        elif tag == "meta":
            key = (values.get("property") or values.get("name") or "").lower()
            if key in MEDIA_META:
                is_video = "video" in key or "stream" in key
                self._add_media(
                    values.get("content"), ContentKind.VIDEO if is_video else ContentKind.IMAGE
                )
        elif tag == "link" and (values.get("rel") or "").lower() == "image_src":
            self._add_media(values.get("href"), ContentKind.IMAGE)

    def handle_endtag(self, tag: str) -> None:
        if tag == "picture":
            self._close_picture()

    def close(self) -> None:
        super().close()
        self._close_picture()


def extract_links(html: str, base: str) -> tuple[list[str], list[tuple[str, ContentKind]]]:
    """Return the page links and the media URLs found in an HTML document.

    Media the page shows comes first, then media it only links to.
    """
    parser = _LinkParser(base)
    parser.feed(html)
    parser.close()
    shown = {url for url, _ in parser.media}
    linked = [(url, kind) for url, kind in parser.linked if url not in shown]
    return parser.pages, parser.media + linked


class WebSource(ContentSource):
    """Media reachable from a URL: the URL itself, the page it names, and pages it links to.

    ``max_depth`` 0 reads only the given page, or the given media file; each
    level beyond follows the page's links once more. Pages are followed only on
    the starting host unless ``same_host`` is off; media embedded in a visited
    page is taken wherever it is hosted.
    """

    name = "web"

    def __init__(
        self,
        config: SourcesConfig,
        max_depth: int | None = None,
        max_pages: int | None = None,
        max_media: int | None = None,
        same_host: bool | None = None,
        fetcher: HttpFetcher | None = None,
    ) -> None:
        """Store the crawl bounds; unset ones fall back to the configured defaults."""
        self.config = config
        self.max_depth = config.crawl_max_depth if max_depth is None else max(0, max_depth)
        self.max_pages = max_pages or config.crawl_max_pages
        self.max_media = max_media or config.crawl_max_media
        self.same_host = config.crawl_same_host if same_host is None else same_host
        self.fetcher = fetcher or HttpFetcher(config)
        self.skipped: list[tuple[str, str]] = []

    def _skip(self, url: str, reason: str) -> None:
        logger.info("skipped %s: %s", url, reason)
        self.skipped.append((url, reason))

    def discover(self, query: str) -> Iterator[ContentItem]:
        """Yield media items reachable from the URL ``query`` within the crawl bounds.

        Raises:
            BlockedSourceError: If the starting URL itself is refused by policy.
            SourceError: If the starting URL cannot be fetched.

        """
        seed = normalise(query)
        direct = kind_from_suffix(seed)
        if direct is not None:
            self.fetcher.check_allowed(seed)
            yield ContentItem(uri=seed, kind=direct, source=self.name)
            return

        start_host = urlsplit(seed).netloc.lower()
        queue: deque[tuple[str, int]] = deque([(seed, 0)])
        seen_pages: set[str] = {seed}
        seen_media: set[str] = set()
        pages_read = 0
        yielded = 0

        while queue and pages_read < self.max_pages and yielded < self.max_media:
            page, depth = queue.popleft()
            try:
                delay = self.fetcher.check_allowed(page)
                response = self.fetcher.get(page, MAX_PAGE_BYTES, delay=delay)
            except SourceError as exc:
                if page == seed:
                    raise
                self._skip(page, str(exc))
                continue
            pages_read += 1

            mime = (response.content_type or "").split(";")[0].strip().lower()
            detected = media_type(response.content_type, response.final_url)
            if detected is not None:
                if response.final_url not in seen_media:
                    seen_media.add(response.final_url)
                    yielded += 1
                    yield ContentItem(uri=response.final_url, kind=detected[0], source=self.name)
                continue
            if mime and mime not in HTML_TYPES:
                self._skip(page, f"not HTML or media ({mime})")
                continue

            html = response.body.decode("utf-8", errors="replace")
            links, media = extract_links(html, response.final_url)
            for url, kind in media:
                if url in seen_media or yielded >= self.max_media:
                    continue
                seen_media.add(url)
                yielded += 1
                yield ContentItem(uri=url, kind=kind, source=self.name, found_on=response.final_url)
            if depth >= self.max_depth:
                continue
            for url in links:
                if url in seen_pages:
                    continue
                if self.same_host and urlsplit(url).netloc.lower() != start_host:
                    continue
                seen_pages.add(url)
                queue.append((url, depth + 1))

    def fetch(self, item: ContentItem, directory: Path) -> FetchedContent:
        """Download one media item, re-checking robots.txt for its own host.

        Raises:
            BlockedSourceError: If policy forbids the download.
            SourceError: If the download fails or is not the media it claimed.

        """
        stem = hashlib.sha256(item.uri.encode("utf-8")).hexdigest()[:16]
        path, response, kind = self.fetcher.download(item.uri, directory, stem)
        if kind != item.kind:
            item = ContentItem(
                uri=item.uri, kind=kind, source=item.source, found_on=item.found_on,
                discovered_at=item.discovered_at,
            )
        data = path.read_bytes()
        return FetchedContent(
            item=item,
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            final_uri=response.final_url,
            content_type=response.content_type,
        )
