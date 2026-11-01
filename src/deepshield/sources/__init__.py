"""Content sources: folders, URLs and a bounded crawler feeding the analysis pipelines."""

from deepshield.sources.base import ContentItem, ContentKind, ContentSource, FetchedContent
from deepshield.sources.local import FolderSource
from deepshield.sources.web import HttpFetcher, WebSource

__all__ = [
    "ContentItem",
    "ContentKind",
    "ContentSource",
    "FetchedContent",
    "FolderSource",
    "HttpFetcher",
    "WebSource",
]
