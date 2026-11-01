"""Planned REST surface.

Declared ahead of implementation so the CLI and the API stay in step and so the
contract can be reviewed before any handler exists. Handlers land in Phase 14.
"""

from __future__ import annotations

ROUTES: tuple[tuple[str, str, str], ...] = (
    ("POST", "/identity/enroll", "Enroll a user identity from reference images"),
    ("POST", "/protect/image", "Apply the protection pipeline to one image"),
    ("POST", "/analyze/image", "Analyse one suspect image"),
    ("POST", "/analyze/video", "Analyse one suspect video"),
    ("POST", "/watermark/detect", "Attempt watermark extraction from one image"),
    ("GET", "/analysis/{analysis_id}", "Fetch a stored evidence record"),
    ("GET", "/health", "Liveness and component availability"),
)
