"""Download a small public-domain face set for manual and integration testing.

No face images are committed to this repository. This script fetches a handful
of public-domain portraits into ``data/test/faces/``, which is git-ignored, so
that the face pipeline can be exercised against real photographs without
shipping anyone's biometrics in version control.

All images are public domain. Two identities appear twice, which is what makes a
genuine identity test possible: same-person pairs must score high and
different-person pairs must score low.

Usage:
    python scripts/fetch_sample_faces.py [--output data/test/faces] [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "deepshield-research/0.1 (identity-protection research; contact repository owner)"
REQUEST_DELAY_SECONDS = 1.0

SAMPLES: dict[str, str] = {
    "einstein_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg",
    "einstein_2.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/3/3e/"
        "Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
    ),
    "curie_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Marie_Curie_c._1920s.jpg",
    "curie_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/7e/Marie_Curie_c1920.jpg",
    "tesla_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/79/Tesla_circa_1890.jpeg",
}

IDENTITY_OF: dict[str, str] = {
    "einstein_1.jpg": "einstein",
    "einstein_2.jpg": "einstein",
    "curie_1.jpg": "curie",
    "curie_2.jpg": "curie",
    "tesla_1.jpg": "tesla",
}


def fetch(url: str, destination: Path, force: bool = False) -> bool:
    """Download one file, returning whether a network request was made."""
    if destination.exists() and not force:
        return False
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())
    return True


def main(argv: list[str] | None = None) -> int:
    """Fetch every sample image into the output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/test/faces"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for filename, url in SAMPLES.items():
        destination = args.output / filename
        try:
            if fetch(url, destination, args.force):
                print(f"downloaded {destination}")
                time.sleep(REQUEST_DELAY_SECONDS)
            else:
                print(f"present    {destination}")
        except (urllib.error.URLError, OSError) as exc:
            failures += 1
            print(f"FAILED     {destination}: {exc}", file=sys.stderr)

    manifest = args.output / "identities.txt"
    manifest.write_text(
        "\n".join(f"{name}\t{identity}" for name, identity in IDENTITY_OF.items()) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
