"""Measure which verdict each real-world situation receives, end to end.

The risk engine replaced a weighted 0-100 score after that score was measured
ranking a user's own protected photograph, reposted unchanged, above a GAN face
swap of the user. This script is the measurement that has to come out the other
way now, and the one research question RQ5 asked for: does fusing identity,
registered-origin and synthetic-media evidence tell situations apart that a
single signal would confuse?

For every identity in the evaluation set, one photograph is protected and
registered, one is held out as an unregistered genuine photograph, and the rest
are enrolled. Then seven situations are built with the real face stack and the
real GAN swapper, and each is analysed through the full pipeline for that user:

repost              the protected file itself
recompressed        the protected file re-encoded as JPEG quality 70
genuine             the held-out genuine photograph of the user
gan_source          the user's face swapped onto another identity's photograph
gan_target          another identity's face swapped onto the user's protected photograph
gan_target_orphan   the same, with the protected file removed from disk first
unrelated           another identity's genuine photograph

The expected verdicts are own_copy, own_copy, identity_match, identity_match,
own_altered, own_unverified and unrelated. ``genuine`` and ``gan_source`` are
expected to be indistinguishable: without a calibrated synthetic-media detector
the engine is designed to say so rather than guess, and the table shows whether
it does.

Everything is written under a temporary directory; the repository's own data
directory is not touched.

Profiles are built directly from every detectable face rather than through
``DefaultIdentityEnroller``, whose quality filter rejects many 250-pixel LFW
photographs outright. That keeps all thirty identities in the measurement, and
it means the table describes the verdict rules, not the enrollment gate.

Usage:
    python scripts/evaluate_verdicts.py
    python scripts/evaluate_verdicts.py --identities 10
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import DeepShieldConfig, load_config
from deepshield.experiments import environment
from deepshield.media import load_image, save_image
from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline
from deepshield.types import IdentityProfile

EXPECTED = {
    "repost": "own_copy",
    "recompressed": "own_copy",
    "genuine": "identity_match",
    "gan_source": "identity_match",
    "gan_target": "own_altered",
    "gan_target_orphan": "own_unverified",
    "unrelated": "unrelated",
}


class GanSwapper:
    """``inswapper_128``: puts one photograph's identity onto another's face."""

    def __init__(self, models: Path, weights: Path) -> None:
        """Load the detector-recogniser pack and the swapper."""
        import insightface
        from insightface.model_zoo import get_model

        if not weights.is_file():
            raise SystemExit(f"missing {weights}; run scripts/fetch_inswapper.py first")
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=str(models),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.model = get_model(str(weights), providers=["CPUExecutionProvider"])

    def _largest(self, image: np.ndarray) -> Any:
        faces = self.app.get(np.ascontiguousarray(image[:, :, ::-1]))
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def __call__(self, identity: np.ndarray, picture: np.ndarray) -> np.ndarray | None:
        """Return ``picture`` with its face replaced by the identity in ``identity``."""
        source, target = self._largest(identity), self._largest(picture)
        if source is None or target is None:
            return None
        swapped = self.model.get(
            np.ascontiguousarray(picture[:, :, ::-1]), target, source, paste_back=True
        )
        return np.ascontiguousarray(swapped[:, :, ::-1])


def isolated(config: DeepShieldConfig, workspace: Path) -> DeepShieldConfig:
    """Point every store at ``workspace`` so the run leaves no trace in the repository."""
    return config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "data_dir": workspace / "data",
                    "results_dir": workspace / "data" / "results",
                    "model_dir": ROOT / config.runtime.model_dir,
                }
            ),
            "storage": config.storage.model_copy(
                update={"embedding_store_dir": workspace / "data" / "embeddings"}
            ),
        }
    )


def photographs(faces: Path) -> dict[str, list[Path]]:
    """Group evaluation photographs by identity."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(faces.glob("*.png")):
        grouped[path.stem.rsplit("_", 1)[0]].append(path)
    return {name: paths for name, paths in grouped.items() if len(paths) >= 3}


def enroll(pipeline: DefaultAnalysisPipeline, user_id: str, paths: list[Path]) -> bool:
    """Store a profile built from every detectable face in ``paths``."""
    vectors = []
    for path in paths:
        image = load_image(path)
        faces = pipeline.detector.detect(image)
        if faces:
            face = max(faces, key=lambda f: f.detection_confidence)
            aligned = pipeline.aligner.align(image, face).image
            vectors.append(pipeline.embedder.embed(aligned).vector)
    if not vectors:
        return False
    references = np.stack(vectors)
    centroid = references.mean(axis=0)
    pipeline.identities.save(
        IdentityProfile(
            user_id=user_id,
            reference_embeddings=references,
            centroid_embedding=centroid / np.linalg.norm(centroid),
            image_count=len(vectors),
            model=pipeline.embedder.model_info,
            embedding_dimension=pipeline.embedder.dimension,
        )
    )
    return True


def scenarios(
    user: str,
    own: list[Path],
    other: Path,
    protected: Path,
    swapper: GanSwapper,
    workspace: Path,
) -> dict[str, Path | None]:
    """Build every situation's image for one user; ``None`` when a swap fails."""
    from PIL import Image

    held_out, other_image = load_image(own[-2]), load_image(other)
    protected_image = load_image(protected)
    recompressed = workspace / f"{user}_recompressed.jpg"
    Image.fromarray(protected_image).save(recompressed, quality=70)

    def written(image: np.ndarray | None, name: str) -> Path | None:
        return None if image is None else save_image(image, workspace / f"{user}_{name}.png")

    return {
        "repost": protected,
        "recompressed": recompressed,
        "genuine": own[-2],
        "gan_source": written(swapper(held_out, other_image), "gan_source"),
        "gan_target": written(swapper(other_image, protected_image), "gan_target"),
        "unrelated": other,
    }


def main(argv: list[str] | None = None) -> int:
    """Run every situation for every identity and tabulate the verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument(
        "--inswapper", type=Path, default=Path("models/inswapper/inswapper_128.onnx")
    )
    parser.add_argument("--identities", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    grouped = photographs(args.faces)
    names = sorted(grouped)[: args.identities] if args.identities else sorted(grouped)
    if len(names) < 2:
        raise SystemExit(f"need at least two identities with three photographs in {args.faces}")

    swapper = GanSwapper(args.models, args.inswapper)
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    levels: dict[str, Counter[str]] = defaultdict(Counter)
    failures: Counter[str] = Counter()

    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        config = isolated(load_config(), workspace)
        pipeline = DefaultAnalysisPipeline(config)
        protection = DefaultProtectionPipeline(config)

        for index, user in enumerate(names):
            own = grouped[user]
            other = grouped[names[(index + 1) % len(names)]][0]
            if not enroll(pipeline, user, own[:-2]):
                failures["enrollment"] += 1
                continue
            report = protection.protect(own[-1], user, "evaluation")
            protected = Path(report["protected_path"])
            built = scenarios(user, own, other, protected, swapper, workspace)

            def judge(situation: str, path: Path | None, user: str = user) -> None:
                if path is None:
                    failures[situation] += 1
                    return
                risk = pipeline.analyze_image(path, user).risk
                assert risk is not None
                outcomes[situation][risk.verdict.value] += 1
                levels[situation][risk.risk_level.value] += 1

            for situation, path in built.items():
                judge(situation, path)
            stash = protected.with_suffix(".moved")
            protected.rename(stash)
            judge("gan_target_orphan", built["gan_target"])
            stash.rename(protected)
            print(f"{index + 1}/{len(names)} {user}", flush=True)

    table = {
        situation: {
            "expected": expected,
            "cases": sum(outcomes[situation].values()),
            "as_expected": outcomes[situation][expected],
            "verdicts": dict(outcomes[situation]),
            "levels": dict(levels[situation]),
        }
        for situation, expected in EXPECTED.items()
    }
    report_payload = {
        "question": "which verdict does each real-world situation receive?",
        "identities": len(names),
        "situations": table,
        "not_built": dict(failures),
        "environment": environment(load_config()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "verdicts.json"
    destination.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'situation':20s} {'expected':16s} {'as expected':>12s}   verdicts")
    for situation, row in table.items():
        print(
            f"{situation:20s} {row['expected']:16s} "
            f"{row['as_expected']:>5d}/{row['cases']:<6d}   {row['verdicts']}"
        )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
