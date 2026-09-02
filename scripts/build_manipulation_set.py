"""Generate a labelled real-versus-manipulated face set for detector calibration.

The deepfake signal is the one piece of evidence this system computes but does
not score, because no threshold has ever been fitted for it. Fitting one needs
labelled data, and no labelled deepfake corpus ships with this project.

This script builds one for a single, clearly named manipulation family:
graphics-based face swapping. The source face is warped onto the target
triangle by triangle over a Delaunay mesh of 106 facial landmarks, then
composited with Poisson blending and re-encoded as JPEG. That is the same
family as the graphics-based FaceSwap in FaceForensics++, and it is the
manipulation an attacker can perform without training anything.

The warp is piecewise rather than a single affine on purpose. One global
transform cannot match two different face shapes, and it leaves misalignment a
person spots instantly. A detector calibrated on obviously broken forgeries
would look excellent and mean nothing, so the generator has to be good enough
that the remaining evidence is the blending itself.

Donors are chosen by pose similarity for the same reason. Warping a
three-quarter view onto a frontal face smears it into something no attacker
would publish, and a detector that learns to spot that has learned nothing
useful. Each target is paired with the face from another identity whose
normalised landmark geometry is closest to its own.

What this set is not: it contains no GAN and no diffusion output. A detector
calibrated here is calibrated for blending-based swaps only. Every downstream
report says so, because a detector's training family is the single best
predictor of where it will fail.

Usage:
    python scripts/build_manipulation_set.py --faces data/test/eval_faces
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.exceptions import ModelNotAvailableError
from deepshield.media import load_image, save_image

FEATHER_PIXELS = 9
MIN_MASK_AREA = 400


def build_analyzer(model_dir: Path) -> Any:
    """Return an InsightFace app providing detection and 106-point landmarks."""
    try:
        import insightface
    except ImportError as exc:
        raise ModelNotAvailableError(
            "insightface is required to build the manipulation set; install the 'face' extra"
        ) from exc
    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(model_dir / "insightface"),
        allowed_modules=["detection", "landmark_2d_106"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def largest_face(app: Any, image: np.ndarray) -> Any:
    """Return the biggest detected face with landmarks, or ``None``."""
    faces = app.get(np.ascontiguousarray(image[:, :, ::-1]))
    usable = [f for f in faces if getattr(f, "landmark_2d_106", None) is not None]
    if not usable:
        return None
    return max(usable, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def hull_mask(cv2: Any, shape: tuple[int, int], landmarks: np.ndarray) -> np.ndarray:
    """Return a mask covering the convex hull of the face landmarks."""
    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = np.ones((FEATHER_PIXELS, FEATHER_PIXELS), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def delaunay_triangles(
    cv2: Any, shape: tuple[int, int], points: np.ndarray
) -> list[tuple[int, int, int]]:
    """Return landmark index triples forming a Delaunay mesh over the face.

    Vertices coming back from ``Subdiv2D`` are matched to landmarks by nearest
    neighbour rather than by an exact coordinate lookup. On a face only a
    hundred pixels across, several of the 106 landmarks round to the same
    integer pixel, and an exact lookup silently returns the wrong index for
    them - which builds a mesh that shreds the face it is supposed to warp.
    """
    height, width = shape
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    clipped = np.stack(
        [np.clip(points[:, 0], 0, width - 1), np.clip(points[:, 1], 0, height - 1)], axis=1
    ).astype(np.float32)
    for point in clipped:
        subdiv.insert((float(point[0]), float(point[1])))

    triangles: list[tuple[int, int, int]] = []
    for triangle in subdiv.getTriangleList():
        corners = np.asarray(triangle, dtype=np.float32).reshape(3, 2)
        distances = np.linalg.norm(clipped[None, :, :] - corners[:, None, :], axis=2)
        indices = distances.argmin(axis=1)
        if float(distances.min(axis=1).max()) > 1.5:
            continue
        if len(set(indices.tolist())) != 3:
            continue
        triangles.append((int(indices[0]), int(indices[1]), int(indices[2])))
    return triangles


def warp_triangle(
    cv2: Any,
    source: np.ndarray,
    destination: np.ndarray,
    source_triangle: np.ndarray,
    target_triangle: np.ndarray,
) -> None:
    """Warp one source triangle onto the destination image in place."""
    source_rect = cv2.boundingRect(source_triangle.astype(np.float32))
    target_rect = cv2.boundingRect(target_triangle.astype(np.float32))
    if source_rect[2] <= 0 or source_rect[3] <= 0:
        return
    if target_rect[2] <= 0 or target_rect[3] <= 0:
        return

    source_local = source_triangle - np.array(source_rect[:2], dtype=np.float32)
    target_local = target_triangle - np.array(target_rect[:2], dtype=np.float32)

    patch = source[
        source_rect[1] : source_rect[1] + source_rect[3],
        source_rect[0] : source_rect[0] + source_rect[2],
    ]
    if patch.size == 0:
        return

    matrix = cv2.getAffineTransform(
        source_local.astype(np.float32), target_local.astype(np.float32)
    )
    warped = cv2.warpAffine(
        patch,
        matrix,
        (target_rect[2], target_rect[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((target_rect[3], target_rect[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, target_local.astype(np.int32), 1.0, cv2.LINE_AA)
    mask = mask[:, :, None]

    region = destination[
        target_rect[1] : target_rect[1] + target_rect[3],
        target_rect[0] : target_rect[0] + target_rect[2],
    ]
    if region.shape[:2] != warped.shape[:2]:
        return
    region[:] = region * (1.0 - mask) + warped.astype(np.float32) * mask


def colour_match(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift the source's colour statistics onto the target's inside the mask.

    Without this the swapped face keeps the source photograph's illumination and
    the result is obvious to a human. Matching it makes the manipulation
    realistic enough that a detector has to find the blending boundary rather
    than a colour step.
    """
    selection = mask > 32
    if selection.sum() < MIN_MASK_AREA:
        return source
    result = source.astype(np.float32)
    for channel in range(3):
        source_values = result[:, :, channel][selection]
        target_values = target[:, :, channel].astype(np.float32)[selection]
        source_std = float(source_values.std()) or 1.0
        scaled = (source_values - source_values.mean()) * (
            float(target_values.std()) / source_std
        ) + target_values.mean()
        plane = result[:, :, channel]
        plane[selection] = scaled
    return np.clip(result, 0, 255).astype(np.uint8)


def swap_face(
    cv2: Any, source: np.ndarray, target: np.ndarray, app: Any
) -> np.ndarray | None:
    """Warp the source face onto the target piecewise and blend it in."""
    source_face = largest_face(app, source)
    target_face = largest_face(app, target)
    if source_face is None or target_face is None:
        return None

    source_points = np.asarray(source_face.landmark_2d_106, dtype=np.float32)
    target_points = np.asarray(target_face.landmark_2d_106, dtype=np.float32)

    height, width = target.shape[:2]
    mask = hull_mask(cv2, (height, width), target_points)
    if (mask > 0).sum() < MIN_MASK_AREA:
        return None

    warped = target.astype(np.float32).copy()
    triangles = delaunay_triangles(cv2, (height, width), target_points)
    if len(triangles) < 20:
        return None
    for a, b, c in triangles:
        warp_triangle(
            cv2,
            source.astype(np.float32),
            warped,
            source_points[[a, b, c]],
            target_points[[a, b, c]],
        )

    warped_uint8 = np.clip(warped, 0, 255).astype(np.uint8)
    warped_uint8 = colour_match(warped_uint8, target, mask)

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    centre = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    try:
        blended = cv2.seamlessClone(
            warped_uint8[:, :, ::-1], target[:, :, ::-1], mask, centre, cv2.NORMAL_CLONE
        )
    except cv2.error:
        return None
    return np.ascontiguousarray(blended[:, :, ::-1])


def pose_descriptor(landmarks: np.ndarray) -> np.ndarray:
    """Return a translation- and scale-invariant description of face geometry.

    Centring on the landmark centroid and dividing by their spread leaves a
    shape vector that varies mainly with head pose and expression, which is what
    donor selection needs to match.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    centred = points - points.mean(axis=0)
    scale = float(np.sqrt((centred**2).sum(axis=1)).mean()) or 1.0
    return (centred / scale).ravel()


def read_manifest(faces_dir: Path) -> dict[str, str]:
    """Read the sample-to-identity mapping written by the evaluation set builder."""
    manifest = faces_dir / "identities.txt"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}; run scripts/build_evaluation_set.py first")
    mapping = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, identity = line.split("\t")
            mapping[name] = identity
    return mapping


def main(argv: list[str] | None = None) -> int:
    """Build matched real and manipulated sets with an identity-disjoint layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--output", type=Path, default=Path("data/test/manipulated"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        import cv2
    except ImportError:
        raise SystemExit("OpenCV is required; install the 'face' extra") from None

    manifest = read_manifest(args.faces)
    app = build_analyzer(args.models)

    real_dir = args.output / "real"
    fake_dir = args.output / "fake"
    for directory in (real_dir, fake_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for existing in directory.glob("*"):
            existing.unlink()

    by_identity: dict[str, list[str]] = {}
    for name, identity in manifest.items():
        by_identity.setdefault(identity, []).append(name)
    identities = sorted(by_identity)

    print("measuring pose for donor selection", flush=True)
    poses: dict[str, np.ndarray] = {}
    for name in sorted(manifest):
        face = largest_face(app, load_image(args.faces / name))
        if face is not None:
            poses[name] = pose_descriptor(face.landmark_2d_106)

    records: list[dict[str, str]] = []
    failures = 0

    for identity in identities:
        targets = sorted(by_identity[identity])
        candidates = [name for name in poses if manifest[name] != identity]

        for position, target_name in enumerate(targets):
            target = load_image(args.faces / target_name)
            if target_name not in poses or not candidates:
                failures += 1
                continue
            reference = poses[target_name]
            donor_name = min(
                candidates, key=lambda name: float(np.linalg.norm(poses[name] - reference))
            )
            source = load_image(args.faces / donor_name)

            real_path = real_dir / f"{Path(target_name).stem}.jpg"
            save_image(target, real_path, quality=args.quality)
            records.append(
                {
                    "path": str(real_path),
                    "label": "real",
                    "identity": identity,
                    "source": target_name,
                }
            )

            swapped = swap_face(cv2, source, target, app)
            if swapped is None:
                failures += 1
                continue
            fake_path = fake_dir / f"{Path(target_name).stem}_swap.jpg"
            save_image(swapped, fake_path, quality=args.quality)
            records.append(
                {
                    "path": str(fake_path),
                    "label": "fake",
                    "identity": identity,
                    "source": target_name,
                    "donor": donor_name,
                    "method": "landmark_affine_hull_blend",
                }
            )
            if position == 0:
                print(
                    f"  {identity:22s} donor {manifest[donor_name]:22s}"
                    f" pose distance {float(np.linalg.norm(poses[donor_name] - reference)):.3f}",
                    flush=True,
                )

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "method": "graphics-based face swap: 106-landmark affine warp, colour "
                "matching, feathered convex-hull blend, JPEG re-encode",
                "covers": ["blending-based face swap"],
                "does_not_cover": ["GAN synthesis", "diffusion synthesis", "reenactment"],
                "jpeg_quality": args.quality,
                "seed": args.seed,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    real = sum(1 for r in records if r["label"] == "real")
    fake = sum(1 for r in records if r["label"] == "fake")
    print(f"\n{real} real and {fake} manipulated images, {failures} swaps failed")
    print(f"wrote {manifest_path}")
    print(
        "\nThis set covers blending-based face swapping only. A detector calibrated "
        "on it says nothing about GAN or diffusion output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
