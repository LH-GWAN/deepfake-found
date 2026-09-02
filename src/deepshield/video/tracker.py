"""Face tracking and representative-frame selection.

Tracking groups the same face across sampled frames so that one identity
comparison and one expensive deepfake inference can serve an entire track rather
than every frame it appears in. That is the cost lever research question RQ7
measures, and on a clip where one person is on screen throughout it turns a
per-frame cost into a per-person cost.

Association uses two rules together, because either alone is wrong at the
sampling rates this system uses.

Geometric: intersection-over-union between a new detection and the last box of
each open track. Adequate at one or two frames per second where faces move only
modestly between samples.

Appearance: correlation between small colour histograms of the two face crops.
Geometry alone fails badly across a cut. Two different people filmed in the same
framing produce nearly identical boxes, so a pure IoU tracker merges them into
one track and the second person disappears from the analysis entirely - a missed
detection, which is the worst failure this system can have. Comparing appearance
costs one histogram per detection and separates them.

Both rules must agree for a detection to extend a track. The appearance
threshold is a heuristic fitted on very little data, so it is set to err toward
splitting: an over-split track costs a redundant embedding and deepfake
inference, while an under-split track silently drops a person from the report.
Fast motion and crossing faces still defeat this.

Representative-frame selection then picks the single frame per track worth
embedding, under one of three criteria: highest detection confidence, most
frontal pose, or least blur.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deepshield.config import RepresentativeFrameConfig, VideoTrackingConfig
from deepshield.types import BoundingBox, DetectedFace

APPEARANCE_BINS = 8
APPEARANCE_GRID = 2


@dataclass
class FaceTrack:
    """A sequence of detections believed to show the same face over time."""

    track_id: int
    faces: list[DetectedFace] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    representative_index: int | None = None
    last_seen: int = 0
    last_appearance: np.ndarray | None = None

    @property
    def representative(self) -> DetectedFace | None:
        """Return the frame chosen to represent this track, if one was selected."""
        if self.representative_index is None:
            return None
        return self.faces[self.representative_index]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "track_id": self.track_id,
            "length": len(self.faces),
            "first_frame": self.faces[0].frame_index if self.faces else None,
            "last_frame": self.faces[-1].frame_index if self.faces else None,
            "representative_index": self.representative_index,
        }


class FaceTracker(ABC):
    """Contract for grouping per-frame detections into tracks."""

    @abstractmethod
    def track(
        self,
        detections_per_frame: list[list[DetectedFace]],
        frames: list[np.ndarray] | None = None,
    ) -> list[FaceTrack]:
        """Group detections from consecutive sampled frames into tracks."""

    @abstractmethod
    def select_representative(
        self, track: FaceTrack, frames: dict[int, np.ndarray] | None = None
    ) -> FaceTrack:
        """Choose the frame of a track that will be embedded and scored."""


def crop_of(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Return the pixels inside a bounding box, clamped to the frame."""
    height, width = image.shape[:2]
    x1 = int(max(0, min(width - 1, box.x1)))
    y1 = int(max(0, min(height - 1, box.y1)))
    x2 = int(max(x1 + 1, min(width, box.x2)))
    y2 = int(max(y1 + 1, min(height, box.y2)))
    return image[y1:y2, x1:x2]


def appearance_descriptor(crop: np.ndarray) -> np.ndarray:
    """Return a normalised colour histogram over a coarse spatial grid of a crop.

    Deliberately crude: it must be cheap enough to run on every detection, and it
    only has to separate different people filmed in similar framing, not identify
    anyone. Identity comparison is the embedder's job, on one frame per track.

    The histogram is computed per grid cell rather than over the whole crop.
    A global histogram is nearly blind to layout, so two portraits with similar
    overall tone - two black-and-white photographs, for instance - score as the
    same person. Splitting into cells keeps the descriptor cheap while making it
    sensitive to where the tones sit.
    """
    length = APPEARANCE_BINS * 3 * APPEARANCE_GRID * APPEARANCE_GRID
    if crop.size == 0:
        return np.zeros(length, dtype=np.float64)

    height, width = crop.shape[:2]
    parts = []
    for row in range(APPEARANCE_GRID):
        for col in range(APPEARANCE_GRID):
            cell = crop[
                row * height // APPEARANCE_GRID : (row + 1) * height // APPEARANCE_GRID,
                col * width // APPEARANCE_GRID : (col + 1) * width // APPEARANCE_GRID,
            ]
            for channel in range(3):
                counts, _ = np.histogram(
                    cell[:, :, channel], bins=APPEARANCE_BINS, range=(0, 256)
                )
                parts.append(counts.astype(np.float64))
    descriptor = np.concatenate(parts)
    total = descriptor.sum()
    return descriptor / total if total > 0 else descriptor


def appearance_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Return histogram intersection between two descriptors, in ``[0, 1]``.

    Returns a permissive 1.0 when either descriptor is missing, so that a caller
    without frame pixels degrades to pure IoU tracking rather than rejecting
    every association.
    """
    if left is None or right is None:
        return 1.0
    return float(np.minimum(left, right).sum())


def frontality(face: DetectedFace) -> float:
    """Estimate how frontal a face is from its five landmarks, in ``[0, 1]``.

    Compares the horizontal distance from each eye to the nose. On a frontal
    face those distances are equal; as the head yaws, one shrinks. Returns a
    neutral 0.5 when landmarks are unavailable, so a detector without landmarks
    degrades to arbitrary selection rather than silently ranking everything last.
    """
    if face.landmarks is None or len(face.landmarks) < 5:
        return 0.5
    points = np.asarray(face.landmarks, dtype=np.float64)
    left_eye, right_eye, nose = points[0], points[1], points[2]
    left = abs(float(nose[0] - left_eye[0]))
    right = abs(float(right_eye[0] - nose[0]))
    total = left + right
    if total <= 0:
        return 0.5
    return float(1.0 - abs(left - right) / total)


class IouFaceTracker(FaceTracker):
    """Greedy IoU tracker with a tolerance for short detection gaps."""

    name = "iou"

    def __init__(
        self,
        config: VideoTrackingConfig | None = None,
        representative: RepresentativeFrameConfig | None = None,
    ) -> None:
        """Store the association and selection policies."""
        self.config = config or VideoTrackingConfig()
        self.representative_config = representative or RepresentativeFrameConfig()

    def track(
        self,
        detections_per_frame: list[list[DetectedFace]],
        frames: list[np.ndarray] | None = None,
    ) -> list[FaceTrack]:
        """Associate detections across frames using geometry and appearance.

        Args:
            detections_per_frame: Detections for each sampled frame, in order.
            frames: The sampled frames. Without them the appearance check is
                skipped and association falls back to IoU alone.

        """
        tracks: list[FaceTrack] = []
        open_tracks: list[FaceTrack] = []
        next_id = 0

        for step, detections in enumerate(detections_per_frame):
            image = frames[step] if frames is not None and step < len(frames) else None
            descriptors = [
                appearance_descriptor(crop_of(image, face.bbox)) if image is not None else None
                for face in detections
            ]
            open_tracks = [
                track
                for track in open_tracks
                if step - track.last_seen <= self.config.max_gap_frames
            ]
            unmatched = list(range(len(detections)))

            for track in open_tracks:
                if not unmatched:
                    break
                last_box = track.faces[-1].bbox
                candidates = [
                    (
                        index,
                        last_box.iou(detections[index].bbox),
                        appearance_similarity(track.last_appearance, descriptors[index]),
                    )
                    for index in unmatched
                ]
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate[1] >= self.config.iou_threshold
                    and candidate[2] >= self.config.appearance_threshold
                ]
                if not eligible:
                    continue
                index, _, _ = max(eligible, key=lambda c: c[1] * c[2])
                unmatched.remove(index)
                track.faces.append(detections[index])
                track.frame_indices.append(step)
                track.last_seen = step
                track.last_appearance = descriptors[index]

            for index in unmatched:
                track = FaceTrack(
                    track_id=next_id,
                    faces=[detections[index]],
                    frame_indices=[step],
                    last_seen=step,
                    last_appearance=descriptors[index],
                )
                next_id += 1
                tracks.append(track)
                open_tracks.append(track)

        return [self.select_representative(track) for track in tracks]

    def select_representative(
        self, track: FaceTrack, frames: dict[int, np.ndarray] | None = None
    ) -> FaceTrack:
        """Pick the frame of a track worth the cost of embedding and scoring."""
        if not track.faces:
            return track

        criterion = self.representative_config.criterion
        if criterion == "frontal_pose":
            scores = [frontality(face) for face in track.faces]
        elif criterion == "sharpness":
            if frames is None:
                scores = [face.detection_confidence for face in track.faces]
            else:
                from deepshield.quality import laplacian_variance

                scores = []
                for face, index in zip(track.faces, track.frame_indices, strict=False):
                    image = frames.get(index)
                    if image is None:
                        scores.append(0.0)
                        continue
                    box = face.bbox
                    crop = image[
                        int(max(0, box.y1)) : int(box.y2), int(max(0, box.x1)) : int(box.x2)
                    ]
                    scores.append(laplacian_variance(crop) if crop.size else 0.0)
        else:
            scores = [face.detection_confidence for face in track.faces]

        track.representative_index = int(np.argmax(scores))
        return track


def build_tracker(
    config: VideoTrackingConfig, representative: RepresentativeFrameConfig
) -> FaceTracker:
    """Instantiate the configured tracker."""
    return IouFaceTracker(config, representative)
