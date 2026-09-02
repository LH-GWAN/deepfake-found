"""Face matching: scoring a probe embedding against enrolled identities.

Plain language: decide how much a face in a suspect image looks like the person
who enrolled, and whether that is close enough to be worth investigating.

Formally, given a probe embedding ``p`` and a set of reference embeddings
``{r_1..r_n}`` for one identity, produce a single similarity in ``[-1, 1]`` and a
binary candidate decision against a configured threshold.

Both are on unit vectors, so cosine similarity is a dot product and Euclidean
distance is a monotone function of it: ``d^2 = 2 - 2 cos``. Cosine drives every
decision; the Euclidean value is logged only so that results can be compared
with literature that reports distances.

Four aggregation strategies are implemented so they can be compared on real data
rather than assumed:

``max``
    The best-matching reference wins. The MVP default, because a single
    reference sharing the probe's pose and lighting often carries the whole
    signal while the others drag an average down.
``mean``
    Every reference contributes. Stable, but one bad enrollment photo
    permanently depresses every score.
``topk_mean``
    Mean of the ``k`` best references. A compromise that resists both a single
    lucky match and a single bad enrollment photo.
``centroid``
    One comparison against the averaged template. Cheapest, and the most
    lossy: averaging discards the pose and lighting variation that made
    multiple enrollment photos worth collecting.

Three guards separate a usable match from a borderline one, because the cost of
a false identity claim is far higher than the cost of a miss:

two thresholds
    ``candidate`` decides what is worth spending an expensive detector on;
    ``high_confidence`` decides what may be reported as a match. Only the second
    populates ``matched_user_id``.
margin
    The gap between the best identity and the runner-up. A probe that scores
    0.62 against one identity and 0.61 against another has not identified
    anybody, however high the absolute number looks.
probe quality
    A small or blurred probe face produces an unreliable embedding. Rather than
    discard it, the decision threshold is raised in proportion, so weak evidence
    has to be stronger to count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from deepshield.config import FaceMatcherConfig, FaceSimilarityThresholds
from deepshield.exceptions import IdentityNotFoundError, ModelNotAvailableError
from deepshield.types import IdentityProfile, SimilarityResult


def l2_normalize(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return a vector scaled to unit L2 norm."""
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, epsilon)


def cosine_similarity(probe: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Return the cosine similarity of one probe against each reference row."""
    probe_unit = l2_normalize(np.asarray(probe, dtype=np.float64).ravel())
    matrix = np.atleast_2d(np.asarray(references, dtype=np.float64))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    similarities: np.ndarray = normalized @ probe_unit
    return similarities


def euclidean_distance(probe: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance of one probe to each reference row."""
    probe_vector = np.asarray(probe, dtype=np.float64).ravel()
    matrix = np.atleast_2d(np.asarray(references, dtype=np.float64))
    distances: np.ndarray = np.linalg.norm(matrix - probe_vector, axis=1)
    return distances


class FaceMatcher(ABC):
    """Contract for scoring a probe embedding against an identity profile."""

    @abstractmethod
    def match(
        self,
        probe: np.ndarray,
        profile: IdentityProfile,
        probe_quality: float | None = None,
    ) -> SimilarityResult:
        """Score one probe embedding against one enrolled identity."""

    @abstractmethod
    def match_many(
        self,
        probe: np.ndarray,
        profiles: list[IdentityProfile],
        probe_quality: float | None = None,
    ) -> list[SimilarityResult]:
        """Score one probe embedding against several enrolled identities."""


class NumpyFaceMatcher(FaceMatcher):
    """Cosine matcher over plain NumPy arrays.

    FAISS and vector databases are deliberately postponed. A personal identity
    store holds tens of embeddings, where an exact NumPy scan costs microseconds
    and an approximate index would only add a recall cliff to debug. Correctness
    and measurement come before optimisation.
    """

    def __init__(
        self,
        config: FaceMatcherConfig | None = None,
        thresholds: FaceSimilarityThresholds | None = None,
    ) -> None:
        """Store matcher configuration and decision thresholds."""
        self.config = config or FaceMatcherConfig()
        self.thresholds = thresholds or FaceSimilarityThresholds()

    def aggregate(self, similarities: np.ndarray) -> float:
        """Reduce per-reference similarities to one score using the configured rule."""
        if similarities.size == 0:
            raise IdentityNotFoundError("identity profile has no reference embeddings")
        strategy = self.config.aggregation
        if strategy == "max":
            return float(np.max(similarities))
        if strategy == "mean":
            return float(np.mean(similarities))
        if strategy == "topk_mean":
            k = min(self.config.top_k, similarities.size)
            return float(np.mean(np.sort(similarities)[-k:]))
        raise ModelNotAvailableError(f"unsupported aggregation strategy '{strategy}'")

    def effective_thresholds(self, probe_quality: float | None) -> tuple[float, float]:
        """Return the candidate and high-confidence thresholds for a probe.

        A low-quality probe raises both thresholds by up to
        ``low_quality_penalty``, so an unreliable measurement has to be stronger
        before it is allowed to make a claim.
        """
        candidate = self.thresholds.candidate_threshold
        high = self.thresholds.high_confidence_threshold
        penalty = self.thresholds.low_quality_penalty
        if probe_quality is None or penalty <= 0.0:
            return candidate, high
        adjustment = penalty * max(0.0, 1.0 - float(probe_quality))
        return min(1.0, candidate + adjustment), min(1.0, high + adjustment)

    def match(
        self,
        probe: np.ndarray,
        profile: IdentityProfile,
        probe_quality: float | None = None,
    ) -> SimilarityResult:
        """Score one probe embedding against one enrolled identity.

        Args:
            probe: The probe embedding.
            profile: The enrolled identity to score against.
            probe_quality: Optional quality in ``[0, 1]``; lower values raise the
                decision thresholds for this comparison.

        Raises:
            IdentityNotFoundError: If the profile holds no references.
            ModelNotAvailableError: If the probe and profile came from different
                embedding models, whose vectors are not comparable.

        """
        probe_vector = np.asarray(probe, dtype=np.float64).ravel()
        if probe_vector.size != profile.embedding_dimension:
            raise ModelNotAvailableError(
                f"probe has {probe_vector.size} dimensions but identity "
                f"'{profile.user_id}' was enrolled with {profile.embedding_dimension}; "
                "embeddings from different models are not comparable"
            )

        references = np.atleast_2d(profile.reference_embeddings)
        if self.config.aggregation == "centroid":
            similarities = cosine_similarity(probe_vector, profile.centroid_embedding)
            score = float(similarities[0])
        else:
            similarities = cosine_similarity(probe_vector, references)
            score = self.aggregate(similarities)

        distances = (
            euclidean_distance(probe_vector, references) if self.config.log_euclidean else None
        )
        candidate_threshold, high_threshold = self.effective_thresholds(probe_quality)
        is_candidate = score >= candidate_threshold
        is_high = score >= high_threshold

        return SimilarityResult(
            matched_user_id=profile.user_id,
            similarity=score,
            aggregation=self.config.aggregation,
            metric=self.config.metric,
            per_reference_similarity=[float(v) for v in np.atleast_1d(similarities)],
            euclidean_distance=None if distances is None else float(np.min(distances)),
            is_candidate=is_candidate,
            is_high_confidence=is_high,
            probe_quality=probe_quality,
            decision="high_confidence" if is_high else "candidate" if is_candidate else "no_match",
        )

    def match_many(
        self,
        probe: np.ndarray,
        profiles: list[IdentityProfile],
        probe_quality: float | None = None,
    ) -> list[SimilarityResult]:
        """Score a probe against several identities, best match first.

        The best result carries the runner-up score and the margin between them.
        When the margin falls below ``min_margin`` the match is demoted: the
        probe resembles two enrolled identities almost equally, which identifies
        neither of them.
        """
        results = [self.match(probe, profile, probe_quality) for profile in profiles]
        results.sort(key=lambda result: result.similarity, reverse=True)
        if not results:
            return results

        best = results[0]
        runner_up = results[1].similarity if len(results) > 1 else None
        margin = None if runner_up is None else best.similarity - runner_up

        demoted = margin is not None and margin < self.thresholds.min_margin
        is_high = best.is_high_confidence and not demoted
        is_candidate = best.is_candidate
        decision = (
            "ambiguous"
            if demoted and best.is_candidate
            else "high_confidence"
            if is_high
            else "candidate"
            if is_candidate
            else "no_match"
        )

        results[0] = replace(
            best,
            runner_up_similarity=runner_up,
            margin=margin,
            is_high_confidence=is_high,
            decision=decision,
        )
        return results

    def best_match(
        self,
        probe: np.ndarray,
        profiles: list[IdentityProfile],
        probe_quality: float | None = None,
    ) -> SimilarityResult | None:
        """Return the single best match, or ``None`` when no identity is enrolled."""
        if not profiles:
            return None
        return self.match_many(probe, profiles, probe_quality)[0]


def build_matcher(
    config: FaceMatcherConfig, thresholds: FaceSimilarityThresholds
) -> FaceMatcher:
    """Instantiate the configured matcher."""
    return NumpyFaceMatcher(config, thresholds)
