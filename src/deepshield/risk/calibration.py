"""Threshold calibration and evaluation metrics.

Every decision boundary in ``configs/thresholds.yaml`` ships uncalibrated,
because a threshold copied from a paper describes that paper's dataset, not this
user's photographs. This module fits boundaries on labelled score pairs and
reports the operating characteristics that justify the choice, so a number in
the config file can always be traced back to the evidence that produced it.

Accuracy alone is never reported. A face matcher run against a realistic
population sees far more impostor pairs than genuine ones, and a detector that
answers "no match" every time would score well on accuracy while being useless.
ROC-AUC, EER and the true and false positive rates at the chosen point describe
the trade-off that accuracy hides.

Implemented with NumPy rather than scikit-learn so that calibration runs in the
minimal install; results agree with scikit-learn to floating-point tolerance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deepshield.exceptions import ConfigurationError


@dataclass
class RocCurve:
    """A receiver operating characteristic curve and its summary statistics."""

    thresholds: np.ndarray
    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    auc: float
    eer: float
    eer_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary without the full curve arrays."""
        return {
            "auc": round(self.auc, 6),
            "eer": round(self.eer, 6),
            "eer_threshold": round(self.eer_threshold, 6),
            "points": int(self.thresholds.size),
        }


@dataclass
class CalibrationResult:
    """A fitted threshold plus everything needed to defend it."""

    threshold: float
    metric: str
    roc: RocCurve
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    genuine_pairs: int
    impostor_pairs: int
    independent_pairs: int | None
    criterion: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "threshold": round(self.threshold, 6),
            "metric": self.metric,
            "criterion": self.criterion,
            "roc": self.roc.to_dict(),
            "true_positive_rate": round(self.true_positive_rate, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "genuine_pairs": self.genuine_pairs,
            "impostor_pairs": self.impostor_pairs,
            "independent_pairs": self.independent_pairs,
            "notes": list(self.notes),
        }


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> RocCurve:
    """Compute the ROC curve, its AUC and the equal error rate.

    Args:
        scores: Higher means more likely positive.
        labels: 1 for genuine pairs, 0 for impostor pairs.

    Raises:
        ConfigurationError: If either class is missing, which makes a ROC
            undefined rather than merely imprecise.

    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels).ravel().astype(int)
    if values.size != truth.size:
        raise ConfigurationError("scores and labels must have the same length")
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        raise ConfigurationError(
            "ROC is undefined: calibration needs both genuine and impostor pairs"
        )

    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    sorted_truth = truth[order]

    true_positives = np.cumsum(sorted_truth)
    false_positives = np.cumsum(1 - sorted_truth)
    tpr = np.concatenate([[0.0], true_positives / positives])
    fpr = np.concatenate([[0.0], false_positives / negatives])
    thresholds = np.concatenate([[np.inf], sorted_scores])

    auc = float(np.trapezoid(tpr, fpr))
    differences = np.abs((1.0 - tpr) - fpr)
    index = int(np.argmin(differences))
    eer = float((1.0 - tpr[index] + fpr[index]) / 2.0)
    eer_threshold = float(thresholds[index] if np.isfinite(thresholds[index]) else values.max())

    return RocCurve(
        thresholds=thresholds,
        false_positive_rate=fpr,
        true_positive_rate=tpr,
        auc=auc,
        eer=eer,
        eer_threshold=eer_threshold,
    )


class RiskCalibrator(ABC):
    """Contract for fitting decision thresholds on labelled data."""

    @abstractmethod
    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        independent_pairs: int | None = None,
    ) -> CalibrationResult:
        """Return the chosen threshold and its operating characteristics."""


class ThresholdCalibrator(RiskCalibrator):
    """Fits one decision boundary under an explicit operating criterion.

    Args:
        criterion: ``eer`` balances the two error rates; ``max_fpr`` picks the
            most permissive threshold whose false positive rate stays under
            ``target_fpr``; ``youden`` maximises ``TPR - FPR``; ``max_margin``
            places the boundary midway between the two distributions when they
            are perfectly separable, which is the safest choice on small samples
            because EER would otherwise sit exactly on the hardest genuine pair
            and leave no headroom at all.
        target_fpr: The ceiling used by the ``max_fpr`` criterion.
        metric: Name of the score being calibrated, recorded in the result.

    """

    def __init__(
        self,
        criterion: str = "eer",
        target_fpr: float = 0.01,
        metric: str = "cosine",
    ) -> None:
        """Store the calibration policy."""
        if criterion not in {"eer", "max_fpr", "youden", "max_margin"}:
            raise ConfigurationError(f"unknown calibration criterion '{criterion}'")
        self.criterion = criterion
        self.target_fpr = target_fpr
        self.metric = metric

    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        independent_pairs: int | None = None,
    ) -> CalibrationResult:
        """Fit a threshold and report the operating point it produces.

        Args:
            scores: Similarity per pair.
            labels: 1 for genuine pairs, 0 for impostor pairs.
            independent_pairs: How many statistically independent comparisons the
                scores came from. Augmenting a handful of photographs into
                thousands of pairs does not create new evidence, and a threshold
                fitted on such a set must say so.

        """
        values = np.asarray(scores, dtype=np.float64).ravel()
        truth = np.asarray(labels).ravel().astype(int)
        curve = roc_curve(values, truth)

        notes: list[str] = []
        if self.criterion == "max_margin":
            genuine_scores = values[truth == 1]
            impostor_scores = values[truth == 0]
            lowest_genuine = float(genuine_scores.min())
            highest_impostor = float(impostor_scores.max())
            if lowest_genuine > highest_impostor:
                threshold = (lowest_genuine + highest_impostor) / 2.0
                notes.append(
                    f"classes are separable; placed midway between the highest impostor "
                    f"({highest_impostor:.4f}) and the lowest genuine ({lowest_genuine:.4f})"
                )
            else:
                threshold = curve.eer_threshold
                notes.append(
                    "classes overlap, so no margin exists; fell back to the EER threshold"
                )
        elif self.criterion == "eer":
            threshold = curve.eer_threshold
        elif self.criterion == "youden":
            index = int(np.argmax(curve.true_positive_rate - curve.false_positive_rate))
            threshold = float(
                curve.thresholds[index]
                if np.isfinite(curve.thresholds[index])
                else values.max()
            )
        else:
            admissible = np.where(curve.false_positive_rate <= self.target_fpr)[0]
            if admissible.size == 0:
                threshold = float(values.max())
                notes.append(
                    f"no threshold reaches a false positive rate of {self.target_fpr:.3f}; "
                    "fell back to the maximum observed score"
                )
            else:
                index = int(admissible[-1])
                threshold = float(
                    curve.thresholds[index]
                    if np.isfinite(curve.thresholds[index])
                    else values.max()
                )

        predicted = values >= threshold
        true_positives = int(np.sum(predicted & (truth == 1)))
        false_positives = int(np.sum(predicted & (truth == 0)))
        false_negatives = int(np.sum(~predicted & (truth == 1)))
        genuine = int(truth.sum())
        impostor = int(truth.size - genuine)

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)

        if genuine < 10 or impostor < 10:
            notes.append(
                f"calibrated on only {genuine} genuine and {impostor} impostor pairs; "
                "treat the threshold as provisional until more data is available"
            )
        if independent_pairs is not None and independent_pairs < 30:
            notes.append(
                f"only {independent_pairs} statistically independent pairs underlie these "
                f"{genuine + impostor} scores; the confidence interval on this threshold is "
                "wide and it must be refitted on a larger, more diverse set before use"
            )

        return CalibrationResult(
            threshold=float(threshold),
            metric=self.metric,
            roc=curve,
            true_positive_rate=recall,
            false_positive_rate=false_positives / max(impostor, 1),
            precision=precision,
            recall=recall,
            genuine_pairs=genuine,
            impostor_pairs=impostor,
            independent_pairs=independent_pairs,
            criterion=self.criterion,
            notes=notes,
        )


def precision_recall_at(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    """Return the confusion counts and rates produced by one threshold."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels).ravel().astype(int)
    predicted = values >= threshold

    true_positives = int(np.sum(predicted & (truth == 1)))
    false_positives = int(np.sum(predicted & (truth == 0)))
    false_negatives = int(np.sum(~predicted & (truth == 1)))
    true_negatives = int(np.sum(~predicted & (truth == 0)))

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "true_negatives": float(true_negatives),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": false_positives / max(false_positives + true_negatives, 1),
    }


def threshold_for_precision(
    scores: np.ndarray, labels: np.ndarray, target_precision: float
) -> tuple[float, dict[str, float]]:
    """Return the lowest threshold reaching a target precision, and its operating point.

    Precision is the quantity that matters for a user-facing alert: telling
    someone their face appears in synthetic content when it does not is a far
    more damaging error than missing one instance, which other evidence or a
    later scan can still catch. Recall is reported alongside so the cost of the
    choice is visible rather than hidden.

    Falls back to the maximum observed score when no threshold reaches the
    target, and says so through the returned operating point.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels).ravel().astype(int)
    candidates = np.unique(values)[::-1]

    best: tuple[float, dict[str, float]] | None = None
    for threshold in candidates:
        point = precision_recall_at(values, truth, float(threshold))
        if point["precision"] >= target_precision and point["true_positives"] > 0:
            best = (float(threshold), point)
    if best is not None:
        return best
    fallback = float(values.max())
    return fallback, precision_recall_at(values, truth, fallback)


def pair_scores(
    embeddings: dict[str, np.ndarray], identity_of: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Build all genuine and impostor pair scores from labelled embeddings.

    Args:
        embeddings: Sample name to L2-normalised embedding.
        identity_of: Sample name to the identity it belongs to.

    Returns:
        Cosine similarity per pair and the matching genuine/impostor labels.

    """
    names = sorted(embeddings)
    scores: list[float] = []
    labels: list[int] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            scores.append(float(np.dot(embeddings[left], embeddings[right])))
            labels.append(int(identity_of[left] == identity_of[right]))
    return np.asarray(scores), np.asarray(labels)
