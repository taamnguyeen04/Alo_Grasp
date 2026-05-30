from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from shapely.geometry import Polygon

from .grasp import Grasp

def grasp_iou(a: Grasp, b: Grasp) -> float:
    poly_a = Polygon(a.corners())
    poly_b = Polygon(b.corners())
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    union = poly_a.union(poly_b).area
    if union <= 0.0:
        return 0.0
    return float(poly_a.intersection(poly_b).area / union)


def angle_diff(theta_a: float, theta_b: float) -> float:
    diff = abs(theta_a - theta_b) % np.pi
    return float(min(diff, np.pi - diff))


def is_correct_grasp(
    pred: Grasp,
    targets: Sequence[Grasp],
    iou_threshold: float = 0.25,
    angle_threshold: float = np.deg2rad(30),
) -> bool:
    for tgt in targets:
        if (grasp_iou(pred, tgt) > iou_threshold
                and angle_diff(pred.theta, tgt.theta) < angle_threshold):
            return True
    return False


class GraspMetric:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        angle_threshold_deg: float = 30.0,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.angle_threshold = np.deg2rad(angle_threshold_deg)
        self.num_correct = 0
        self.num_total = 0

    def reset(self) -> None:
        self.num_correct = 0
        self.num_total = 0

    def update(
        self,
        predictions: List[List[Grasp]],
        targets: List[List[Grasp]],
    ) -> None:
        assert len(predictions) == len(targets), "batch size mismatch"
        for preds, tgts in zip(predictions, targets):
            self.num_total += 1
            if not preds:
                continue
            if is_correct_grasp(preds[0], tgts, self.iou_threshold, self.angle_threshold):
                self.num_correct += 1

    def compute(self) -> float:
        if self.num_total == 0:
            return 0.0
        return self.num_correct / self.num_total


def harmonic_mean(a: float, b: float, eps: float = 1e-8) -> float:
    if a + b < eps:
        return 0.0
    return float(2.0 * a * b / (a + b))


def split_base_new(
    label_counts: Dict[str, int],
    base_fraction: float = 0.70,
) -> tuple[Set[str], Set[str]]:
    if not label_counts:
        return set(), set()
    # Sort labels by descending frequency
    sorted_labels = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
    n = len(sorted_labels)
    cut = max(1, int(round(n * base_fraction)))
    base = {lbl for lbl, _ in sorted_labels[:cut]}
    new  = {lbl for lbl, _ in sorted_labels[cut:]}
    return base, new


class BaseToNewMetric:
    def __init__(
        self,
        base_labels: Set[str],
        new_labels: Set[str],
        iou_threshold: float = 0.25,
        angle_threshold_deg: float = 30.0,
    ) -> None:
        self.base_labels = base_labels
        self.new_labels  = new_labels
        self.iou_threshold = iou_threshold
        self.angle_threshold = np.deg2rad(angle_threshold_deg)

        self.base_correct = 0
        self.base_total   = 0
        self.new_correct  = 0
        self.new_total    = 0
        self.unknown_skipped = 0

    def reset(self) -> None:
        self.base_correct = self.base_total = 0
        self.new_correct  = self.new_total  = 0
        self.unknown_skipped = 0

    def update(
        self,
        predictions: List[List[Grasp]],
        targets:     List[List[Grasp]],
        labels:      List[str],
    ) -> None:
        assert len(predictions) == len(targets) == len(labels), \
            "batch size mismatch between preds/targets/labels"

        for preds, tgts, lab in zip(predictions, targets, labels):
            in_base = lab in self.base_labels
            in_new  = lab in self.new_labels
            if not (in_base or in_new):
                # Label not in our base/new split — skip (and count it)
                self.unknown_skipped += 1
                continue

            correct = (
                preds
                and is_correct_grasp(preds[0], tgts,
                                     self.iou_threshold, self.angle_threshold)
            )

            if in_base:
                self.base_total += 1
                if correct:
                    self.base_correct += 1
            else:
                self.new_total += 1
                if correct:
                    self.new_correct += 1

    def compute(self) -> Dict[str, float]:
        base = self.base_correct / self.base_total if self.base_total else 0.0
        new  = self.new_correct  / self.new_total  if self.new_total  else 0.0
        h    = harmonic_mean(base, new)
        return {
            "base":             base,
            "new":              new,
            "H":                h,
            "base_count":       self.base_total,
            "new_count":        self.new_total,
            "unknown_skipped":  self.unknown_skipped,
        }


# Common stopwords / verbs in grasp prompts that we want to skip.
_STOPWORDS = {
    "grasp", "pick", "pickup", "hold", "give", "bring", "get", "take", "use",
    "the", "a", "an", "its", "this", "that", "me", "by", "at", "on", "in",
    "with", "of", "from", "up", "to", "and", "or", "for", "into", "onto",
    "please", "can", "could", "would", "you",
}

def extract_label_from_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    # Normalise: lowercase, strip trailing punctuation, split on whitespace
    tokens = [
        t.strip(".,;:!?\"'()[]").lower()
        for t in prompt.split()
    ]
    # Keep only "content" tokens
    content = [t for t in tokens if t and t not in _STOPWORDS]
    if not content:
        return ""
    # The main object noun in Grasp-Anything prompts is typically the first
    # content noun after the verb. "Pick up apple by its skin" -> "apple"
    return content[0]
