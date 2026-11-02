"""Probability map -> non-overlapping instance masks -> RLE rows.

Instances are connected components of the thresholded map (annotators' filaments essentially never touch: <0.6 %
even after 2 px dilation, analysis/check_raster.py), so connected components are a sound instance definition and
are disjoint by construction, which the submission validator requires. Parameters come from config `postprocess.*`
and are chosen globally on OOF (protocol.md §1, at most 4 free parameters per experiment), never per fold.

Steps: upsample prob to 2048 -> threshold -> disk mask -> closing -> (merge fragments within merge_radius)
       -> connected components -> drop area < min_area -> drop mean prob < score_min
"""

from __future__ import annotations

import cv2
import numpy as np

from src.annotations import IMAGE_SIZE, mask_to_rle


def upsample_prob(prob: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    if prob.shape[0] == size:
        return prob.astype(np.float32)
    return cv2.resize(prob.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def label_components(binary: np.ndarray, merge_radius: int = 0) -> tuple[int, np.ndarray]:
    """Connected components of `binary`. With merge_radius > 0, components whose dilations by that radius touch
    share one label (fragment bridging) while the pixels themselves stay unchanged."""
    if merge_radius <= 0:
        n, labels = cv2.connectedComponents(binary, connectivity=8)
        return n, labels
    dilated = cv2.dilate(binary, _ellipse(merge_radius))
    n, labels = cv2.connectedComponents(dilated, connectivity=8)
    labels = labels * binary  # keep only the original pixels
    return n, labels


def hysteresis(binary: np.ndarray, p: np.ndarray, seed_threshold: float) -> np.ndarray:
    """Keep only the connected components of `binary` (p > low threshold) that contain a pixel with
    p > seed_threshold (postprocess.seed_threshold; 0 = off). Low threshold grows the mask / bridges gaps, the high
    threshold decides existence."""
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    keep = np.unique(labels[(p > seed_threshold) & (binary > 0)])
    return np.isin(labels, keep[keep > 0]).astype(np.uint8)


def instances_from_prob(prob: np.ndarray, threshold: float, min_area: int, closing_radius: int,
                        disk_mask: np.ndarray | None = None, merge_radius: int = 0,
                        score_min: float = 0.0, seed_threshold: float = 0.0) -> list[dict]:
    """Return [{'mask': (H,W) uint8, 'area': int, 'score': mean prob}] for one image at full resolution."""
    p = upsample_prob(prob)
    binary = (p > threshold).astype(np.uint8)
    if disk_mask is not None:
        binary &= disk_mask.astype(np.uint8)
    if closing_radius > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ellipse(closing_radius))
    if seed_threshold > threshold:
        binary = hysteresis(binary, p, seed_threshold)
    n, labels = label_components(binary, merge_radius)
    out = []
    for i in range(1, n):
        mask = labels == i
        area = int(mask.sum())
        if area < min_area:
            continue
        score = float(p[mask].mean())
        if score < score_min:
            continue
        out.append({"mask": mask.astype(np.uint8), "area": area, "score": score})
    return out


def instances_to_rows(stem: str, instances: list[dict]) -> list[dict]:
    """Submission rows for one image; ids are '<stem>_<k>' with k from 1 in descending score order."""
    ordered = sorted(instances, key=lambda d: -d["score"])
    return [{"filament_id": f"{stem}_{k}", "segmentation_rle": mask_to_rle(inst["mask"]),
             "score": inst["score"], "area": inst["area"]}
            for k, inst in enumerate(ordered, start=1)]


def prob_to_rows(stem: str, prob: np.ndarray, cfg: dict, disk_mask: np.ndarray | None = None) -> list[dict]:
    inst = instances_from_prob(prob, cfg["threshold"], cfg["min_area"], cfg["closing_radius"], disk_mask,
                               merge_radius=cfg.get("merge_radius", 0), score_min=cfg.get("score_min", 0.0),
                               seed_threshold=cfg.get("seed_threshold", 0.0))
    return instances_to_rows(stem, inst)
