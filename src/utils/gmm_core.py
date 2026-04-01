from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

SEGMENT_COLORS = [
    (60,  180, 75),  (255, 225,  25), (0,  130, 200), (245, 130,  48),
    (145,  30, 180), (70,  240, 240), (240,  50, 230), (210, 245,  60),
    (250, 190, 212), (0,  128, 128),
]

_GMM_MAX_SIDE = 512


def encode_rle(mask: np.ndarray) -> list[int]:
    flat = mask.ravel(order='F').astype(bool)
    if flat.size == 0:
        return []

    changes = np.diff(flat.view(np.uint8))
    change_positions = np.flatnonzero(changes) + 1  # start of each new run in flat

    boundaries = np.concatenate(([0], change_positions, [flat.size]))
    runs = np.diff(boundaries).tolist()

    if flat[0]:
        runs = [0] + runs

    return [int(r) for r in runs]


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return [0, 0, 0, 0]

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def run_gmm(image_bgr: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Fit a Gaussian Mixture Model on HSV pixels and return an H×W int32 label map.

    n_components=1 is handled as a special case: all pixels get label 0
    without invoking scikit-learn.

    Down-scales large images to _GMM_MAX_SIDE before fitting, then scales
    labels back with nearest-neighbour interpolation for speed.
    """
    if n_components == 1:
        return np.zeros(image_bgr.shape[:2], dtype=np.int32)

    h, w = image_bgr.shape[:2]

    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    scale = min(1.0, _GMM_MAX_SIDE / max(h, w))
    if scale < 1.0:
        small_h = max(1, int(h * scale))
        small_w = max(1, int(w * scale))
        small = cv2.resize(blurred, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        small = blurred
        small_h, small_w = h, w

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float64)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=100,
        random_state=42,
        n_init=3,
        reg_covar=1e-3,
    )
    labels_small = gmm.fit_predict(pixels).reshape(small_h, small_w)

    if scale < 1.0:
        labels = cv2.resize(
            labels_small.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    else:
        labels = labels_small.astype(np.int32)

    return labels


def explode_gmm_clusters(
    image_bgr: np.ndarray,
    n_components: int = 2,
    point: Optional[tuple[int, int]] = None,
) -> dict:
    """
    Run GMM on the full image and return component masks.

    Parameters
    ----------
    image_bgr   : full-size BGR image
    n_components: number of Gaussian components
    point       : optional (x, y) — if given, returns only the single
                  component whose label covers that pixel

    Returns
    -------
    dict with keys: masks (list of dicts), n_components, width, height
    Each mask dict: cluster_id, mask_rle, box_2d
    """
    full_h, full_w = image_bgr.shape[:2]

    labels = run_gmm(image_bgr, n_components=n_components)

    if point is not None:
        px, py = point
        if not (0 <= px < full_w and 0 <= py < full_h):
            raise ValueError(
                f"Point ({px},{py}) is outside the image ({full_w}×{full_h})"
            )
        target_id = int(labels[py, px])
        binary_mask = (labels == target_id).astype(bool)
        return {
            "masks": [{
                "cluster_id": target_id,
                "mask_rle": encode_rle(binary_mask),
                "box_2d": mask_to_bbox(binary_mask),
            }],
            "n_components": n_components,
            "width": full_w,
            "height": full_h,
        }

    masks = []
    for comp_id in np.unique(labels):
        comp_id = int(comp_id)
        binary_mask = (labels == comp_id).astype(bool)
        if binary_mask.sum() == 0:
            continue
        masks.append({
            "cluster_id": comp_id,
            "mask_rle": encode_rle(binary_mask),
            "box_2d": mask_to_bbox(binary_mask),
        })

    return {
        "masks": masks,
        "n_components": n_components,
        "width": full_w,
        "height": full_h,
    }
