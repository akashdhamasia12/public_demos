from dataclasses import dataclass
import numpy as np
import PIL.Image
import cv2
import torch
import kornia
from kornia.feature import LoFTR
import matplotlib.pyplot as plt

from data_preprocessing.utils.generic import resize_image, normalize_image_coordinates, denormalize_image_coordinates


class MatchingError(Exception):
    pass


@dataclass
class MatchingPair:
    frame1: PIL.Image
    frame2: PIL.Image
    pts1_2d: np.ndarray
    pts2_2d: np.ndarray
    pts_3d: np.ndarray


@dataclass
class MatchingConfig:
    # parameters for DESCRIPTOR matching
    lowes_ratio: float = 0.8

    # parameters for FLANN matching
    flann_algorithm: str = 'KDTREE'
    flann_branching: int = 8
    flann_iterations: int = 10
    flann_tree: int = 8
    flann_checks: int = 20

    # parameters for LOFTR matching
    loftr_process_size: int = 1024

    # parameters for matching
    matching_gps_distance: float = 150.0

    # parameters for robust matching
    robust_matching_threshold: float = 0.004
    robust_matching_min_match: int = 20


def build_flann_index(desc: np.ndarray, config: MatchingConfig) -> cv2.flann_Index:
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    FLANN_INDEX_LSH = 6

    if desc.dtype.type is np.float32:
        algorithm_type = config.flann_algorithm.upper()
        if algorithm_type == 'KMEANS':
            FLANN_INDEX_METHOD = FLANN_INDEX_KMEANS
        elif algorithm_type == 'KDTREE':
            FLANN_INDEX_METHOD = FLANN_INDEX_KDTREE
        else:
            raise ValueError(f'Unknown flann algorithm type {algorithm_type}')
    else:
        FLANN_INDEX_METHOD = FLANN_INDEX_LSH

    flann_params = {
        'algorithm': FLANN_INDEX_METHOD,
        'branching': config.flann_branching,
        'iterations': config.flann_iterations,
        'tree': config.flann_tree,
    }

    return cv2.flann_Index(desc, flann_params)  # not deterministic


def match_flann(idx1: cv2.flann_Index, desc2: np.ndarray, config: MatchingConfig) -> np.ndarray:
    search_params = dict(checks=config.flann_checks)
    results, dists = idx1.knnSearch(desc2, 2, params=search_params)
    squared_ratio = config.lowes_ratio**2  # FLANN returns squared L2 distances
    good = dists[:, 0] < squared_ratio * dists[:, 1]
    return np.array(list(zip(results[good, 0], good.nonzero()[0])))


def match_flann_symmetric(desc1: np.ndarray, idx1: cv2.flann_Index, desc2: np.ndarray, idx2: cv2.flann_Index,
                          config: MatchingConfig) -> np.ndarray:
    matches_ij = [(a, b) for a, b in match_flann(idx1, desc2, config)]
    matches_ji = [(b, a) for a, b in match_flann(idx2, desc1, config)]
    return np.array(list(set(matches_ij).intersection(set(matches_ji))))



def match_descriptor_robust(pts1: np.ndarray, desc1: np.ndarray, idx1: cv2.flann_Index, pts2: np.ndarray,
                            desc2: np.ndarray, idx2: cv2.flann_Index, config: MatchingConfig) -> np.ndarray:
    matches = match_flann_symmetric(desc1, idx1, desc2, idx2, config)
    if len(matches) < config.robust_matching_min_match:
        raise MatchingError(f'Matching failed because only {len(matches)} matches found')

    matches_robust = filter_matches(pts1, pts2, matches, config)
    if len(matches_robust) < config.robust_matching_min_match:
        raise MatchingError(f'Matching failed because only {len(matches_robust)} robust matches found')

    return matches_robust


def prepare_image_for_loftr(img: np.ndarray, loftr_process_size: int) -> torch.tensor:
    img_resized = resize_image(img, loftr_process_size)
    img_tensor = kornia.image_to_tensor(img_resized, False).float() / 255.0
    img_tensor_gray = kornia.color.bgr_to_grayscale(img_tensor)
    return img_tensor_gray


def match_loftr(img1: np.ndarray, img2: np.ndarray, config: MatchingConfig) -> tuple:
    img1_prepared = prepare_image_for_loftr(img1, config.loftr_process_size)
    img2_prepared = prepare_image_for_loftr(img2, config.loftr_process_size)

    matcher = LoFTR(pretrained='outdoor')
    input_dict = {"image0": img1_prepared, "image1": img2_prepared}
    with torch.no_grad():
        correspondences = matcher(input_dict)

    pts1 = correspondences['keypoints0'].cpu().numpy()
    pts2 = correspondences['keypoints1'].cpu().numpy()
    assert len(pts1) == len(pts2), 'Number of feature points differ'
    pts1 = normalize_image_coordinates(pts1, img1_prepared.shape[3], img1_prepared.shape[2])
    pts2 = normalize_image_coordinates(pts2, img2_prepared.shape[3], img2_prepared.shape[2])

    matches = np.repeat([range(len(pts1))], 2, axis=0).T
    return pts1, pts2, matches


def match_loftr_robust(img1: np.ndarray, img2: np.ndarray, config: MatchingConfig) -> tuple:
    pts1, pts2, matches = match_loftr(img1, img2, config)
    if len(matches) < config.robust_matching_min_match:
        raise MatchingError(f'Matching failed because only {len(matches)} matches found')

    matches_robust = filter_matches(pts1, pts2, matches, config)
    if len(matches_robust) < config.robust_matching_min_match:
        raise MatchingError(f'Matching failed because only {len(matches_robust)} robust matches found')

    return pts1, pts2, matches_robust


def filter_matches(pts1: np.ndarray, pts2: np.ndarray, matches: np.ndarray, config: MatchingConfig) -> np.ndarray:
    if len(matches) < 8:
        return np.array([])

    pts1 = pts1[matches[:, 0]][:, :2].copy()
    pts2 = pts2[matches[:, 1]][:, :2].copy()

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, config.robust_matching_threshold, 0.9999)
    inliers = mask.ravel().nonzero()

    if F is None or F[2, 2] == 0.0:
        return np.array([])

    return matches[inliers]


def plot_matches(ax: plt.Axes, img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> None:
    # based on https://github.com/mapillary/OpenSfM/blob/main/bin/plot_matches.py#L17
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=img1.dtype)
    img[0:h1, 0:w1, :] = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img[0:h2, w1:(w1 + w2), :] = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    pts1 = denormalize_image_coordinates(pts1, w1, h1)
    pts2 = denormalize_image_coordinates(pts2, w2, h2)
    ax.imshow(img)
    for a, b in zip(pts1, pts2):
        ax.plot([a[0], b[0] + w1], [a[1], b[1]], linewidth=1)

    ax.plot(pts1[:, 0], pts1[:, 1], 'ob', markersize=3)
    ax.plot(pts2[:, 0] + w1, pts2[:, 1], 'ob', markersize=3)


def plot_matching_pair(ax: plt.Axes, matching_pair: MatchingPair) -> None:
    img1 = cv2.cvtColor(np.array(matching_pair.frame1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(matching_pair.frame2), cv2.COLOR_RGB2BGR)
    plot_matches(ax, img1, img2, matching_pair.pts1_2d, matching_pair.pts2_2d)


def plot_matching_pair_heatmap(ax: plt.Axes, matching_pair: MatchingPair) -> None:
    img = np.array(matching_pair.frame1)
    h1, w1 = img.shape[:2]
    pts1 = denormalize_image_coordinates(matching_pair.pts1_2d, w1, h1)
    ax.imshow(img)
    ax.plot(pts1[:, 0], pts1[:, 1], 'or', markersize=10, alpha=0.1, markeredgewidth=0)
    ax.plot(pts1[:, 0], pts1[:, 1], 'or', markersize=5, alpha=0.1, markeredgewidth=0)
