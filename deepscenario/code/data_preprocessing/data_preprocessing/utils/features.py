# This file is based on https://github.com/mapillary/OpenSfM/blob/master/opensfm/features.py

from dataclasses import dataclass
import numpy as np
import cv2

from data_preprocessing.utils.generic import resize_image, normalize_image_coordinates


@dataclass
class FeatureConfig:
    # parameters for feature extraction
    feature_type: str = 'SIFT'
    feature_min_frames: int = 4000
    feature_process_size: int = 2048

    # parameters for SIFT
    sift_peak_threshold: float = 0.066
    sift_edge_threshold: float = 10.0


def get_root_features(desc: np.ndarray, l2_normalization=False) -> np.ndarray:
    if l2_normalization:
        s2 = np.linalg.norm(desc, axis=1)
        desc = (desc.T / s2).T
    s = np.sum(desc, 1)
    desc = np.sqrt(desc.T / s).T
    return desc


def extract_features_sift(img: np.ndarray, config: FeatureConfig) -> tuple:
    sift_edge_threshold, sift_peak_threshold = config.sift_edge_threshold, config.sift_peak_threshold
    detector = cv2.SIFT_create(edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold)
    descriptor = detector
    while True:
        detector = cv2.SIFT_create(edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold)
        pts = detector.detect(img)
        if len(pts) < config.feature_min_frames and sift_peak_threshold > 0.0001:
            sift_peak_threshold = (sift_peak_threshold * 2) / 3
        else:
            break
    pts, desc = descriptor.compute(img, pts)
    desc = get_root_features(desc)
    pts = np.array([(pt.pt[0], pt.pt[1]) for pt in pts])
    return pts, desc


def extract_features(img: np.ndarray, config: FeatureConfig) -> tuple:
    # convert image: RGB (https://github.com/mapillary/OpenSfM/blob/main/opensfm/io.py#L1226) -> resize -> grayscale
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_resized = resize_image(img_rgb, config.feature_process_size)
    img_rgb_resized_gray = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2GRAY)

    feature_type = config.feature_type.upper()
    if feature_type == 'SIFT':
        pts, desc = extract_features_sift(img_rgb_resized_gray, config)
    elif feature_type == 'ORB':
        pts, desc = extract_features_orb(img_rgb_resized_gray, config)
    else:
        raise ValueError( f'Unknown feature type {feature_type} (must be SIFT or ORB)')

    pts[:, :2] = normalize_image_coordinates(pts[:, :2], img_rgb_resized_gray.shape[1], img_rgb_resized_gray.shape[0])
    return pts, desc
