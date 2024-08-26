import os
from abc import ABC, abstractmethod
import numpy as np
import PIL.Image
import PIL.ExifTags
import PIL.TiffImagePlugin
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_preprocessing.utils.reference_mapping import ReferenceMapping
from data_preprocessing.utils.generic import denormalize_image_coordinates
from data_preprocessing.utils.opensfm import get_extrinsic_matrix
from data_preprocessing.utils.matching import MatchingError, MatchingPair, MatchingConfig, match_loftr_robust, \
    plot_matching_pair, plot_matching_pair_heatmap
from data_preprocessing.raycast_features import raycast_features



class MatcherABC(ABC):
    def __init__(self, rec_frame: PIL.Image.Image, _: ReferenceMapping) -> None:
        self.rec_frame = rec_frame
        self.rec_img = cv2.cvtColor(np.array(rec_frame), cv2.COLOR_RGB2BGR)
        self.matching_config = None

    @abstractmethod
    def match(self, ref_frame: PIL.Image.Image) -> tuple:
        pass



class LoftrMatcher(MatcherABC):
    def __init__(self, rec_frame: PIL.Image.Image, ref_mapping: ReferenceMapping) -> None:
        super().__init__(rec_frame, ref_mapping)
        self.matching_config = MatchingConfig(loftr_process_size=ref_mapping.feature_config.feature_process_size // 2)

    def match(self, ref_frame: PIL.Image.Image) -> tuple:
        ref_img = cv2.cvtColor(np.array(ref_frame), cv2.COLOR_RGB2BGR)
        ref_pts_2d, rec_pts_2d, matches = match_loftr_robust(ref_img, self.rec_img, self.matching_config)
        return ref_pts_2d[matches[:, 0]], rec_pts_2d[matches[:, 1]]


def match_reference_images(matcher: MatcherABC, ref_img_fnames: list, ref_mapping: ReferenceMapping, use_gps: bool,
                           quiet: bool = False) -> list:
    matching_pairs = []
    for ref_img_fname in tqdm(ref_img_fnames, desc='Matching reference images', disable=quiet):
        # skip reference image based on GPS distance
        ref_frame = PIL.Image.open(os.path.join(ref_mapping.images_dir, ref_img_fname))
        assert not use_gps, 'Not needed for this showcase'

        try:
            ref_pts_2d_matched, rec_pts_2d_matched = matcher.match(ref_frame)
            extrinsic_matrix = get_extrinsic_matrix(ref_mapping.reconstruction, shot_id=ref_img_fname)
            ref_img_width, ref_img_height = ref_frame.size
            ref_pts_2d_matched_denorm = denormalize_image_coordinates(ref_pts_2d_matched, ref_img_width, ref_img_height)
            ref_pts_3d_matched, idxs_hit = raycast_features(ref_pts_2d_matched_denorm, ref_mapping.dist_coeffs,
                                                            ref_mapping.intrinsic_matrix, extrinsic_matrix,
                                                            ref_mapping.mesh)
            ref_pts_2d_matched = ref_pts_2d_matched[idxs_hit]
            ref_pts_3d_matched = ref_pts_3d_matched[idxs_hit]
            rec_pts_2d_matched = rec_pts_2d_matched[idxs_hit]
            pair = MatchingPair(ref_frame, matcher.rec_frame, ref_pts_2d_matched, rec_pts_2d_matched,
                                ref_pts_3d_matched)
            matching_pairs.append(pair)
        except MatchingError:
            continue

    return matching_pairs


def match_reference_images_wrapper(rec_frame: PIL.Image.Image, calib_fnames: list, ref_mapping: ReferenceMapping,
                                   quiet: bool = False) -> list:
    ref_img_fnames = calib_fnames if calib_fnames else ref_mapping.reconstruction['shots'].keys()
    use_gps = len(calib_fnames) == 0

    Matcher = LoftrMatcher if len(calib_fnames) == 1 else DescriptorMatcher
    matcher = Matcher(rec_frame, ref_mapping)
    return match_reference_images(matcher, ref_img_fnames, ref_mapping, use_gps, quiet=quiet)


def plot_matching_pairs(matching_pairs: list, save_dir: str, quiet: bool = False) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for pair in tqdm(matching_pairs, desc='Plotting matches', disable=quiet):
        img1_fname = os.path.basename(os.path.splitext(pair.frame1.filename)[0])
        img2_fname = os.path.basename(os.path.splitext(pair.frame2.filename)[0])
        # plot matches:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_matching_pair(ax, pair)
        ax.set_title(f'Images: {img1_fname} - {img2_fname}, matches {len(pair.pts1_2d)}')
        fig.savefig(os.path.join(save_dir, f'{img1_fname}__{img2_fname}.jpg'), bbox_inches='tight', dpi=200)
        plt.close(fig=fig)
        # plot heatmap:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_matching_pair_heatmap(ax, pair)
        ax.set_title(f'Images Heatmap: {img1_fname} - {img2_fname}, matches {len(pair.pts1_2d)}')
        fig.savefig(os.path.join(save_dir, f'{img1_fname}__{img2_fname}_heatmap.jpg'), bbox_inches='tight', dpi=200)
        plt.close()
