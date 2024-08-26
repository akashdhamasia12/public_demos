import os
import numpy as np
import cv2
from PIL import Image
import piexif
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from data_preprocessing.metas.video_meta import VideoMeta
from data_preprocessing.utils.video_capture import VideoCapture


@dataclass
class FrameIdExtractor:
    start_frame_id: int
    frame_count: int
    fps: float
    total_frames: Optional[int]
    subtitles: None

    def __post_init__(self):
        assert 0 <= self.start_frame_id < self.frame_count, \
            f'Start frame id {self.start_frame_id} oustide of valid interval [0, {self.frame_count - 1}]'

    def get_stop_frame_id(self, frame_dist: int) -> int:
        stop_frame_id_candidates = [self.frame_count - 1]
        if self.total_frames is not None:
            stop_frame_id_candidates.append(self.start_frame_id + self.total_frames * frame_dist - 1)
        if self.subtitles is not None:
            stop_frame_id_candidates.append(self.subtitles.get_last_valid_frame_id(self.fps))
        return min(stop_frame_id_candidates)

    def get_frame_ids_from_frame_dist(self, frame_dist: int) -> list:
        stop_frame_id = self.get_stop_frame_id(frame_dist)
        return list(range(self.start_frame_id, stop_frame_id + 1, frame_dist))


def save_image(img: np.ndarray, img_basename: str, img_format: str, frame_id: int, max_digits: int, exif_dict: dict,
               save_dir: str) -> str:
    img_fname = img_basename + "_" + str(frame_id).zfill(max_digits) + '.' + img_format
    img_file = os.path.join(save_dir, img_fname)
    # jpg
    if img_format == 'jpg':
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(img_file, exif=piexif.dump(exif_dict))
    # png
    elif img_format == 'png':
        cv2.imwrite(img_file, img)
    else:
        raise ValueError(f'Unknown file format {img_format}')
    return img_file


def extract_frames(path_to_video: str, save_dir: str, total_frames: Optional[int] = None, start_frame_id: int = 0,
                   frame_dist: Optional[int] = None, min_dist: Optional[float] = None, min_speed: float = 0.0,
                   img_format: str = 'jpg', require_exif: bool = False, quiet: bool = False) -> list:
    assert (frame_dist is not None) ^ (min_dist is not None), 'Either frame_dist or min_dist must be defined'
    if require_exif:
        assert img_format == 'jpg', 'Exif requires jpg as image format'

    # init
    video_name = os.path.splitext(os.path.basename(path_to_video))[0]
    video_meta = VideoMeta(path_to_video)
    video_cap = VideoCapture(path_to_video)
    max_digits = len(str(video_cap.frame_count))
    subtitles = Subtitles.load_from_video_file(path_to_video) if require_exif else None

    # get frame ids
    frame_id_extractor = FrameIdExtractor(start_frame_id, video_cap.frame_count, video_cap.fps, total_frames, subtitles)
    if frame_dist is not None:
        frame_ids = frame_id_extractor.get_frame_ids_from_frame_dist(frame_dist)
    else:
        frame_ids = frame_id_extractor.get_frame_ids_from_min_dist(min_dist, min_speed)

    # extract frames
    img_files = []
    for frame_id in tqdm(frame_ids, desc=f'Extracting frames', disable=quiet):
        success, img = video_cap.get_frame(frame_id)
        assert success, f'Extracting frame {frame_id} failed'

        # save image
        exif_dict = get_exif_dict(video_meta, subtitles, frame_id, video_cap.fps) if require_exif else {}
        img_file = save_image(img, video_name, img_format, frame_id, max_digits, exif_dict, save_dir)
        img_files.append(img_file)

    video_cap.release()
    return img_files

