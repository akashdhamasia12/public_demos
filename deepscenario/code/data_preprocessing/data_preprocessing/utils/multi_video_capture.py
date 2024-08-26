import numpy as np
import cv2
from typing import Tuple


class MultiVideoCapture:
    def __init__(self, path_to_videos: list) -> None:
        self.path_to_videos = path_to_videos
        self.video_caps = [cv2.VideoCapture(path_to_video) for path_to_video in path_to_videos]
        self.frame_counts = [int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) for video_cap in self.video_caps]
        self.total_frame_count = sum(self.frame_counts)
        self.frame_offsets = np.cumsum([0] + self.frame_counts[:-1])
        self.frame_ids_curr = self.frame_offsets.copy()
        self.fps = self.video_caps[0].get(cv2.CAP_PROP_FPS)

    @property
    def n_videos(self) -> int:
        return len(self.video_caps)

    def is_valid(self, frame_id: int) -> bool:
        return 0 <= frame_id < self.total_frame_count

    def get_video_index(self, frame_id: int) -> int:
        assert self.is_valid(frame_id), \
            f'Frame id {frame_id} outside of valid interval [0, {self.total_frame_count - 1}]'
        return np.where(frame_id >= self.frame_offsets)[0][-1]

    def get_frame(self, frame_id: int) -> Tuple[bool, np.ndarray]:
        # get corresponding video capture
        idx_curr = self.get_video_index(frame_id)
        video_cap_curr = self.video_caps[idx_curr]
        frame_id_curr = self.frame_ids_curr[idx_curr]

        assert frame_id >= frame_id_curr, f'Requested frame in the past: {frame_id} < {frame_id_curr}.'
        while not frame_id_curr == frame_id:
            video_cap_curr.grab()
            frame_id_curr += 1

        success, img = video_cap_curr.read()
        self.frame_ids_curr[idx_curr] = frame_id + 1
        return success, img

    def release(self) -> None:
        for video_cap in self.video_caps:
            video_cap.release()

    def __del__(self):
        self.release()