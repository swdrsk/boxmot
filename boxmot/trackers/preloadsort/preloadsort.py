# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path
from boxmot.utils import logger as LOGGER

import numpy as np
import torch

from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.preloadsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.preloadsort.preloadsort_track import STrack
from boxmot.trackers.preloadsort.preloadsort_utils import (joint_stracks,
                                                   remove_duplicate_stracks,
                                                   sub_stracks)
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
import cv2


class PreloadSort(BaseTracker):
    """
    Initialize the PreloadSort tracker with various parameters.
    
    PreloadSort extends BotSort to track only pre-registered persons using reference images.

    Parameters:
    - reid_weights (Path): Path to the re-identification model weights.
    - device (torch.device): Device to run the model on (e.g., 'cpu', 'cuda').
    - half (bool): Whether to use half-precision (fp16) for faster inference.
    - det_thresh (float): Detection threshold for considering detections.
    - max_age (int): Maximum age (in frames) of a track before it is considered lost.
    - max_obs (int): Maximum number of historical observations stored for each track. Always greater than max_age by minimum 5.
    - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
    - iou_threshold (float): IOU threshold for determining match between detection and tracks.
    - per_class (bool): Enables class-separated tracking.
    - nr_classes (int): Total number of object classes that the tracker will handle (for per_class=True).
    - asso_func (str): Algorithm name used for data association between detections and tracks.
    - is_obb (bool): Work with Oriented Bounding Boxes (OBB) instead of standard axis-aligned bounding boxes.
    
    PreloadSort-specific parameters:
    - registered_images_path (Path): Path to folder containing registered person images.
    - match_threshold (float): Similarity threshold for matching with registered images (0.0-1.0).
    
    BotSort-specific parameters:
    - track_high_thresh (float): Detection confidence threshold for first association.
    - track_low_thresh (float): Detection confidence threshold for ignoring detections.
    - new_track_thresh (float): Threshold for creating a new track.
    - track_buffer (int): Frames to keep a track alive after last detection.
    - match_thresh (float): Matching threshold for data association.
    - proximity_thresh (float): IoU threshold for first-round association.
    - appearance_thresh (float): Appearance embedding distance threshold for ReID.
    - cmc_method (str): Method for correcting camera motion, e.g., "sof" (simple optical flow).
    - frame_rate (int): Video frame rate, used to scale the track buffer.
    - fuse_first_associate (bool): Fuse appearance and motion in the first association step.
    - with_reid (bool): Use ReID features for association.
    
    Attributes:
    - frame_count (int): Counter for the frames processed.
    - active_tracks (list): List to hold active tracks.
    - lost_stracks (list[STrack]): List of lost tracks.
    - removed_stracks (list[STrack]): List of removed tracks.
    - buffer_size (int): Size of the track buffer based on frame rate.
    - max_time_lost (int): Maximum time a track can be lost.
    - kalman_filter (KalmanFilterXYWH): Kalman filter for motion prediction.
    - registered_ids (dict): Mapping of registered person IDs to their averaged embeddings.
    - match_threshold (float): Threshold for matching detections with registered images.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        # PreloadSort-specific parameters
        registered_images_path: Path = None,
        match_threshold: float = 0.5,
        enable_scene_change_detection: bool = False,  # デフォルトで無効
        scene_change_threshold: float = 0.3,
        # BotSort-specific parameters
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "ecc",
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        **kwargs  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='PreloadSort', **kwargs)
        
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        if self.with_reid:
            self.model = ReidAutoBackend(
                weights=reid_weights, device=device, half=half
            ).model

        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate
        
        # PreloadSort-specific: registered images
        self.registered_ids = {}  # {id: list of embeddings}
        self.match_threshold = match_threshold
        
        # Scene change detection
        self.enable_scene_change_detection = enable_scene_change_detection
        self.prev_frame = None
        self.scene_change_threshold = scene_change_threshold
        
        # Optimization settings
        self.reverify_interval = 10  # Frames
        self.static_threshold = 30   # Frames
        self.static_movement_threshold = 2.0  # Pixel movement to be considered "static"
        
        if registered_images_path:
            self._load_registered_images(registered_images_path)
    
    def _load_registered_images(self, path: Path):
        """Load registered person images and extract their features (Winner-take-all)."""
        person_dirs = sorted(path.iterdir())
        
        for idx, person_dir in enumerate(person_dirs, start=1):
            if not person_dir.is_dir():
                continue
            
            # Use folder name as ID if it's an integer, otherwise use sequential number
            try:
                person_id = int(person_dir.name)
            except ValueError:
                person_id = idx
            
            embeddings = []
            for img_file in person_dir.glob('*.png'):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                # Treat entire image as bbox
                h, w = img.shape[:2]
                bbox = np.array([[0, 0, w, h]])
                emb = self.model.get_features(bbox, img)
                # Normalize each embedding individually
                emb_normalized = emb[0] / np.linalg.norm(emb[0])
                embeddings.append(emb_normalized)
            
            if len(embeddings) == 0:
                continue
                
            # Store all embeddings (not averaged) for Winner-take-all matching
            self.registered_ids[person_id] = embeddings
    
    def _match_detection_with_registered(self, detection):
        """Match a single detection with registered person embeddings (Winner-take-all)."""
        if detection.curr_feat is None:
            return None
        
        best_match_id = None
        best_similarity = -1
        
        for reg_id, reg_embs in self.registered_ids.items():
            # Winner-take-all: compare with all registered images and take the maximum
            for reg_emb in reg_embs:
                # Calculate cosine similarity (features are already normalized)
                similarity = np.dot(detection.curr_feat, reg_emb)
                
                if similarity > best_similarity and similarity >= self.match_threshold:
                    best_similarity = similarity
                    best_match_id = reg_id
        
        return best_match_id
    
    def _find_track_by_registered_id(self, registered_id):
        """Find an active or lost track with the given registered ID."""
        all_tracks = self.active_tracks + self.lost_stracks
        for track in all_tracks:
            if hasattr(track, 'registered_id') and track.registered_id == registered_id:
                return track
        return None
    
    def _detect_scene_change(self, img: np.ndarray) -> bool:
        """Detect scene change using frame correlation."""
        # シーン切り替え検出が無効の場合は常にFalseを返す
        if not self.enable_scene_change_detection:
            return False
        
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return False
        
        # Scene change detected if:
        # 1. Correlation is below threshold, OR
        # 2. Frames are identical (correlation = 1.0 and all pixels same)
        curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        correlation = np.corrcoef(self.prev_frame.flatten(), curr_gray.flatten())[0, 1]
        is_identity = np.array_equal(self.prev_frame, curr_gray)
        scene_changed = correlation < self.scene_change_threshold or (correlation > 0.999 and is_identity)
        if scene_changed:
            LOGGER.warning(f"Scene change detected! (correlation: {correlation:.3f}, identity: {is_identity})")
        
        return scene_changed
    
    def _reset_all_tracks(self):
        """Reset all tracks when scene change is detected."""
        LOGGER.info("Resetting all tracks due to scene change")
        self.active_tracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        # Note: frame_count is NOT reset to maintain timeline continuity

    def _reverify_track(self, track: STrack, img: np.ndarray) -> bool:
        """
        Re-verify if the track still matches the registered person's appearance.
        Returns True if verified, False if identity is lost.
        """
        if not hasattr(track, 'registered_id') or track.registered_id is None:
            return True # Should not happen in PreloadSort
            
        # Get current track image
        x1, y1, x2, y2 = map(int, track.xyxy)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        track_img = img[y1:y2, x1:x2]
        # Extract features for current track area
        # Note: Re-using the model logic from BaseTracker/PreloadSort
        bbox = np.array([[0, 0, track_img.shape[1], track_img.shape[0]]])
        current_emb = self.model.get_features(bbox, track_img)[0]
        current_emb /= np.linalg.norm(current_emb)
        
        # Compare with registered embeddings
        reg_embs = self.registered_ids.get(track.registered_id, [])
        max_sim = -1
        for reg_emb in reg_embs:
            sim = np.dot(current_emb, reg_emb)
            if sim > max_sim:
                max_sim = sim
        
        # If similarity drops significantly below match_threshold, we lost the identity
        # Use a slightly more lenient threshold for re-verification to avoid flickering
        reverify_thresh = self.match_threshold * 0.8
        
        is_valid = max_sim >= reverify_thresh
        if not is_valid:
            LOGGER.info(f"Identity lost for track {track.id} (registered_id: {track.registered_id}). Similarity: {max_sim:.3f}")
        
        return is_valid

    def _check_ghost_track(self, track: STrack) -> bool:
        """
        Check if the track is a 'ghost' (stuck on background or non-moving object).
        Returns True if it's a ghost.
        """
        if track.last_xyxy is None:
            track.last_xyxy = track.xyxy
            return False
            
        curr_xyxy = track.xyxy
        # Calculate movement of center point
        prev_center = (track.last_xyxy[:2] + track.last_xyxy[2:]) / 2
        curr_center = (curr_xyxy[:2] + curr_xyxy[2:]) / 2
        movement = np.linalg.norm(curr_center - prev_center)
        
        if movement < self.static_movement_threshold:
            track.static_frames += 1
        else:
            track.static_frames = 0
            
        track.last_xyxy = curr_xyxy
        
        # If stuck for too long, it's likely a ghost in a dynamic scene like dance
        return track.static_frames > self.static_threshold

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.frame_count += 1
        self.excluded_detections = []

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # 1. Periodic Re-authentication & Ghost Killing
        remaining_active = []
        for track in self.active_tracks:
            # Ghost check
            if self._check_ghost_track(track):
                LOGGER.info(f"Ghost track {track.id} removed (static for {track.static_frames} frames)")
                track.mark_removed()
                removed_stracks.append(track)
                continue
                
            # Periodic identity re-verification
            if self.frame_count - track.last_reauth_frame >= self.reverify_interval:
                if not self._reverify_track(track, img):
                    track.mark_removed()
                    removed_stracks.append(track)
                    continue
                track.last_reauth_frame = self.frame_count
            
            remaining_active.append(track)
        self.active_tracks = remaining_active

        # Preprocess detections
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        if self.with_reid and embs is None:
            features_high = self.model.get_features(dets_first[:, 0:4], img)
        else:
            features_high = embs_first if embs_first is not None else []

        # Create detections
        detections = self._create_detections(dets_first, features_high)
        
        # Match each detection with registered images
        det_to_registered = {}  # {det_idx: registered_id}
        for idx, det in enumerate(detections):
            reg_id = self._match_detection_with_registered(det)
            if reg_id is not None:
                det_to_registered[idx] = reg_id
                # Store registered_id in detection for later use
                det.registered_id = reg_id
        
        # Filter detections: only keep those matched with registered images
        filtered_indices = list(det_to_registered.keys())
        filtered_detections = [detections[i] for i in filtered_indices]
        self.excluded_detections = [detections[i] for i in range(len(detections)) if i not in det_to_registered]
        
        # If no registered persons detected, return empty
        if len(filtered_detections) == 0:
            self._update_track_states(removed_stracks)
            return self._prepare_output(
                activated_stracks, refind_stracks, lost_stracks, removed_stracks
            )

        # Separate unconfirmed and active tracks
        unconfirmed, active_tracks = self._separate_tracks()

        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # First association with filtered detections
        matches_first, u_track_first, u_detection_first = self._first_association(
            dets,
            dets_first,
            active_tracks,
            unconfirmed,
            img,
            filtered_detections,
            activated_stracks,
            refind_stracks,
            strack_pool,
        )

        # Second association (skipped for PreloadSort - no low confidence detections for unregistered persons)

        # Handle unconfirmed tracks
        matches_unc, u_track_unc, u_detection_unc = self._handle_unconfirmed_tracks(
            u_detection_first,
            filtered_detections,
            activated_stracks,
            removed_stracks,
            unconfirmed,
        )

        # Initialize new tracks (only for registered persons)
        self._initialize_new_tracks(
            u_detection_unc,
            activated_stracks,
            filtered_detections,
        )

        # Update lost and removed tracks
        self._update_track_states(removed_stracks)

        # Merge and prepare output
        return self._prepare_output(
            activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )

    def _split_detections(self, dets, embs):
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        second_mask = np.logical_and(
            confs > self.track_low_thresh, confs < self.track_high_thresh
        )
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [
                    STrack(det, f, max_obs=self.max_obs)
                    for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for det in dets_first]
        else:
            detections = []
        return detections

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(
        self,
        dets,
        dets_first,
        active_tracks,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):

        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets)
        
        # Detect scene change
        if self._detect_scene_change(img):
            self._reset_all_tracks()
            # Return empty matches since all tracks were reset
            return [], [], list(range(len(detections)))
        
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high confidence detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)
        

        return matches, u_track, u_detection

    def _second_association(
        self,
        dets_second,
        activated_stracks,
        lost_stracks,
        refind_stracks,
        u_track_first,
        strack_pool,
    ):
        # PreloadSort: Skip second association for low confidence detections
        # Only track registered persons
        return [], u_track_first, []

    def _handle_unconfirmed_tracks(
        self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed
    ):
        """
        Handle unconfirmed tracks (tracks with only one detection frame).

        Args:
            u_detection: Unconfirmed detection indices.
            detections: Current list of detections.
            activated_stracks: List of newly activated tracks.
            removed_stracks: List of tracks to remove.
        """
        # Only use detections that are unconfirmed (filtered by u_detection)
        detections = [detections[i] for i in u_detection]

        # Calculate IoU distance between unconfirmed tracks and detections
        ious_dists = iou_distance(unconfirmed, detections)

        # Apply IoU mask to filter out distances that exceed proximity threshold
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)

        # Fuse scores for IoU-based and embedding-based matching (if applicable)
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = (
                1.0  # Apply the IoU mask to embedding distances
            )
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        # Perform data association using linear assignment on the combined distances
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        # Update matched unconfirmed tracks
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        # Mark unmatched unconfirmed tracks as removed
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        return matches, u_unconfirmed, u_detection

    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
        """Initialize new tracks only for registered persons."""
        for inew in u_detections:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue
            
            # Check if track has registered_id
            if not hasattr(track, 'registered_id'):
                continue
            
            # Check if a track with this registered_id already exists
            existing_track = self._find_track_by_registered_id(track.registered_id)
            if existing_track is not None:
                # Track with this ID already exists, skip
                continue

            track.activate(self.kalman_filter, self.frame_count)
            # Override the auto-generated ID with registered ID
            track.id = track.registered_id
            activated_stracks.append(track)

    def _update_tracks(
        self,
        matches,
        strack_pool,
        detections,
        activated_stracks,
        refind_stracks,
        mark_removed=False,
    ):
        # Update or reactivate matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        # Mark only unmatched tracks as removed, if mark_removed flag is True
        if mark_removed:
            unmatched_tracks = [
                strack_pool[i]
                for i in range(len(strack_pool))
                if i not in [m[0] for m in matches]
            ]
            for track in unmatched_tracks:
                track.mark_removed()

    def _update_track_states(self, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(
        self, activated_stracks, refind_stracks, lost_stracks, removed_stracks
    ):
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            for t in self.active_tracks
            if t.is_activated
        ]

        return np.asarray(outputs)
