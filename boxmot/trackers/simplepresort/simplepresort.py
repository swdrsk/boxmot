# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
SimplePreSort: Simplified version of PreloadSort tracker.

This tracker performs frame-by-frame matching without Kalman filtering or
temporal tracking. It simply:
1. Detects persons (class=0) using YOLO
2. Extracts ReID features
3. Matches against pre-registered person database
4. Assigns IDs based on best match
5. Discards detections below threshold
"""

from pathlib import Path
from boxmot.utils import logger as LOGGER

import numpy as np
import torch
import cv2

from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.trackers.basetracker import BaseTracker


class SimplePreSort(BaseTracker):
    """
    SimplePreSort tracker - simplified version of PreloadSort.
    
    This tracker performs per-frame matching without temporal tracking:
    - No Kalman filter
    - No track state management
    - No camera motion compensation
    - Just ReID matching against registered database
    
    Parameters:
    - reid_weights (Path): Path to the re-identification model weights.
    - device (torch.device): Device to run the model on (e.g., 'cpu', 'cuda').
    - half (bool): Whether to use half-precision (fp16) for faster inference.
    - registered_images_path (Path): Path to folder containing registered person images.
    - match_threshold (float): Similarity threshold for matching (0.0-1.0).
    - per_class (bool): Enables class-separated tracking.
    - nr_classes (int): Total number of object classes.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        # SimplePreSort-specific parameters
        registered_images_path: Path = None,
        match_threshold: float = 0.5,
        **kwargs  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='SimplePreSort', **kwargs)
        
        self.match_threshold = match_threshold
        self.registered_ids = {}  # {id: list of embeddings}
        
        # ReID model
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model
        
        # Load registered images
        if registered_images_path:
            self._load_registered_images(registered_images_path)
        else:
            LOGGER.warning("No registered_images_path provided. SimplePreSort will not track any detections.")
    
    def _load_registered_images(self, path: Path):
        """
        Load registered person images and extract their features (Winner-take-all).
        
        Directory structure expected:
        registered_images/
            1/
                img1.png
                img2.png
            2/
                img1.png
                ...
        
        Folder names are used as person IDs.
        """
        if not path.exists():
            LOGGER.error(f"Registered images path does not exist: {path}")
            return
        
        person_dirs = sorted([d for d in path.iterdir() if d.is_dir()])
        
        if len(person_dirs) == 0:
            LOGGER.warning(f"No person directories found in {path}")
            return
        
        for idx, person_dir in enumerate(person_dirs, start=1):
            # Use folder name as ID if it's an integer, otherwise use sequential number
            try:
                person_id = int(person_dir.name)
            except ValueError:
                person_id = idx
            
            embeddings = []
            image_files = list(person_dir.glob('*.png')) + list(person_dir.glob('*.jpg'))
            
            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is None:
                    LOGGER.warning(f"Failed to load image: {img_file}")
                    continue
                
                # Treat entire image as bbox
                h, w = img.shape[:2]
                bbox = np.array([[0, 0, w, h]])
                emb = self.model.get_features(bbox, img)
                
                # Normalize embedding
                emb_normalized = emb[0] / np.linalg.norm(emb[0])
                embeddings.append(emb_normalized)
            
            if len(embeddings) == 0:
                LOGGER.warning(f"No valid images found for person {person_id} in {person_dir}")
                continue
            
            # Store all embeddings (not averaged) for Winner-take-all matching
            self.registered_ids[person_id] = embeddings
            LOGGER.info(f"Loaded {len(embeddings)} images for person ID {person_id}")
        
        LOGGER.info(f"Registered {len(self.registered_ids)} persons with {sum(len(v) for v in self.registered_ids.values())} total images")
    
    def _match_detection_with_registered(self, detection_emb):
        """
        Match a single detection embedding with registered person embeddings (Winner-take-all).
        
        Args:
            detection_emb: Normalized embedding vector for the detection
            
        Returns:
            Tuple of (best_match_id, best_similarity) or (None, -1) if no match above threshold
        """
        if detection_emb is None:
            return None, -1
        
        best_match_id = None
        best_similarity = -1
        
        for reg_id, reg_embs in self.registered_ids.items():
            # Winner-take-all: compare with all registered images and take the maximum
            for reg_emb in reg_embs:
                # Calculate cosine similarity (features are already normalized)
                similarity = np.dot(detection_emb, reg_emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = reg_id
        
        # Only return match if above threshold
        if best_similarity >= self.match_threshold:
            return best_match_id, best_similarity
        else:
            return None, best_similarity
    
    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            dets: Detections array (N x 6: x1, y1, x2, y2, conf, class)
            img: Current frame image
            embs: Optional pre-computed embeddings (N x D)
            
        Returns:
            Tracked objects array (M x 8: x1, y1, x2, y2, track_id, conf, class, det_ind)
        """
        self.check_inputs(dets, img, embs)
        self.frame_count += 1
        
        # Clear previous active tracks
        self.active_tracks = []
        
        # No detections
        if len(dets) == 0:
            return np.empty((0, 8))
        
        # Extract ReID features if not provided
        if embs is None:
            bboxes = dets[:, 0:4]
            features = self.model.get_features(bboxes, img)
        else:
            features = embs
        
        # Normalize features
        features_normalized = []
        for feat in features:
            norm = np.linalg.norm(feat)
            if norm > 0:
                features_normalized.append(feat / norm)
            else:
                features_normalized.append(feat)
        
        # Match each detection with registered database and collect all matches
        # Store as: (match_id, similarity, det_idx, det_data)
        all_matches = []
        for idx, (det, feat) in enumerate(zip(dets, features_normalized)):
            match_id, similarity = self._match_detection_with_registered(feat)
            
            if match_id is not None:
                x1, y1, x2, y2, conf, cls = det
                all_matches.append({
                    'match_id': match_id,
                    'similarity': similarity,
                    'det_idx': idx,
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'cls': cls
                })
        
        # Apply NMS per ID: keep only the detection with highest similarity for each ID
        if len(all_matches) > 0:
            # Group matches by ID
            id_groups = {}
            for match in all_matches:
                mid = match['match_id']
                if mid not in id_groups:
                    id_groups[mid] = []
                id_groups[mid].append(match)
            
            # For each ID, keep only the match with highest similarity
            outputs = []
            for match_id, matches in id_groups.items():
                # Sort by similarity (descending) and take the best one
                best_match = max(matches, key=lambda m: m['similarity'])
                
                x1, y1, x2, y2 = best_match['bbox']
                conf = best_match['conf']
                cls = best_match['cls']
                idx = best_match['det_idx']
                
                # Format: x1, y1, x2, y2, track_id, conf, class, det_ind
                outputs.append([x1, y1, x2, y2, match_id, conf, cls, idx])
                
                # Create a simple track object for visualization
                track = type('SimpleTrack', (), {
                    'id': match_id,
                    'xyxy': np.array([x1, y1, x2, y2]),
                    'conf': conf,
                    'cls': cls,
                    'history_observations': [np.array([x1, y1, x2, y2])],
                    'is_activated': True,
                })()
                self.active_tracks.append(track)
            
            return np.asarray(outputs)
        else:
            return np.empty((0, 8))


