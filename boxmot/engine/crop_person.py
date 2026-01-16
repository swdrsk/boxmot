# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Crop persons from images/videos for PreloadSORT registration."""

from pathlib import Path
import random

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.utils import logger as LOGGER
from boxmot.utils.clustering import cluster_embeddings


def crop_from_image(
    image_path: Path,
    yolo_model: YOLO,
    reid_model: ReidAutoBackend,
    output_folder: Path,
    conf_thresh: float = 0.5
):
    """
    Crop persons from a single image.
    
    Args:
        image_path: Path to input image
        yolo_model: YOLO model for person detection
        reid_model: ReID model for feature extraction
        output_folder: Output folder path
        conf_thresh: Confidence threshold for detection
    """
    LOGGER.info(f"Processing image: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        LOGGER.error(f"Failed to load image: {image_path}")
        return
    
    # Detect persons
    results = yolo_model(img, classes=[0], conf=conf_thresh, verbose=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        LOGGER.warning(f"No persons detected in {image_path}")
        return
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    LOGGER.info(f"Detected {len(boxes)} person(s)")
    
    # Create output folders for each person
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Crop person
        crop = img[y1:y2, x1:x2]
        
        # Create folder for this person
        person_folder = output_folder / str(idx + 1)
        person_folder.mkdir(exist_ok=True)
        
        # Save cropped image
        crop_path = person_folder / f"crop_0.png"
        cv2.imwrite(str(crop_path), crop)
        LOGGER.info(f"Saved crop to {crop_path}")


def crop_from_video(
    video_path: Path,
    num_persons: int,
    yolo_model: YOLO,
    reid_model: ReidAutoBackend,
    output_folder: Path,
    conf_thresh: float = 0.5,
    num_frames: int = 20
):
    """
    Crop persons from video with clustering.
    
    Args:
        video_path: Path to input video
        num_persons: Number of persons to cluster
        yolo_model: YOLO model for person detection
        reid_model: ReID model for feature extraction
        output_folder: Output folder path
        conf_thresh: Confidence threshold for detection
        num_frames: Number of frames to sample
    """
    LOGGER.info(f"Processing video: {video_path}")
    LOGGER.info(f"Target persons: {num_persons}, Sampling {num_frames} frames")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.error(f"Failed to open video: {video_path}")
        return
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LOGGER.info(f"Total frames: {total_frames}")
    
    # Sample random frames
    if total_frames < num_frames:
        LOGGER.warning(f"Video has only {total_frames} frames, using all")
        frame_indices = list(range(total_frames))
    else:
        frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    # Collect crops and embeddings
    crops = []
    embeddings = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            LOGGER.warning(f"Failed to read frame {frame_idx}")
            continue
        
        # Detect persons
        results = yolo_model(frame, classes=[0], conf=conf_thresh, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            continue
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Process each detection
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Crop person
            crop = frame[y1:y2, x1:x2]
            
            # Skip very small crops
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            
            # Extract ReID features
            bbox_for_reid = np.array([[0, 0, crop.shape[1], crop.shape[0]]])
            emb = reid_model.model.get_features(bbox_for_reid, crop)
            
            crops.append(crop)
            embeddings.append(emb[0])
    
    cap.release()
    
    if len(crops) == 0:
        LOGGER.error("No persons detected in sampled frames")
        return
    
    LOGGER.info(f"Collected {len(crops)} person crops")
    
    # Cluster embeddings
    embeddings = np.array(embeddings)
    
    if len(embeddings) < num_persons:
        LOGGER.warning(
            f"Only {len(embeddings)} crops found, less than requested {num_persons} persons. "
            f"Using {len(embeddings)} clusters instead."
        )
        num_persons = len(embeddings)
    
    LOGGER.info(f"Clustering into {num_persons} groups...")
    labels = cluster_embeddings(embeddings, num_persons)
    
    # Create output folders and save crops
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Count crops per cluster
    cluster_counts = {}
    for cluster_id in range(num_persons):
        cluster_counts[cluster_id] = 0
    
    for idx, (crop, label) in enumerate(zip(crops, labels)):
        cluster_id = int(label) + 1  # 1-indexed folders
        
        # Create folder for this cluster
        person_folder = output_folder / str(cluster_id)
        person_folder.mkdir(exist_ok=True)
        
        # Save cropped image
        crop_count = cluster_counts[label]
        crop_path = person_folder / f"crop_{crop_count}.png"
        cv2.imwrite(str(crop_path), crop)
        
        cluster_counts[label] += 1
    
    # Log results
    LOGGER.info("Clustering complete:")
    for cluster_id in range(num_persons):
        count = cluster_counts[cluster_id]
        LOGGER.info(f"  Person {cluster_id + 1}: {count} images")


def run_crop_person(
    source: str,
    num_persons: int,
    reid_model: str,
    folder: str,
    yolo_model: str = "yolo11n",
    device: str = "",
    conf: float = 0.5,
    num_frames: int = 20
):
    """
    Main function to crop persons from image or video.
    
    Args:
        source: Path to image or video file
        num_persons: Number of persons (for video only)
        reid_model: ReID model name
        folder: Output folder path
        yolo_model: YOLO model name
        device: Device to use (cuda/cpu)
        conf: Confidence threshold
        num_frames: Number of frames to sample from video
    """
    source_path = Path(source)
    output_folder = Path(folder)
    
    if not source_path.exists():
        LOGGER.error(f"Source not found: {source}")
        return
    
    # Determine device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    LOGGER.info(f"Using device: {device}")
    
    # Load YOLO model
    LOGGER.info(f"Loading YOLO model: {yolo_model}")
    yolo = YOLO(yolo_model)
    
    # Load ReID model
    LOGGER.info(f"Loading ReID model: {reid_model}")
    reid = ReidAutoBackend(
        weights=Path(reid_model + ".pt"),
        device=torch.device(device),
        half=False
    )
    
    # Determine if source is image or video
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    ext = source_path.suffix.lower()
    
    if ext in image_extensions:
        LOGGER.info("Processing as image")
        crop_from_image(source_path, yolo, reid, output_folder, conf)
    elif ext in video_extensions:
        if num_persons is None:
            LOGGER.error("Number of persons must be specified for video source")
            return
        LOGGER.info("Processing as video")
        crop_from_video(source_path, num_persons, yolo, reid, output_folder, conf, num_frames)
    else:
        LOGGER.error(f"Unsupported file format: {ext}")
        return
    
    LOGGER.info(f"✅ Processing complete! Output saved to: {output_folder}")
