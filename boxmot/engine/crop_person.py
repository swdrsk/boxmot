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
from boxmot.utils.clustering import cluster_embeddings, get_diverse_samples


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
    num_frames: int = 300,
    num_samples: int = 10
):
    """
    Crop persons from video using high-density sampling and hierarchical clustering.
    
    Args:
        video_path: Path to input video
        num_persons: Number of persons to cluster
        yolo_model: YOLO model for person detection
        reid_model: ReID model for feature extraction
        output_folder: Output folder path
        conf_thresh: Confidence threshold for detection
        num_frames: Number of frames to sample (default increased for density)
        num_samples: Number of output images per person
    """
    LOGGER.info(f"Processing video: {video_path}")
    LOGGER.info(f"Target persons: {num_persons}, High-density sampling: {num_frames} frames, Output samples: {num_samples}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.error(f"Failed to open video: {video_path}")
        return
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LOGGER.info(f"Total frames: {total_frames}")
    
    # Sample frames (Random Uniform Sampling)
    # We want more frames to capture diverse poses
    if total_frames < num_frames:
        LOGGER.warning(f"Video has only {total_frames} frames, using all")
        frame_indices = list(range(total_frames))
    else:
        # Sort indices to read sequentially for efficiency (seek forward is better)
        frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    # Collect crops and embeddings
    crops = []
    embeddings = []
    
    valid_count = 0
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        frame_h, frame_w = frame.shape[:2]
        
        # Detect persons with stricter confidence if not provided explicit high value
        # But we use the user provided conf_thresh. If user provided 0.5, we might want to be stricter internally?
        # Let's stick to user provided threshold but add size filtering
        results = yolo_model(frame, classes=[0], conf=conf_thresh, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            continue
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Process each detection
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Constraints
            w, h = x2 - x1, y2 - y1
            
            # 1. Size Check: Ignore very small crops (likely background artifacts or far away people)
            if w < 50 or h < 100:  # Minimum pixel dimensions for a usable person crop
                continue
                
            # 2. Aspect Ratio Check: Person should be generally taller than wide, or at least not extremely wide
            if w / h > 2.0: # Too wide (e.g. lying down or misdetection)
                continue
                
            # 3. Edge check: Ignore crops that touch the image edge (often truncated)
            margin = 5
            if x1 < margin or y1 < margin or x2 > frame_w - margin or y2 > frame_h - margin:
                continue

            # Crop person
            crop = frame[y1:y2, x1:x2]
            
            # Extract ReID features
            bbox_for_reid = np.array([[0, 0, w, h]])
            # Note: get_features expects batch of bboxes/images, but here we do one by one for simplicity in loop
            # Efficiency note: batching would be faster but high-density sampling is fast enough usually
            emb = reid_model.model.get_features(bbox_for_reid, crop)
            
            crops.append(crop)
            embeddings.append(emb[0])
            valid_count += 1
            
    cap.release()
    
    if len(crops) == 0:
        LOGGER.error("No valid persons detected in sampled frames (strict filtering applied)")
        return
    
    LOGGER.info(f"Collected {len(crops)} high-quality person crops from {valid_count} detections")
    
    # Cluster embeddings
    embeddings = np.array(embeddings)
    
    # Handle edge case where we found fewer people than requested
    active_num_persons = num_persons
    if len(embeddings) < num_persons:
        LOGGER.warning(
            f"Only {len(embeddings)} crops found, less than requested {num_persons} persons. "
            f"Using {len(embeddings)} clusters instead."
        )
        active_num_persons = len(embeddings)
    
    LOGGER.info(f"Clustering into {active_num_persons} primary identities...")
    labels = cluster_embeddings(embeddings, active_num_persons)
    
    # Create output folders
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # For each cluster (person), select diverse samples
    
    for cluster_id in range(active_num_persons):
        # Find indices belonging to this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_embeddings_subset = embeddings[cluster_indices]
        
        # Select diverse samples using sub-clustering
        # We want up to num_samples samples
        n_select = min(len(cluster_indices), num_samples)
        
        selected_subset_indices = get_diverse_samples(
            cluster_embeddings_subset, 
            n_select
        )
        
        # Map back to original indices
        selected_global_indices = cluster_indices[selected_subset_indices]
        
        # Create folder
        person_folder = output_folder / str(cluster_id + 1)
        person_folder.mkdir(exist_ok=True)
        
        # Save images
        for save_idx, global_idx in enumerate(selected_global_indices):
            crop = crops[global_idx]
            crop_path = person_folder / f"crop_{save_idx}.png"
            cv2.imwrite(str(crop_path), crop)
            
        LOGGER.info(f"  Person {cluster_id + 1}: Saved {len(selected_global_indices)} diverse images")

    LOGGER.info(f"✅ Processing complete! Output saved to: {output_folder}")


def run_crop_person(
    source: str,
    num_persons: int,
    reid_model: str,
    folder: str,
    yolo_model: str = "yolo11n",
    device: str = "",
    conf: float = 0.5,
    num_frames: int = 300,
    num_samples: int = 10
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
        num_samples: Number of output images per person
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
        crop_from_video(source_path, num_persons, yolo, reid, output_folder, conf, num_frames, num_samples)
    else:
        LOGGER.error(f"Unsupported file format: {ext}")
        return
    
    LOGGER.info(f"✅ Processing complete! Output saved to: {output_folder}")
