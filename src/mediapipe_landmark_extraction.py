import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
import json

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MediaPipe
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision.core import vision_task_running_mode
    MEDIAPIPE_VERSION = "new"
except ImportError as e:
    logger.error(f"MediaPipe task API not available: {e}")
    raise ImportError("This module requires MediaPipe 0.10+ with task APIs")

import mediapipe as mp


class LandmarkExtractor:
    """
    Extract hand landmarks from images using MediaPipe Hands.
    
    MediaPipe detects 21 hand landmarks:
    - 0: Wrist
    - 1-4: Thumb (CMC, MCP, IP, TIP)
    - 5-8: Index (MCP, PIP, DIP, TIP)
    - 9-12: Middle (MCP, PIP, DIP, TIP)
    - 13-16: Ring (MCP, PIP, DIP, TIP)
    - 17-20: Pinky (MCP, PIP, DIP, TIP)
    
    Attributes:
        dataset_dir (Path): Path to dataset root directory
        output_dir (Path): Path to save extracted landmarks
        classes (List[str]): List of class labels
        hand_landmarker: MediaPipe Hand detector
        statistics (Dict): Extraction statistics
    """
    
    # Landmark names for reference
    LANDMARK_NAMES = [
        'Wrist',
        'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_TIP',
        'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP',
        'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP',
        'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_TIP',
        'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP'
    ]
    
    # Model URL for downloading
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task"
    
    def __init__(self, dataset_dir: str, output_dir: str = 'data/landmarks'):
        """
        Initialize LandmarkExtractor.
        
        Args:
            dataset_dir (str): Path to dataset directory
            output_dir (str): Path to save extracted landmarks
            
        Raises:
            FileNotFoundError: If dataset directory doesn't exist
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe HandLandmarker
        logger.info("Initializing MediaPipe HandLandmarker...")
        
        try:
            # Ensure model is available
            model_path = self._ensure_model_available()
            
            # Create base options with model path
            base_options = base_options_module.BaseOptions(
                model_asset_path=model_path
            )
            
            # Create HandLandmarker options
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("MediaPipe HandLandmarker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe HandLandmarker: {e}")
            logger.error("Please ensure the model file is available and MediaPipe is properly installed")
            raise
        
        # Get classes
        self.classes = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        
        # Initialize statistics
        self.statistics = {
            'total_images': 0,
            'successful': 0,
            'no_hand_detected': 0,
            'multiple_hands': 0,
            'processing_errors': 0,
            'by_class': {}
        }
        
        logger.info(f"Initialized LandmarkExtractor with {len(self.classes)} classes")
        logger.info(f"Output directory: {self.output_dir}")
    
    @staticmethod
    def _ensure_model_available() -> str:
        """
        Ensure the hand landmarker model is available locally.
        Downloads if necessary.
        
        Returns:
            str: Path to the model file
        """
        model_path = Path("models/hand_landmarker.task")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_path.exists():
            logger.info(f"Model found at {model_path}")
            return str(model_path)
        
        logger.info(f"Downloading MediaPipe hand landmarker model...")
        logger.info(f"This is a one-time download (~20 MB)")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(
                LandmarkExtractor.MODEL_URL,
                model_path,
                reporthook=LandmarkExtractor._download_progress
            )
            logger.info(f"Model downloaded successfully to {model_path}")
            return str(model_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    @staticmethod
    def _download_progress(block_num, block_size, total_size):
        """Report download progress."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rDownload progress: {percent:.1f}%", end="", flush=True)
    
    
    def extract_landmarks_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract landmarks from a single image using MediaPipe HandLandmarker.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Flattened array of 21 landmarks (63 values: x, y, z for each)
            None: If no hand detected or processing error
            
        Raises:
            Exception: If image cannot be read
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect hand landmarks
            detection_result = self.hand_landmarker.detect(mp_image)
            
            # Check if hands detected
            if not detection_result.hand_landmarks:
                return None  # No hand detected
            
            # If multiple hands, use the first one (already limited to 1 by options)
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Extract coordinates from landmarks
            landmarks = []
            for landmark in hand_landmarks:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def process_class(self, class_label: str) -> Dict:
        """
        Process all images in a class.
        
        Args:
            class_label (str): Class label (e.g., 'A', '0')
            
        Returns:
            Dict: Processing statistics for this class
        """
        class_dir = self.dataset_dir / class_label
        image_files = sorted(list(class_dir.glob('*')))
        
        landmarks_list = []
        class_stats = {
            'total': len(image_files),
            'successful': 0,
            'no_hand': 0,
            'errors': 0
        }
        
        for img_file in tqdm(image_files, desc=f"Processing class {class_label}",
                            leave=False):
            landmarks = self.extract_landmarks_from_image(str(img_file))
            
            if landmarks is None:
                class_stats['no_hand'] += 1
            else:
                landmarks_list.append(landmarks)
                class_stats['successful'] += 1
            
            # Handle processing errors
            if landmarks is None and img_file.exists():
                # Image was read successfully but no landmarks found
                pass
            elif landmarks is None and not img_file.exists():
                class_stats['errors'] += 1
        
        # Store statistics
        self.statistics['by_class'][class_label] = class_stats
        self.statistics['total_images'] += class_stats['total']
        self.statistics['successful'] += class_stats['successful']
        self.statistics['no_hand_detected'] += class_stats['no_hand']
        self.statistics['processing_errors'] += class_stats['errors']
        
        return landmarks_list, class_stats
    
    def process_dataset(self) -> Dict:
        """
        Process entire dataset and extract landmarks.
        
        Returns:
            Dict: Complete dataset with landmarks and labels
        """
        logger.info("Processing dataset for landmark extraction...")
        
        all_landmarks = []
        all_labels = []
        
        for class_label in tqdm(self.classes, desc="Classes"):
            landmarks_list, class_stats = self.process_class(class_label)
            
            # Add landmarks and labels
            all_landmarks.extend(landmarks_list)
            all_labels.extend([class_label] * len(landmarks_list))
            
            logger.info(f"Class {class_label}: {class_stats['successful']}/{class_stats['total']} "
                       f"({100*class_stats['successful']/class_stats['total']:.1f}%)")
        
        return {
            'landmarks': np.array(all_landmarks, dtype=np.float32),
            'labels': np.array(all_labels),
            'statistics': self.statistics
        }
    
    def save_landmarks_npz(self, data: Dict, filename: str = 'landmarks') -> Path:
        """
        Save landmarks as compressed NPZ file.
        
        Args:
            data (Dict): Dictionary with 'landmarks', 'labels', 'statistics'
            filename (str): Output filename (without extension)
            
        Returns:
            Path: Path to saved file
        """
        output_path = self.output_dir / f"{filename}.npz"
        
        logger.info(f"Saving landmarks to {output_path}...")
        
        np.savez_compressed(
            output_path,
            landmarks=data['landmarks'],
            labels=data['labels'],
            landmark_names=self.LANDMARK_NAMES,
            statistics=json.dumps(data['statistics'])  # JSON serialize stats
        )
        
        logger.info(f"Saved {len(data['landmarks'])} landmark vectors to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return output_path
    
    def save_landmarks_csv(self, data: Dict, filename: str = 'landmarks') -> Path:
        """
        Save landmarks as CSV file (for debugging and inspection).
        
        Args:
            data (Dict): Dictionary with 'landmarks', 'labels', 'statistics'
            filename (str): Output filename (without extension)
            
        Returns:
            Path: Path to saved file
        """
        output_path = self.output_dir / f"{filename}.csv"
        
        logger.info(f"Saving landmarks to CSV {output_path}...")
        
        # Create column names
        columns = []
        for i, name in enumerate(self.LANDMARK_NAMES):
            columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        columns.append('label')
        
        # Create DataFrame
        data_dict = {}
        for i, col in enumerate(columns[:-1]):
            data_dict[col] = data['landmarks'][:, i]
        data_dict['label'] = data['labels']
        
        df = pd.DataFrame(data_dict)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} landmark vectors to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return output_path
    
    def save_landmarks_by_class(self, data: Dict) -> Dict[str, Path]:
        """
        Save landmarks as separate CSV files per class.
        
        Args:
            data (Dict): Dictionary with 'landmarks', 'labels', 'statistics'
            
        Returns:
            Dict: Mapping of class label to saved file path
        """
        logger.info("Saving landmarks by class...")
        
        saved_files = {}
        
        # Create column names
        columns = []
        for i, name in enumerate(self.LANDMARK_NAMES):
            columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        
        # Group by class
        unique_labels = np.unique(data['labels'])
        
        for label in tqdm(unique_labels, desc="Saving by class"):
            mask = data['labels'] == label
            class_landmarks = data['landmarks'][mask]
            
            # Create DataFrame
            data_dict = {}
            for i, col in enumerate(columns):
                data_dict[col] = class_landmarks[:, i]
            
            df = pd.DataFrame(data_dict)
            
            # Save to file
            output_path = self.output_dir / f"landmarks_{label}.csv"
            df.to_csv(output_path, index=False)
            saved_files[label] = output_path
            
            logger.info(f"Saved {len(df)} landmarks for class {label}")
        
        return saved_files
    
    def print_statistics(self) -> None:
        """Print extraction statistics to console."""
        logger.info("\n" + "="*70)
        logger.info("LANDMARK EXTRACTION STATISTICS")
        logger.info("="*70)
        
        total = self.statistics['total_images']
        successful = self.statistics['successful']
        no_hand = self.statistics['no_hand_detected']
        errors = self.statistics['processing_errors']
        
        logger.info(f"Total Images Processed: {total}")
        logger.info(f"Successfully Extracted: {successful} ({100*successful/total:.2f}%)")
        logger.info(f"No Hand Detected: {no_hand} ({100*no_hand/total:.2f}%)")
        logger.info(f"Processing Errors: {errors} ({100*errors/total:.2f}%)")
        
        logger.info("\nPer-Class Statistics:")
        logger.info("-" * 70)
        
        for class_label in sorted(self.statistics['by_class'].keys()):
            stats = self.statistics['by_class'][class_label]
            success_rate = 100 * stats['successful'] / stats['total']
            logger.info(f"  {class_label}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
        
        logger.info("="*70 + "\n")
    
    def save_statistics(self, filename: str = 'extraction_statistics.json') -> Path:
        """
        Save extraction statistics to JSON file.
        
        Args:
            filename (str): Output filename
            
        Returns:
            Path: Path to saved file
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving statistics to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        logger.info(f"Statistics saved to {output_path}")
        return output_path


def create_landmark_visualization(landmarks: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Draw landmarks on image for visualization.
    
    Args:
        landmarks (np.ndarray): Array of 21 landmarks (63 values)
        image (np.ndarray): Image to draw on (BGR format)
        
    Returns:
        np.ndarray: Image with drawn landmarks
    """
    h, w = image.shape[:2]
    image_copy = image.copy()
    
    # Convert landmarks to pixel coordinates
    landmarks_px = landmarks.reshape(21, 3)
    
    # Drawing parameters
    radius = 5
    thickness = -1
    color_connections = (0, 255, 0)  # Green for connections
    color_joints = (0, 0, 255)  # Red for joints
    
    # Hand connections (from MediaPipe)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        start = (int(landmarks_px[start_idx, 0] * w), int(landmarks_px[start_idx, 1] * h))
        end = (int(landmarks_px[end_idx, 0] * w), int(landmarks_px[end_idx, 1] * h))
        cv2.line(image_copy, start, end, color_connections, 2)
    
    # Draw joints
    for i in range(21):
        x = int(landmarks_px[i, 0] * w)
        y = int(landmarks_px[i, 1] * h)
        cv2.circle(image_copy, (x, y), radius, color_joints, thickness)
    
    return image_copy


def main():
    """Main execution function."""
    import time
    
    # Set paths
    dataset_dir = Path('dataset')
    output_dir = Path('data/landmarks')
    
    logger.info("Starting MediaPipe Landmark Extraction...")
    start_time = time.time()
    
    # Initialize extractor
    extractor = LandmarkExtractor(str(dataset_dir), str(output_dir))
    
    # Process dataset
    data = extractor.process_dataset()
    
    # Save landmarks in different formats
    extractor.save_landmarks_npz(data, 'landmarks')
    extractor.save_landmarks_csv(data, 'landmarks')
    extractor.save_landmarks_by_class(data)
    
    # Save statistics
    extractor.save_statistics()
    
    # Print summary
    extractor.print_statistics()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time/60:.2f} minutes")
    logger.info("Landmark extraction complete!")
    logger.info(f"Landmarks saved to {output_dir}")


if __name__ == "__main__":
    main()
