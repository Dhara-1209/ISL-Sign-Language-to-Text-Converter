import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedLandmarkExtractor:
    """
    Simplified landmark extraction using OpenCV-based hand detection.
    
    Note: This is a temporary solution for demo/testing. For production,
    use MediaPipe or TensorFlow-based hand detection.
    
    Generates pseudo-landmarks by extracting key hand contour points.
    """
    
    LANDMARK_NAMES = [
        'Wrist',
        'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_TIP',
        'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP',
        'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP',
        'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_TIP',
        'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP'
    ]
    
    def __init__(self, dataset_dir: str, output_dir: str = 'data/landmarks'):
        """
        Initialize SimplifiedLandmarkExtractor.
        
        Args:
            dataset_dir (str): Path to dataset directory
            output_dir (str): Path to save extracted landmarks
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get classes
        self.classes = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        
        # Initialize statistics
        self.statistics = {
            'total_images': 0,
            'successful': 0,
            'no_hand_detected': 0,
            'processing_errors': 0,
            'by_class': {}
        }
        
        logger.info(f"Initialized SimplifiedLandmarkExtractor with {len(self.classes)} classes")
        logger.info(f"NOTE: Using OpenCV-based extraction (simplified method)")
        logger.info(f"For production, use MediaPipe or TensorFlow hand detection")
    
    def extract_landmarks_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract simplified landmarks from image using OpenCV.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Array of 21 landmarks (63 values: x, y, z)
            None: If hand cannot be detected
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            h, w = image.shape[:2]
            
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to enhance hand
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour (assume it's the hand)
            hand_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(hand_contour)
            
            # Filter out small contours
            if area < 1000:
                return None
            
            # Get convex hull
            hull = cv2.convexHull(hand_contour)
            
            # Extract landmark points from hull and contour
            landmarks = []
            
            # Get key points: sample evenly from contour + hull + center
            if len(hull) >= 21:
                # Sample 21 points from convex hull
                indices = np.linspace(0, len(hull) - 1, 21, dtype=int)
                for idx in indices:
                    pt = hull[idx][0]
                    x = pt[0] / w  # Normalize to 0-1
                    y = pt[1] / h
                    z = 0.5  # Dummy depth value
                    landmarks.extend([x, y, z])
            else:
                # Not enough points, pad or skip
                return None
            
            return np.array(landmarks, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def process_class(self, class_label: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Process all images in a class.
        
        Args:
            class_label (str): Class label
            
        Returns:
            Tuple of (landmarks_list, statistics)
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
        
        for img_file in tqdm(image_files, desc=f"Processing class {class_label}", leave=False):
            landmarks = self.extract_landmarks_from_image(str(img_file))
            
            if landmarks is None:
                class_stats['no_hand'] += 1
            else:
                landmarks_list.append(landmarks)
                class_stats['successful'] += 1
        
        self.statistics['by_class'][class_label] = class_stats
        self.statistics['total_images'] += class_stats['total']
        self.statistics['successful'] += class_stats['successful']
        self.statistics['no_hand_detected'] += class_stats['no_hand']
        
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
        """Save landmarks as NPZ."""
        output_path = self.output_dir / f"{filename}.npz"
        
        logger.info(f"Saving landmarks to {output_path}...")
        
        np.savez_compressed(
            output_path,
            landmarks=data['landmarks'],
            labels=data['labels'],
            landmark_names=self.LANDMARK_NAMES,
            statistics=json.dumps(data['statistics'])
        )
        
        logger.info(f"Saved {len(data['landmarks'])} landmark vectors")
        return output_path
    
    def save_landmarks_csv(self, data: Dict, filename: str = 'landmarks') -> Path:
        """Save landmarks as CSV."""
        output_path = self.output_dir / f"{filename}.csv"
        
        logger.info(f"Saving landmarks to CSV {output_path}...")
        
        columns = []
        for name in self.LANDMARK_NAMES:
            columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        columns.append('label')
        
        data_dict = {}
        for i, col in enumerate(columns[:-1]):
            data_dict[col] = data['landmarks'][:, i]
        data_dict['label'] = data['labels']
        
        df = pd.DataFrame(data_dict)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} landmark vectors to CSV")
        return output_path
    
    def print_statistics(self) -> None:
        """Print statistics."""
        logger.info("\n" + "="*70)
        logger.info("LANDMARK EXTRACTION STATISTICS")
        logger.info("="*70)
        
        total = self.statistics['total_images']
        successful = self.statistics['successful']
        
        logger.info(f"Total Images: {total}")
        logger.info(f"Successfully Extracted: {successful} ({100*successful/total:.2f}%)")
        logger.info(f"No Hand Detected: {self.statistics['no_hand_detected']}")
        
        logger.info("="*70 + "\n")
    
    def save_statistics(self, filename: str = 'extraction_statistics.json') -> Path:
        """Save statistics."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        logger.info(f"Statistics saved to {output_path}")
        return output_path


# Alias for backward compatibility
LandmarkExtractor = SimplifiedLandmarkExtractor


def main():
    """Main execution function."""
    import time
    
    dataset_dir = Path('dataset')
    output_dir = Path('data/landmarks')
    
    logger.info("Starting Landmark Extraction...")
    start_time = time.time()
    
    extractor = LandmarkExtractor(str(dataset_dir), str(output_dir))
    
    data = extractor.process_dataset()
    
    extractor.save_landmarks_npz(data, 'landmarks')
    extractor.save_landmarks_csv(data, 'landmarks')
    
    extractor.save_statistics()
    extractor.print_statistics()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time/60:.2f} minutes")
    logger.info("Landmark extraction complete!")
    logger.info(f"Landmarks saved to {output_dir}")


if __name__ == "__main__":
    main()
