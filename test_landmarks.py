"""
Quick test of landmark extraction on a small sample (1 image per class)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Use practical version
from landmark_extraction_practical import LandmarkExtractor

# Set paths
dataset_dir = Path('dataset')
output_dir = Path('data/landmarks_test')

print("Testing landmark extraction on sample images...")
print(f"Dataset: {dataset_dir}")
print(f"Output: {output_dir}\n")

# Initialize extractor
extractor = LandmarkExtractor(str(dataset_dir), str(output_dir))

# Test on just 1 image per class
print("Processing 1 image per class (36 images total)...\n")

all_landmarks = []
all_labels = []

for class_label in extractor.classes:
    class_dir = dataset_dir / class_label
    image_files = list(class_dir.glob('*'))[:1]  # Just 1 image
    
    for img_file in image_files:
        landmarks = extractor.extract_landmarks_from_image(str(img_file))
        if landmarks is not None:
            all_landmarks.append(landmarks)
            all_labels.append(class_label)
            status = "[OK]"
        else:
            status = "[ERROR]"
        print(f"{status} Class {class_label}: Landmarks extracted (63 features)")

import numpy as np
data = {
    'landmarks': np.array(all_landmarks, dtype=np.float32),
    'labels': np.array(all_labels),
    'statistics': extractor.statistics
}

print(f"\n\nExtracted {len(all_landmarks)} landmark vectors")
if len(all_landmarks) > 0:
    print(f"Shape: {data['landmarks'].shape}")
    print(f"Labels: {np.unique(data['labels'])}")
    
    # Save
    extractor.save_landmarks_npz(data, 'landmarks_sample')
    extractor.save_landmarks_csv(data, 'landmarks_sample')
    
    print("\n[SUCCESS] Test passed!")
    print(f"Sample data saved to {output_dir}")
    
    # List output files
    print("\nOutput files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
else:
    print("[FAILED] No landmarks extracted")
