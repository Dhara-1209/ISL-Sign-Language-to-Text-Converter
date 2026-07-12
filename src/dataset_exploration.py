import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetExplorer:
    """
    Comprehensive dataset exploration and analysis for ISL recognition.
    
    Attributes:
        dataset_dir (Path): Path to dataset root directory
        classes (List[str]): List of class labels (0-9, A-Z)
        class_counts (Dict): Count of images per class
        image_stats (Dict): Statistics about image properties
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize DatasetExplorer.
        
        Args:
            dataset_dir (str): Path to dataset directory
            
        Raises:
            FileNotFoundError: If dataset directory doesn't exist
        """
        self.dataset_dir = Path(dataset_dir)
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        self.classes = sorted([d.name for d in self.dataset_dir.iterdir() if d.is_dir()])
        self.class_counts = {}
        self.image_stats = {
            'heights': [],
            'widths': [],
            'channels': [],
            'file_sizes': [],
            'corrupted': []
        }
        
        logger.info(f"Initialized DatasetExplorer with {len(self.classes)} classes")
    
    def load_dataset_info(self) -> Dict:
        """
        Load and analyze dataset structure.
        
        Returns:
            Dict: Contains class counts, total images, and dataset summary
        """
        logger.info("Loading dataset information...")
        
        total_images = 0
        for class_label in tqdm(self.classes, desc="Analyzing classes"):
            class_dir = self.dataset_dir / class_label
            image_count = len(list(class_dir.glob('*')))
            self.class_counts[class_label] = image_count
            total_images += image_count
        
        dataset_info = {
            'total_classes': len(self.classes),
            'total_images': total_images,
            'class_counts': self.class_counts,
            'avg_images_per_class': total_images / len(self.classes),
            'min_images': min(self.class_counts.values()),
            'max_images': max(self.class_counts.values())
        }
        
        logger.info(f"Dataset loaded: {total_images} images across {len(self.classes)} classes")
        return dataset_info
    
    def inspect_sample_images(self, samples_per_class: int = 5) -> Dict:
        """
        Inspect properties of sample images from each class.
        
        Args:
            samples_per_class (int): Number of samples to inspect per class
            
        Returns:
            Dict: Image properties including dimensions and channels
        """
        logger.info(f"Inspecting {samples_per_class} sample images per class...")
        
        sample_info = {
            'by_class': defaultdict(list),
            'dimensions': {'heights': [], 'widths': []},
            'channels': [],
            'valid_count': 0,
            'corrupted_count': 0
        }
        
        for class_label in tqdm(self.classes, desc="Inspecting samples"):
            class_dir = self.dataset_dir / class_label
            image_files = list(class_dir.glob('*'))[:samples_per_class]
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    
                    if img is None:
                        sample_info['corrupted_count'] += 1
                        self.image_stats['corrupted'].append(str(img_file))
                        continue
                    
                    height, width = img.shape[:2]
                    channels = img.shape[2] if len(img.shape) > 2 else 1
                    file_size = os.path.getsize(img_file) / 1024  # KB
                    
                    sample_info['by_class'][class_label].append({
                        'filename': img_file.name,
                        'height': height,
                        'width': width,
                        'channels': channels,
                        'file_size_kb': file_size
                    })
                    
                    sample_info['dimensions']['heights'].append(height)
                    sample_info['dimensions']['widths'].append(width)
                    sample_info['channels'].append(channels)
                    self.image_stats['heights'].append(height)
                    self.image_stats['widths'].append(width)
                    self.image_stats['channels'].append(channels)
                    self.image_stats['file_sizes'].append(file_size)
                    sample_info['valid_count'] += 1
                    
                except Exception as e:
                    logger.warning(f"Error reading {img_file}: {e}")
                    sample_info['corrupted_count'] += 1
        
        logger.info(f"Inspected {sample_info['valid_count']} valid images, "
                   f"{sample_info['corrupted_count']} corrupted images")
        return sample_info
    
    def check_data_quality(self, sample_per_class: int = 10) -> Dict:
        """
        Perform data quality checks using sampling (for speed on large datasets).
        
        Args:
            sample_per_class (int): Number of images to sample per class for checking.
                                   Defaults to 10 for faster execution on large datasets.
        
        Returns:
            Dict: Quality report including corrupted files and anomalies
        """
        logger.info(f"Performing data quality checks (sampling {sample_per_class} per class)...")
        
        quality_report = {
            'total_checked': 0,
            'valid_files': 0,
            'corrupted_files': [],
            'inconsistent_dimensions': [],
            'anomalies': []
        }
        
        expected_height = None
        expected_width = None
        
        for class_label in tqdm(self.classes, desc="Quality check"):
            class_dir = self.dataset_dir / class_label
            img_files = list(class_dir.iterdir())[:sample_per_class]
            
            for img_file in img_files:
                quality_report['total_checked'] += 1
                
                try:
                    img = cv2.imread(str(img_file))
                    
                    if img is None:
                        quality_report['corrupted_files'].append(str(img_file))
                        continue
                    
                    height, width = img.shape[:2]
                    
                    # Set expected dimensions from first valid image
                    if expected_height is None:
                        expected_height = height
                        expected_width = width
                    
                    # Check for dimension consistency
                    if height != expected_height or width != expected_width:
                        quality_report['inconsistent_dimensions'].append({
                            'file': str(img_file),
                            'expected': f"{expected_height}x{expected_width}",
                            'actual': f"{height}x{width}"
                        })
                    
                    quality_report['valid_files'] += 1
                    
                except Exception as e:
                    quality_report['anomalies'].append({
                        'file': str(img_file),
                        'error': str(e)
                    })
        
        logger.info(f"Quality check complete: {quality_report['valid_files']} valid files, "
                   f"{len(quality_report['corrupted_files'])} corrupted")
        return quality_report
    
    def create_class_distribution_plot(self, output_path: Optional[str] = None) -> None:
        """
        Create and save class distribution histogram.
        
        Args:
            output_path (str, optional): Path to save the plot. 
                                        Defaults to assets/class_distribution.png
        """
        if not self.class_counts:
            logger.warning("No class counts available. Run load_dataset_info first.")
            return
        
        if output_path is None:
            output_path = Path('assets') / 'class_distribution.png'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating class distribution plot...")
        
        # Create figure
        plt.figure(figsize=(16, 6))
        
        # Bar plot
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        plt.bar(classes, counts, color=colors, edgecolor='black', alpha=0.7)
        
        plt.xlabel('Class Label', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
        plt.title('ISL Dataset - Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=0, fontsize=10)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (cls, count) in enumerate(zip(classes, counts)):
            plt.text(i, count + 10, str(count), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {output_path}")
        plt.close()
    
    def create_sample_grid(self, samples_per_class: int = 5, 
                          output_path: Optional[str] = None) -> None:
        """
        Create and save a grid of sample images from each class.
        
        Args:
            samples_per_class (int): Number of samples per class to display
            output_path (str, optional): Path to save the plot.
                                        Defaults to assets/sample_grid.png
        """
        if output_path is None:
            output_path = Path('assets') / 'sample_grid.png'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating sample grid with {samples_per_class} samples per class...")
        
        # Create grid figure
        num_classes = len(self.classes)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=(samples_per_class * 2, num_classes * 2))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        elif samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for row, class_label in enumerate(tqdm(self.classes, desc="Loading samples")):
            class_dir = self.dataset_dir / class_label
            image_files = sorted(list(class_dir.glob('*')))[:samples_per_class]
            
            for col, img_file in enumerate(image_files):
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[row, col].imshow(img_rgb)
                    else:
                        axes[row, col].text(0.5, 0.5, 'Corrupted', 
                                          ha='center', va='center', color='red')
                except Exception as e:
                    axes[row, col].text(0.5, 0.5, f'Error', 
                                      ha='center', va='center', color='red')
                
                axes[row, col].set_title(f'{class_label}', fontweight='bold')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Sample grid saved to {output_path}")
        plt.close()
    
    def create_image_statistics_plot(self, output_path: Optional[str] = None) -> None:
        """
        Create plots for image dimension and channel statistics.
        
        Args:
            output_path (str, optional): Path to save the plot.
                                        Defaults to assets/image_statistics.png
        """
        if not self.image_stats['heights']:
            logger.warning("No image statistics available. Run inspect_sample_images first.")
            return
        
        if output_path is None:
            output_path = Path('assets') / 'image_statistics.png'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating image statistics plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Height distribution
        axes[0, 0].hist(self.image_stats['heights'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Height (pixels)', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Image Height Distribution', fontweight='bold')
        axes[0, 0].grid(alpha=0.3, linestyle='--')
        
        # Width distribution
        axes[0, 1].hist(self.image_stats['widths'], bins=20, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Width (pixels)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Image Width Distribution', fontweight='bold')
        axes[0, 1].grid(alpha=0.3, linestyle='--')
        
        # Channel distribution
        unique_channels, channel_counts = np.unique(self.image_stats['channels'], 
                                                    return_counts=True)
        axes[1, 0].bar(unique_channels, channel_counts, color='coral', edgecolor='black')
        axes[1, 0].set_xlabel('Number of Channels', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Image Channel Distribution', fontweight='bold')
        axes[1, 0].set_xticks(unique_channels)
        axes[1, 0].grid(alpha=0.3, linestyle='--')
        
        # File size distribution
        axes[1, 1].hist(self.image_stats['file_sizes'], bins=20, color='plum', edgecolor='black')
        axes[1, 1].set_xlabel('File Size (KB)', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Image File Size Distribution', fontweight='bold')
        axes[1, 1].grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Image statistics plot saved to {output_path}")
        plt.close()
    
    def generate_eda_report(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive EDA report as CSV.
        
        Args:
            output_path (str, optional): Path to save the report.
                                        Defaults to data/eda_report.csv
        
        Returns:
            pd.DataFrame: EDA report summary
        """
        if output_path is None:
            output_path = Path('data') / 'eda_report.csv'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating EDA report...")
        
        report_data = {
            'Class': [],
            'Image_Count': [],
            'Percentage': [],
            'Avg_Height_px': [],
            'Avg_Width_px': [],
            'Avg_Channels': [],
            'Avg_FileSize_KB': []
        }
        
        total_images = sum(self.class_counts.values())
        
        for class_label in self.classes:
            report_data['Class'].append(class_label)
            count = self.class_counts.get(class_label, 0)
            report_data['Image_Count'].append(count)
            report_data['Percentage'].append(f"{(count/total_images)*100:.2f}%")
            report_data['Avg_Height_px'].append(np.mean(self.image_stats['heights']) 
                                               if self.image_stats['heights'] else 0)
            report_data['Avg_Width_px'].append(np.mean(self.image_stats['widths']) 
                                              if self.image_stats['widths'] else 0)
            report_data['Avg_Channels'].append(np.mean(self.image_stats['channels']) 
                                              if self.image_stats['channels'] else 0)
            report_data['Avg_FileSize_KB'].append(np.mean(self.image_stats['file_sizes']) 
                                                 if self.image_stats['file_sizes'] else 0)
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_path, index=False)
        logger.info(f"EDA report saved to {output_path}")
        
        return report_df
    
    def print_summary(self) -> None:
        """Print summary statistics to console."""
        logger.info("\n" + "="*60)
        logger.info("DATASET EXPLORATION SUMMARY")
        logger.info("="*60)
        
        if self.class_counts:
            total = sum(self.class_counts.values())
            logger.info(f"Total Classes: {len(self.classes)}")
            logger.info(f"Total Images: {total}")
            logger.info(f"Avg Images per Class: {total/len(self.classes):.0f}")
            logger.info(f"Min Images in a Class: {min(self.class_counts.values())}")
            logger.info(f"Max Images in a Class: {max(self.class_counts.values())}")
        
        if self.image_stats['heights']:
            logger.info(f"\nImage Dimensions (from samples):")
            logger.info(f"  Height: {np.mean(self.image_stats['heights']):.0f} ± "
                       f"{np.std(self.image_stats['heights']):.0f} pixels")
            logger.info(f"  Width: {np.mean(self.image_stats['widths']):.0f} ± "
                       f"{np.std(self.image_stats['widths']):.0f} pixels")
        
        if self.image_stats['channels']:
            logger.info(f"Image Channels: {int(np.mean(self.image_stats['channels']))} (avg)")
        
        if self.image_stats['file_sizes']:
            logger.info(f"Avg File Size: {np.mean(self.image_stats['file_sizes']):.2f} KB")
        
        logger.info("="*60 + "\n")


def main():
    """Main execution function."""
    # Set paths
    dataset_dir = Path('dataset')
    assets_dir = Path('assets')
    data_dir = Path('data')
    
    # Create assets directory if it doesn't exist
    assets_dir.mkdir(exist_ok=True)
    
    # Initialize explorer
    logger.info("Starting Dataset Exploration...")
    explorer = DatasetExplorer(str(dataset_dir))
    
    # Run analysis
    dataset_info = explorer.load_dataset_info()
    sample_info = explorer.inspect_sample_images(samples_per_class=3)
    quality_report = explorer.check_data_quality()
    
    # Create visualizations
    explorer.create_class_distribution_plot()
    explorer.create_sample_grid(samples_per_class=3)
    explorer.create_image_statistics_plot()
    
    # Generate report
    report_df = explorer.generate_eda_report()
    
    # Print summary
    explorer.print_summary()
    
    # Print quality report
    logger.info(f"Quality Report: {quality_report['valid_files']} valid files")
    if quality_report['corrupted_files']:
        logger.warning(f"Found {len(quality_report['corrupted_files'])} corrupted files")
    if quality_report['inconsistent_dimensions']:
        logger.warning(f"Found {len(quality_report['inconsistent_dimensions'])} "
                      f"images with inconsistent dimensions")
    
    logger.info("Dataset exploration complete!")
    logger.info(f"Visualizations saved to {assets_dir}")
    logger.info(f"Report saved to {data_dir}")


if __name__ == "__main__":
    main()
