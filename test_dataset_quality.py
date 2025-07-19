#!/usr/bin/env python3
"""
Dataset Quality Testing Script
==============================

This script tests the quality of generated datasets for rhyme probe experiments.
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.dataset_creation import analyze_dataset, RhymeDatasetCreator
from metrics.rhyme_metrics import RhymeDetector

def test_dataset_quality(dataset_path: str, dataset_type: str = "rhyming"):
    """Test the quality of a dataset."""
    print(f"🧪 Testing {dataset_type} dataset: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("❌ Dataset is empty!")
        return False
    
    print(f"📊 Dataset contains {len(data)} samples")
    
    # Basic statistics
    rhyme_densities = [sample['rhyme_density'] for sample in data]
    line_counts = [sample['metadata']['num_lines'] for sample in data]
    line_lengths = [sample['metadata']['avg_line_length'] for sample in data]
    
    print(f"📈 Statistics:")
    print(f"  - Avg rhyme density: {np.mean(rhyme_densities):.4f} ± {np.std(rhyme_densities):.4f}")
    print(f"  - Avg lines per poem: {np.mean(line_counts):.1f} ± {np.std(line_counts):.1f}")
    print(f"  - Avg line length: {np.mean(line_lengths):.1f} ± {np.std(line_lengths):.1f}")
    
    # Quality checks
    quality_issues = []
    
    # Check rhyme density
    if dataset_type == "rhyming":
        if np.mean(rhyme_densities) < 0.01:
            quality_issues.append("Rhyme density too low for rhyming dataset")
        if np.mean(rhyme_densities) > 0.1:
            quality_issues.append("Rhyme density too high (might be over-rhyming)")
    else:  # non-rhyming
        if np.mean(rhyme_densities) > 0.05:
            quality_issues.append("Rhyme density too high for non-rhyming dataset")
    
    # Check line consistency
    if np.std(line_counts) > 1.0:
        quality_issues.append("Inconsistent number of lines per poem")
    
    # Check line length consistency
    if np.std(line_lengths) > 10.0:
        quality_issues.append("Inconsistent line lengths")
    
    # Check pattern distribution
    if dataset_type == "rhyming":
        patterns = [sample['rhyme_pattern'] for sample in data]
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        print(f"📋 Pattern distribution:")
        for pattern, count in pattern_counts.items():
            percentage = (count / len(patterns)) * 100
            print(f"  - {pattern}: {count} ({percentage:.1f}%)")
        
        # Check if all patterns are represented
        expected_patterns = {"AABB", "ABAB", "ABBA", "AAAA"}
        missing_patterns = expected_patterns - set(pattern_counts.keys())
        if missing_patterns:
            quality_issues.append(f"Missing patterns: {missing_patterns}")
    
    # Check topic distribution
    topics = [sample['metadata']['topic'] for sample in data]
    topic_counts = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"🎯 Topic distribution:")
    for topic, count in topic_counts.items():
        percentage = (count / len(topics)) * 100
        print(f"  - {topic}: {count} ({percentage:.1f}%)")
    
    # Check for balance
    topic_percentages = [(count / len(topics)) * 100 for count in topic_counts.values()]
    if max(topic_percentages) - min(topic_percentages) > 20:
        quality_issues.append("Unbalanced topic distribution")
    
    # Manual verification of a few samples
    print(f"\n🔍 Manual verification of 3 random samples:")
    creator = RhymeDatasetCreator()
    
    for i in range(min(3, len(data))):
        sample = data[i]
        print(f"\nSample {i+1}:")
        print(f"Text: {sample['text']}")
        print(f"Pattern: {sample['rhyme_pattern']}")
        print(f"Topic: {sample['metadata']['topic']}")
        print(f"Style: {sample['metadata']['style']}")
        
        # Verify rhyming if it's a rhyming dataset
        if dataset_type == "rhyming":
            is_rhyming = creator.verify_rhyming(sample['text'], sample['rhyme_pattern'])
            print(f"Verification: {'✅ PASS' if is_rhyming else '❌ FAIL'}")
        else:
            is_non_rhyming = creator.verify_non_rhyming(sample['text'])
            print(f"Verification: {'✅ PASS' if is_non_rhyming else '❌ FAIL'}")
    
    # Report quality issues
    if quality_issues:
        print(f"\n⚠️  Quality issues found:")
        for issue in quality_issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✅ No quality issues found!")
        return True

def compare_datasets(rhyme_path: str, non_rhyme_path: str):
    """Compare rhyming and non-rhyming datasets."""
    print("🔍 Comparing rhyming and non-rhyming datasets...")
    
    # Load datasets
    with open(rhyme_path, 'r') as f:
        rhyme_data = json.load(f)
    
    with open(non_rhyme_path, 'r') as f:
        non_rhyme_data = json.load(f)
    
    # Extract metrics
    rhyme_densities = [sample['rhyme_density'] for sample in rhyme_data]
    non_rhyme_densities = [sample['rhyme_density'] for sample in non_rhyme_data]
    
    # Statistical comparison
    print(f"📊 Comparison:")
    print(f"  - Rhyming dataset: {np.mean(rhyme_densities):.4f} ± {np.std(rhyme_densities):.4f}")
    print(f"  - Non-rhyming dataset: {np.mean(non_rhyme_densities):.4f} ± {np.std(non_rhyme_densities):.4f}")
    print(f"  - Separation: {np.mean(rhyme_densities) - np.mean(non_rhyme_densities):.4f}")
    
    # Check if datasets are well-separated
    separation = np.mean(rhyme_densities) - np.mean(non_rhyme_densities)
    if separation < 0.01:
        print("⚠️  Warning: Datasets are not well-separated!")
        return False
    else:
        print("✅ Datasets are well-separated!")
        return True

def create_quality_report(rhyme_path: str, non_rhyme_path: str, output_dir: str = "data/analysis"):
    """Create a comprehensive quality report."""
    print("📝 Creating quality report...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Test individual datasets
    rhyme_quality = test_dataset_quality(rhyme_path, "rhyming")
    non_rhyme_quality = test_dataset_quality(non_rhyme_path, "non-rhyming")
    
    # Compare datasets
    comparison_quality = compare_datasets(rhyme_path, non_rhyme_path)
    
    # Create summary
    summary = {
        "timestamp": str(Path(rhyme_path).stem),
        "rhyme_dataset": {
            "path": rhyme_path,
            "quality_passed": rhyme_quality,
            "stats": analyze_dataset(rhyme_path)
        },
        "non_rhyme_dataset": {
            "path": non_rhyme_path,
            "quality_passed": non_rhyme_quality,
            "stats": analyze_dataset(non_rhyme_path)
        },
        "comparison": {
            "well_separated": comparison_quality
        },
        "overall_quality": rhyme_quality and non_rhyme_quality and comparison_quality
    }
    
    # Convert sets to lists for JSON serialization
    def convert_sets_to_lists(obj):
        if isinstance(obj, dict):
            return {k: convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets_to_lists(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    summary = convert_sets_to_lists(summary)
    
    # Save report
    report_path = Path(output_dir) / f"quality_report_{Path(rhyme_path).stem}.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Quality report saved to: {report_path}")
    
    # Print summary
    print(f"\n🎯 Quality Summary:")
    print(f"  - Rhyming dataset: {'✅ PASS' if rhyme_quality else '❌ FAIL'}")
    print(f"  - Non-rhyming dataset: {'✅ PASS' if non_rhyme_quality else '❌ FAIL'}")
    print(f"  - Dataset separation: {'✅ PASS' if comparison_quality else '❌ FAIL'}")
    print(f"  - Overall quality: {'✅ PASS' if summary['overall_quality'] else '❌ FAIL'}")
    
    return summary['overall_quality']

def main():
    """Main function to test dataset quality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset quality")
    parser.add_argument("--rhyme-dataset", type=str, help="Path to rhyming dataset")
    parser.add_argument("--non-rhyme-dataset", type=str, help="Path to non-rhyming dataset")
    parser.add_argument("--output-dir", type=str, default="data/analysis", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # If no datasets specified, find the most recent ones
    if not args.rhyme_dataset or not args.non_rhyme_dataset:
        data_dir = Path("data")
        if data_dir.exists():
            # Look for both regular and high-quality datasets
            rhyme_files = list(data_dir.glob("*rhyme_dataset_*.json"))
            non_rhyme_files = list(data_dir.glob("*non_rhyme_dataset_*.json"))
            
            # Filter out non-rhyming from rhyming files
            rhyme_files = [f for f in rhyme_files if "non_rhyme" not in f.name]
            
            if rhyme_files and non_rhyme_files:
                # Get most recent files
                rhyme_path = str(max(rhyme_files, key=lambda x: x.stat().st_mtime))
                non_rhyme_path = str(max(non_rhyme_files, key=lambda x: x.stat().st_mtime))
                
                print(f"🔍 Found datasets:")
                print(f"  - Rhyming: {rhyme_path}")
                print(f"  - Non-rhyming: {non_rhyme_path}")
            else:
                print("❌ No datasets found in data/ directory")
                return
        else:
            print("❌ data/ directory not found")
            return
    else:
        rhyme_path = args.rhyme_dataset
        non_rhyme_path = args.non_rhyme_dataset
    
    # Create quality report
    quality_passed = create_quality_report(rhyme_path, non_rhyme_path, args.output_dir)
    
    if quality_passed:
        print("\n🎉 All quality checks passed! Your datasets are ready for experiments.")
    else:
        print("\n⚠️  Some quality checks failed. Please review and regenerate datasets.")

if __name__ == "__main__":
    main() 