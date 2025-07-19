#!/usr/bin/env python3
"""
Generate Better Quality Datasets for Rhyme Probe Experiments
Creates datasets with better rhyme separation and more diverse patterns.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from src.data.dataset_creation import RhymeDatasetCreator, DatasetSample
from src.metrics.rhyme_metrics import extract_rhyming_features

def create_high_quality_rhyming_dataset(num_samples=500):
    """Create a high-quality rhyming dataset with better patterns."""
    creator = RhymeDatasetCreator()
    samples = []
    
    # More diverse rhyme patterns with better separation
    patterns = ["AABB", "ABAB", "ABBA", "AAAA"]
    topics = ["nature", "emotions", "abstract"]
    styles = ["nature_poem", "emotional_poem", "abstract_poem"]
    
    print(f"🎯 Generating {num_samples} high-quality rhyming samples...")
    
    for i in range(num_samples):
        # Ensure AAAA pattern gets more representation
        if i < num_samples // 4:
            pattern = "AAAA"
        else:
            pattern = random.choice(patterns)
        
        topic = random.choice(topics)
        style = random.choice(styles)
        
        # Generate rhyming text
        text = creator.create_rhyming_text(pattern, style, topic)
        
        # Verify rhyming with stricter criteria
        if not creator.verify_rhyming(text, pattern):
            continue
        
        # Analyze text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        analysis = creator.rhyme_detector.analyze_rhyme_pattern(text)
        features = extract_rhyming_features(text)
        
        # Only accept samples with good rhyme density
        if features['rhyme_density'] < 0.02:  # Minimum rhyme density
            continue
        
        sample = DatasetSample(
            text=text,
            lines=lines,
            rhyme_pattern=pattern,
            rhyme_type=analysis.rhyme_type,
            rhyme_density=features['rhyme_density'],
            features=features,
            metadata={
                "topic": topic,
                "style": style,
                "num_lines": len(lines),
                "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
                "generation_id": i
            }
        )
        
        samples.append(sample)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {len(samples)} valid samples...")
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/high_quality_rhyme_dataset_{timestamp}.json"
    Path("data").mkdir(exist_ok=True)
    
    data = []
    for sample in samples:
        data.append({
            "text": sample.text,
            "lines": sample.lines,
            "rhyme_pattern": sample.rhyme_pattern,
            "rhyme_type": sample.rhyme_type,
            "rhyme_density": sample.rhyme_density,
            "features": sample.features,
            "metadata": sample.metadata
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ High-quality rhyming dataset created: {output_path}")
    print(f"   - Samples: {len(samples)}")
    print(f"   - Patterns: {set(s.rhyme_pattern for s in samples)}")
    print(f"   - Avg rhyme density: {sum(s.rhyme_density for s in samples) / len(samples):.4f}")
    
    return output_path

def create_high_quality_non_rhyming_dataset(num_samples=500):
    """Create a high-quality non-rhyming dataset with minimal rhymes."""
    creator = RhymeDatasetCreator()
    samples = []
    
    topics = ["nature", "emotions", "abstract"]
    styles = ["nature_poem", "emotional_poem", "abstract_poem"]
    
    print(f"🎯 Generating {num_samples} high-quality non-rhyming samples...")
    
    for i in range(num_samples):
        topic = random.choice(topics)
        style = random.choice(styles)
        
        # Generate non-rhyming text
        text = creator.create_non_rhyming_text(style, topic)
        
        # Verify no significant rhymes with stricter criteria
        if not creator.verify_non_rhyming(text):
            continue
        
        # Analyze text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        analysis = creator.rhyme_detector.analyze_rhyme_pattern(text)
        features = extract_rhyming_features(text)
        
        # Only accept samples with very low rhyme density
        if features['rhyme_density'] > 0.005:  # Maximum rhyme density for non-rhyming
            continue
        
        sample = DatasetSample(
            text=text,
            lines=lines,
            rhyme_pattern="none",
            rhyme_type=analysis.rhyme_type,
            rhyme_density=features['rhyme_density'],
            features=features,
            metadata={
                "topic": topic,
                "style": style,
                "num_lines": len(lines),
                "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
                "generation_id": i
            }
        )
        
        samples.append(sample)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {len(samples)} valid samples...")
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/high_quality_non_rhyme_dataset_{timestamp}.json"
    Path("data").mkdir(exist_ok=True)
    
    data = []
    for sample in samples:
        data.append({
            "text": sample.text,
            "lines": sample.lines,
            "rhyme_pattern": sample.rhyme_pattern,
            "rhyme_type": sample.rhyme_type,
            "rhyme_density": sample.rhyme_density,
            "features": sample.features,
            "metadata": sample.metadata
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ High-quality non-rhyming dataset created: {output_path}")
    print(f"   - Samples: {len(samples)}")
    print(f"   - Avg rhyme density: {sum(s.rhyme_density for s in samples) / len(samples):.4f}")
    
    return output_path

if __name__ == "__main__":
    print("🚀 Generating high-quality datasets for rhyme probe experiments...")
    
    # Generate datasets
    rhyme_path = create_high_quality_rhyming_dataset(200)
    non_rhyme_path = create_high_quality_non_rhyming_dataset(200)
    
    print("\n🎉 Dataset generation complete!")
    print(f"Rhyming dataset: {rhyme_path}")
    print(f"Non-rhyming dataset: {non_rhyme_path}")
    
    print("\n📊 Next steps:")
    print("1. Run: python test_dataset_quality.py")
    print("2. If quality is good, proceed to GCP deployment")
    print("3. Launch GPU instance and run experiments") 