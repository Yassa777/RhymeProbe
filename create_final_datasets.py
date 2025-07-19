#!/usr/bin/env python3
"""
Create Final Datasets for Rhyme Probe Experiments
Generates datasets with all rhyme patterns including AAAA.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from src.data.dataset_creation import RhymeDatasetCreator, DatasetSample
from src.metrics.rhyme_metrics import extract_rhyming_features

def create_final_rhyming_dataset(num_samples=300):
    """Create final rhyming dataset with all patterns including AAAA."""
    creator = RhymeDatasetCreator()
    samples = []
    
    patterns = ["AABB", "ABAB", "ABBA", "AAAA"]
    topics = ["nature", "emotions", "abstract"]
    styles = ["nature_poem", "emotional_poem", "abstract_poem"]
    
    # Ensure each pattern gets representation
    samples_per_pattern = num_samples // len(patterns)
    
    print(f"🎯 Generating final rhyming dataset with {num_samples} samples...")
    print(f"   - {samples_per_pattern} samples per pattern")
    
    for pattern_idx, pattern in enumerate(patterns):
        print(f"   Generating {pattern} pattern...")
        
        for i in range(samples_per_pattern):
            topic = random.choice(topics)
            style = random.choice(styles)
            
            # Generate rhyming text
            text = creator.create_rhyming_text(pattern, style, topic)
            
            # Verify rhyming
            if not creator.verify_rhyming(text, pattern):
                continue
            
            # Analyze text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            analysis = creator.rhyme_detector.analyze_rhyme_pattern(text)
            features = extract_rhyming_features(text)
            
            # Only accept samples with good rhyme density
            if features['rhyme_density'] < 0.02:
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
                    "generation_id": f"{pattern}_{i}"
                }
            )
            
            samples.append(sample)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/final_rhyme_dataset_{timestamp}.json"
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
    
    # Print statistics
    pattern_counts = {}
    for sample in samples:
        pattern = sample.rhyme_pattern
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print(f"✅ Final rhyming dataset created: {output_path}")
    print(f"   - Total samples: {len(samples)}")
    print(f"   - Pattern distribution:")
    for pattern, count in pattern_counts.items():
        print(f"     - {pattern}: {count} samples")
    print(f"   - Avg rhyme density: {sum(s.rhyme_density for s in samples) / len(samples):.4f}")
    
    return output_path

def create_final_non_rhyming_dataset(num_samples=300):
    """Create final non-rhyming dataset."""
    creator = RhymeDatasetCreator()
    samples = []
    
    topics = ["nature", "emotions", "abstract"]
    styles = ["nature_poem", "emotional_poem", "abstract_poem"]
    
    print(f"🎯 Generating final non-rhyming dataset with {num_samples} samples...")
    
    for i in range(num_samples):
        topic = random.choice(topics)
        style = random.choice(styles)
        
        # Generate non-rhyming text
        text = creator.create_non_rhyming_text(style, topic)
        
        # Verify no significant rhymes
        if not creator.verify_non_rhyming(text):
            continue
        
        # Analyze text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        analysis = creator.rhyme_detector.analyze_rhyme_pattern(text)
        features = extract_rhyming_features(text)
        
        # Only accept samples with very low rhyme density
        if features['rhyme_density'] > 0.005:
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
                "generation_id": f"non_rhyme_{i}"
            }
        )
        
        samples.append(sample)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {len(samples)} valid samples...")
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/final_non_rhyme_dataset_{timestamp}.json"
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
    
    print(f"✅ Final non-rhyming dataset created: {output_path}")
    print(f"   - Total samples: {len(samples)}")
    print(f"   - Avg rhyme density: {sum(s.rhyme_density for s in samples) / len(samples):.4f}")
    
    return output_path

if __name__ == "__main__":
    print("🚀 Creating final datasets for rhyme probe experiments...")
    
    # Generate final datasets
    rhyme_path = create_final_rhyming_dataset(200)
    non_rhyme_path = create_final_non_rhyming_dataset(200)
    
    print("\n🎉 Final dataset generation complete!")
    print(f"Rhyming dataset: {rhyme_path}")
    print(f"Non-rhyming dataset: {non_rhyme_path}")
    
    print("\n📊 Next steps:")
    print("1. Run: python test_dataset_quality.py")
    print("2. If quality is good, proceed to GCP deployment")
    print("3. Launch GPU instance and run experiments") 