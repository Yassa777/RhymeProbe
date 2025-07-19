"""
Dataset Creation for Rhyme Probe Experiments
============================================

This module creates comprehensive datasets for rhyme probe experiments,
including rhyming and non-rhyming text generation with various strategies.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics.rhyme_metrics import RhymeDetector, extract_rhyming_features

@dataclass
class DatasetSample:
    """Single sample in the dataset."""
    text: str
    lines: List[str]
    rhyme_pattern: str
    rhyme_type: str
    rhyme_density: float
    features: Dict[str, float]
    metadata: Dict[str, any]

class RhymeDatasetCreator:
    """Creates rhyming datasets with various patterns and types."""
    
    def __init__(self):
        self.rhyme_detector = RhymeDetector()
        
        # Common rhyming word pairs
        self.rhyme_pairs = {
            "nature": [
                ("tree", "free"), ("sky", "high"), ("moon", "soon"),
                ("star", "far"), ("light", "bright"), ("night", "sight"),
                ("wind", "find"), ("rain", "pain"), ("sun", "run"),
                ("sea", "free"), ("wave", "save"), ("bird", "word")
            ],
            "emotions": [
                ("love", "dove"), ("heart", "start"), ("soul", "whole"),
                ("tears", "fears"), ("joy", "boy"), ("pain", "rain"),
                ("hope", "rope"), ("dream", "seem"), ("feel", "real"),
                ("care", "share"), ("mind", "find"), ("life", "strife")
            ],
            "abstract": [
                ("time", "rhyme"), ("space", "place"), ("world", "unfurled"),
                ("thought", "bought"), ("word", "heard"), ("way", "day"),
                ("end", "friend"), ("start", "heart"), ("path", "bath"),
                ("light", "sight"), ("dark", "mark"), ("true", "blue")
            ]
        }
        
        # Rhyme patterns
        self.rhyme_patterns = {
            "AABB": [(0, 1), (2, 3)],  # First two lines rhyme, next two rhyme
            "ABAB": [(0, 2), (1, 3)],  # Alternating rhyme
            "ABBA": [(0, 3), (1, 2)],  # Envelope rhyme
            "AAAA": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # All rhyme
        }
        
        # Templates for different styles
        self.templates = {
            "nature_poem": [
                "The {noun1} stands tall and {adj1}",
                "While {noun2} {verb1} in the {noun3}",
                "The {noun4} {verb2} with {adj2} {noun5}",
                "As {noun6} {verb3} through the {noun7}"
            ],
            "emotional_poem": [
                "My {noun1} is filled with {adj1} {noun2}",
                "As {noun3} {verb1} through my {noun4}",
                "The {noun5} of {adj2} {noun6}",
                "Makes my {noun7} {verb2} with {adj3} {noun8}"
            ],
            "abstract_poem": [
                "In the {noun1} of {adj1} {noun2}",
                "Where {noun3} {verb1} and {noun4} {verb2}",
                "The {noun5} of {adj2} {noun6}",
                "Brings {adj3} {noun7} to {noun8}"
            ]
        }
        
        # Vocabulary for generation
        self.vocabulary = {
            "nouns": [
                "tree", "bird", "wind", "sky", "moon", "star", "sun", "sea",
                "heart", "soul", "mind", "love", "hope", "dream", "life", "time",
                "world", "space", "thought", "word", "way", "path", "light", "dark"
            ],
            "verbs": [
                "dances", "flows", "shines", "glows", "sings", "rings", "brings",
                "flies", "cries", "dies", "lives", "gives", "takes", "makes",
                "breaks", "wakes", "sleeps", "weeps", "leaps", "creeps"
            ],
            "adjectives": [
                "bright", "light", "dark", "deep", "high", "low", "slow", "fast",
                "soft", "hard", "warm", "cold", "old", "new", "true", "blue",
                "green", "clean", "mean", "keen", "seen", "been"
            ]
        }
    
    def create_rhyming_text(self, pattern: str, style: str, topic: str) -> str:
        """Create rhyming text with specified pattern and style."""
        if pattern not in self.rhyme_patterns:
            raise ValueError(f"Unknown rhyme pattern: {pattern}")
        
        # Get rhyming pairs for the topic
        topic_pairs = self.rhyme_pairs.get(topic, self.rhyme_pairs["nature"])
        
        # Get template for the style
        template = self.templates.get(style, self.templates["nature_poem"])
        
        # Generate lines with proper rhyming
        lines = []
        used_rhymes = {}  # Dictionary to track which rhyme word is used for each line
        rhyme_groups = {}  # Track which lines rhyme together
        
        # First, determine rhyme groups based on pattern
        for pair in self.rhyme_patterns[pattern]:
            line1, line2 = pair
            if line1 not in rhyme_groups:
                rhyme_groups[line1] = []
            if line2 not in rhyme_groups:
                rhyme_groups[line2] = []
            rhyme_groups[line1].append(line2)
            rhyme_groups[line2].append(line1)
        
        for i, line_template in enumerate(template):
            # Check if this line should rhyme with a previous line
            should_rhyme_with = None
            for j in range(i):
                if j in rhyme_groups.get(i, []):
                    should_rhyme_with = j
                    break
            
            if should_rhyme_with is not None and should_rhyme_with in used_rhymes:
                # Use a different word that rhymes with the previous line
                previous_word = used_rhymes[should_rhyme_with]
                # Find a rhyming pair that contains the previous word
                for pair in topic_pairs:
                    if pair[0] == previous_word:
                        rhyme_word = pair[1]  # Use the second word in the pair
                        break
                    elif pair[1] == previous_word:
                        rhyme_word = pair[0]  # Use the first word in the pair
                        break
                else:
                    # Fallback: choose a new rhyming pair
                    rhyme_pair = random.choice(topic_pairs)
                    rhyme_word = rhyme_pair[0]
            else:
                # Choose new rhyming pair
                rhyme_pair = random.choice(topic_pairs)
                rhyme_word = rhyme_pair[0]
                used_rhymes[i] = rhyme_word
            
            # Fill template with rhyming word at the end
            filled_line = self._fill_template(line_template, rhyme_word)
            lines.append(filled_line)
        
        return "\n".join(lines)
    
    def _fill_template(self, template: str, end_word: str) -> str:
        """Fill a template with appropriate words, ending with the specified word."""
        # Extract placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Fill placeholders
        filled = template
        for placeholder in placeholders:
            if placeholder == "noun1":
                word = random.choice(self.vocabulary["nouns"])
            elif placeholder == "verb1":
                word = random.choice(self.vocabulary["verbs"])
            elif placeholder == "adj1":
                word = random.choice(self.vocabulary["adjectives"])
            else:
                # Generic noun/verb/adjective
                if placeholder.startswith("noun"):
                    word = random.choice(self.vocabulary["nouns"])
                elif placeholder.startswith("verb"):
                    word = random.choice(self.vocabulary["verbs"])
                elif placeholder.startswith("adj"):
                    word = random.choice(self.vocabulary["adjectives"])
                else:
                    word = random.choice(self.vocabulary["nouns"])
            
            filled = filled.replace(f"{{{placeholder}}}", word)
        
        # Ensure the line ends with the rhyming word
        if not filled.endswith(end_word):
            # Replace the last word with the rhyming word
            words = filled.split()
            if words:
                words[-1] = end_word
                filled = " ".join(words)
        
        return filled
    
    def create_non_rhyming_text(self, style: str, topic: str) -> str:
        """Create non-rhyming text with the same style."""
        template = self.templates.get(style, self.templates["nature_poem"])
        
        lines = []
        for line_template in template:
            # Fill template without considering rhymes
            filled_line = self._fill_template_random(line_template)
            lines.append(filled_line)
        
        return "\n".join(lines)
    
    def _fill_template_random(self, template: str) -> str:
        """Fill template with random words without rhyme consideration."""
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        filled = template
        for placeholder in placeholders:
            if placeholder.startswith("noun"):
                word = random.choice(self.vocabulary["nouns"])
            elif placeholder.startswith("verb"):
                word = random.choice(self.vocabulary["verbs"])
            elif placeholder.startswith("adj"):
                word = random.choice(self.vocabulary["adjectives"])
            else:
                word = random.choice(self.vocabulary["nouns"])
            
            filled = filled.replace(f"{{{placeholder}}}", word)
        
        return filled
    
    def verify_rhyming(self, text: str, expected_pattern: str) -> bool:
        """Verify that text follows the expected rhyme pattern."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return False
        
        # Get last words of each line
        last_words = []
        for line in lines:
            words = line.split()
            if words:
                last_words.append(words[-1].lower().strip(".,!?;:"))
        
        # Check rhyme pattern
        expected_rhymes = self.rhyme_patterns[expected_pattern]
        
        for line1_idx, line2_idx in expected_rhymes:
            if line1_idx < len(last_words) and line2_idx < len(last_words):
                word1 = last_words[line1_idx]
                word2 = last_words[line2_idx]
                
                rhyme_type, confidence = self.rhyme_detector.detect_rhyme_type(word1, word2)
                if confidence < 0.8:  # High threshold for verification
                    return False
        
        return True
    
    def verify_non_rhyming(self, text: str) -> bool:
        """Verify that text has no significant rhymes."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return True
        
        # Check consecutive lines for rhymes
        for i in range(len(lines) - 1):
            words1 = lines[i].split()
            words2 = lines[i + 1].split()
            
            if words1 and words2:
                word1 = words1[-1].lower().strip(".,!?;:")
                word2 = words2[-1].lower().strip(".,!?;:")
                
                rhyme_type, confidence = self.rhyme_detector.detect_rhyme_type(word1, word2)
                if confidence > 0.5:  # Medium threshold for non-rhyming
                    return False
        
        return True

def create_rhyme_dataset(
    num_samples: int = 1000,
    min_length: int = 10,
    max_length: int = 100,
    rhyme_patterns: List[str] = ["AABB", "ABAB", "ABBA", "AAAA"],
    rhyme_types: List[str] = ["perfect", "slant", "assonance"]
) -> str:
    """Create a rhyming dataset."""
    creator = RhymeDatasetCreator()
    samples = []
    
    topics = ["nature", "emotions", "abstract"]
    styles = ["nature_poem", "emotional_poem", "abstract_poem"]
    
    for i in range(num_samples):
        # Randomly select parameters
        pattern = random.choice(rhyme_patterns)
        topic = random.choice(topics)
        style = random.choice(styles)
        
        # Generate rhyming text
        text = creator.create_rhyming_text(pattern, style, topic)
        
        # Verify rhyming
        if not creator.verify_rhyming(text, pattern):
            continue  # Skip if verification fails
        
        # Analyze text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        analysis = creator.rhyme_detector.analyze_rhyme_pattern(text)
        features = extract_rhyming_features(text)
        
        # Create sample
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
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} rhyming samples")
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/rhyme_dataset_{timestamp}.json"
    Path("data").mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
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
    
    print(f"✅ Rhyming dataset created: {output_path}")
    print(f"   - Samples: {len(samples)}")
    print(f"   - Patterns: {set(s.rhyme_pattern for s in samples)}")
    print(f"   - Topics: {set(s.metadata['topic'] for s in samples)}")
    
    return output_path

def create_non_rhyme_dataset(
    num_samples: int = 1000,
    min_length: int = 10,
    max_length: int = 100,
    ensure_no_rhymes: bool = True
) -> str:
    """Create a non-rhyming dataset."""
    creator = RhymeDatasetCreator()
    samples = []
    
    topics = ["nature", "emotions", "abstract"]
    styles = ["nature_poem", "emotional_poem", "abstract_poem"]
    
    for i in range(num_samples):
        # Randomly select parameters
        topic = random.choice(topics)
        style = random.choice(styles)
        
        # Generate non-rhyming text
        text = creator.create_non_rhyming_text(style, topic)
        
        # Verify no significant rhymes
        if ensure_no_rhymes and not creator.verify_non_rhyming(text):
            continue  # Skip if verification fails
        
        # Analyze text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        analysis = creator.rhyme_detector.analyze_rhyme_pattern(text)
        features = extract_rhyming_features(text)
        
        # Create sample
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
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} non-rhyming samples")
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/non_rhyme_dataset_{timestamp}.json"
    Path("data").mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
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
    
    print(f"✅ Non-rhyming dataset created: {output_path}")
    print(f"   - Samples: {len(samples)}")
    print(f"   - Topics: {set(s.metadata['topic'] for s in samples)}")
    print(f"   - Avg rhyme density: {np.mean([s.rhyme_density for s in samples]):.3f}")
    
    return output_path

def analyze_dataset(dataset_path: str) -> Dict:
    """Analyze a dataset and return statistics."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = [DatasetSample(**sample) for sample in data]
    
    stats = {
        "total_samples": len(samples),
        "avg_rhyme_density": np.mean([s.rhyme_density for s in samples]),
        "std_rhyme_density": np.std([s.rhyme_density for s in samples]),
        "avg_lines": np.mean([s.metadata['num_lines'] for s in samples]),
        "std_lines": np.std([s.metadata['num_lines'] for s in samples]),
        "avg_line_length": np.mean([s.metadata['avg_line_length'] for s in samples]),
        "rhyme_patterns": set(s.rhyme_pattern for s in samples),
        "rhyme_types": set(s.rhyme_type for s in samples),
        "topics": set(s.metadata['topic'] for s in samples),
        "styles": set(s.metadata['style'] for s in samples)
    }
    
    return stats

if __name__ == "__main__":
    # Test dataset creation
    print("🧪 Testing dataset creation...")
    
    # Create small test datasets
    rhyme_path = create_rhyme_dataset(num_samples=10)
    non_rhyme_path = create_non_rhyme_dataset(num_samples=10)
    
    # Analyze datasets
    rhyme_stats = analyze_dataset(rhyme_path)
    non_rhyme_stats = analyze_dataset(non_rhyme_path)
    
    print("\n📊 Dataset Analysis:")
    print("Rhyming dataset:", rhyme_stats)
    print("Non-rhyming dataset:", non_rhyme_stats) 