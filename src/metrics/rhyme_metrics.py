"""
Rhyme Detection and Analysis Metrics
===================================

This module provides comprehensive rhyme detection and analysis capabilities
for the rhyme probe experiments.
"""

import pronouncing
import re
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass
class RhymeInfo:
    """Information about a rhyme pattern."""
    words: List[str]
    rhyme_pattern: str
    rhyme_sounds: List[str]
    confidence: float
    rhyme_type: str  # 'perfect', 'slant', 'assonance', etc.

class RhymeDetector:
    """Advanced rhyme detection and analysis."""
    
    def __init__(self):
        self.vowel_sounds = set('AEIOU')
        self.consonant_sounds = set('BCDFGHJKLMNPQRSTVWXYZ')
        
    def get_pronunciation(self, word: str) -> List[str]:
        """Get pronunciation for a word."""
        try:
            pronunciations = pronouncing.phones_for_word(word.lower())
            if pronunciations:
                return pronouncing.pronouncing_list(pronunciations[0])
            return []
        except:
            return []
    
    def get_rhyming_part(self, pronunciation: List[str]) -> str:
        """Extract the rhyming part (last stressed vowel + following sounds)."""
        if not pronunciation:
            return ""
        
        # Find the last stressed vowel
        for i in range(len(pronunciation) - 1, -1, -1):
            if any(char in self.vowel_sounds for char in pronunciation[i]):
                # Check if it's stressed (has a digit)
                if any(char.isdigit() for char in pronunciation[i]):
                    return "".join(pronunciation[i:])
        
        # If no stressed vowel found, return the last syllable
        return "".join(pronunciation[-2:]) if len(pronunciation) >= 2 else "".join(pronunciation)
    
    def detect_rhyme_type(self, word1: str, word2: str) -> Tuple[str, float]:
        """Detect the type of rhyme between two words."""
        # Use pronouncing library's built-in rhyme detection
        try:
            # Check if words rhyme using pronouncing
            rhymes1 = pronouncing.rhymes(word1.lower())
            rhymes2 = pronouncing.rhymes(word2.lower())
            
            # Perfect rhyme
            if word2.lower() in rhymes1 or word1.lower() in rhymes2:
                return "perfect", 1.0
            
            # Get pronunciations for more detailed analysis
            pron1 = self.get_pronunciation(word1)
            pron2 = self.get_pronunciation(word2)
            
            if not pron1 or not pron2:
                return "none", 0.0
            
            rhyme1 = self.get_rhyming_part(pron1)
            rhyme2 = self.get_rhyming_part(pron2)
            
            if not rhyme1 or not rhyme2:
                return "none", 0.0
            
            # Slant rhyme (similar ending sounds)
            if rhyme1[-3:] == rhyme2[-3:] and len(rhyme1) >= 3 and len(rhyme2) >= 3:
                return "slant", 0.8
            
            # Assonance (same vowel sounds)
            vowels1 = "".join(c for c in rhyme1 if c in self.vowel_sounds)
            vowels2 = "".join(c for c in rhyme2 if c in self.vowel_sounds)
            if vowels1 == vowels2 and vowels1:
                return "assonance", 0.6
            
            # Consonance (same consonant sounds)
            cons1 = "".join(c for c in rhyme1 if c in self.consonant_sounds)
            cons2 = "".join(c for c in rhyme2 if c in self.consonant_sounds)
            if cons1 == cons2 and cons1:
                return "consonance", 0.4
            
            return "none", 0.0
            
        except Exception as e:
            print(f"Error in rhyme detection for {word1}-{word2}: {e}")
            return "none", 0.0
    
    def find_rhymes(self, word: str, word_list: List[str]) -> List[Tuple[str, str, float]]:
        """Find all rhyming words in a list."""
        rhymes = []
        for other_word in word_list:
            if other_word.lower() != word.lower():
                rhyme_type, confidence = self.detect_rhyme_type(word, other_word)
                if confidence > 0.0:
                    rhymes.append((other_word, rhyme_type, confidence))
        return sorted(rhymes, key=lambda x: x[2], reverse=True)
    
    def analyze_rhyme_pattern(self, text: str) -> RhymeInfo:
        """Analyze rhyme pattern in a text."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 2:
            return RhymeInfo(words, "", [], 0.0, "none")
        
        # Find rhyming pairs
        rhyme_pairs = []
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words[i+1:], i+1):
                rhyme_type, confidence = self.detect_rhyme_type(word1, word2)
                if confidence > 0.5:  # Threshold for meaningful rhymes
                    rhyme_pairs.append((word1, word2, rhyme_type, confidence))
        
        if not rhyme_pairs:
            return RhymeInfo(words, "", [], 0.0, "none")
        
        # Determine dominant rhyme type
        rhyme_counts = defaultdict(int)
        for _, _, rhyme_type, _ in rhyme_pairs:
            rhyme_counts[rhyme_type] += 1
        
        dominant_type = max(rhyme_counts.items(), key=lambda x: x[1])[0]
        
        # Create rhyme pattern
        pattern = []
        for word1, word2, rhyme_type, conf in rhyme_pairs:
            if rhyme_type == dominant_type:
                pattern.append(f"{word1}-{word2}")
        
        return RhymeInfo(
            words=words,
            rhyme_pattern=", ".join(pattern),
            rhyme_sounds=[f"{w1}-{w2}" for w1, w2, _, _ in rhyme_pairs],
            confidence=np.mean([conf for _, _, _, conf in rhyme_pairs]),
            rhyme_type=dominant_type
        )

def calculate_rhyme_density(text: str) -> float:
    """Calculate the density of rhyming words in text."""
    detector = RhymeDetector()
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) < 2:
        return 0.0
    
    rhyme_count = 0
    total_pairs = 0
    
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words[i+1:], i+1):
            total_pairs += 1
            _, confidence = detector.detect_rhyme_type(word1, word2)
            if confidence > 0.5:
                rhyme_count += 1
    
    return rhyme_count / total_pairs if total_pairs > 0 else 0.0

def extract_rhyming_features(text: str) -> Dict[str, float]:
    """Extract comprehensive rhyming features from text."""
    detector = RhymeDetector()
    analysis = detector.analyze_rhyme_pattern(text)
    
    features = {
        'rhyme_density': calculate_rhyme_density(text),
        'perfect_rhyme_ratio': 0.0,
        'slant_rhyme_ratio': 0.0,
        'assonance_ratio': 0.0,
        'consonance_ratio': 0.0,
        'avg_rhyme_confidence': analysis.confidence,
        'rhyme_type_diversity': 0.0
    }
    
    if analysis.rhyme_sounds:
        # Count different rhyme types
        rhyme_types = defaultdict(int)
        words = re.findall(r'\b\w+\b', text.lower())
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words[i+1:], i+1):
                rhyme_type, _ = detector.detect_rhyme_type(word1, word2)
                rhyme_types[rhyme_type] += 1
        
        total_rhymes = sum(rhyme_types.values())
        if total_rhymes > 0:
            features['perfect_rhyme_ratio'] = rhyme_types.get('perfect', 0) / total_rhymes
            features['slant_rhyme_ratio'] = rhyme_types.get('slant', 0) / total_rhymes
            features['assonance_ratio'] = rhyme_types.get('assonance', 0) / total_rhymes
            features['consonance_ratio'] = rhyme_types.get('consonance', 0) / total_rhymes
            features['rhyme_type_diversity'] = len(rhyme_types) / 4  # Normalize by max types
    
    return features

# Test function
def test_rhyme_detection():
    """Test the rhyme detection functionality."""
    detector = RhymeDetector()
    
    # Test cases with known rhymes
    test_pairs = [
        ("cat", "hat"),
        ("light", "bright"),
        ("moon", "spoon"),
        ("love", "dove"),
        ("cat", "dog"),
        ("light", "dark"),
        ("blue", "true"),
        ("night", "bright"),
        ("star", "far"),
        ("heart", "start")
    ]
    
    print("Testing rhyme detection:")
    for word1, word2 in test_pairs:
        rhyme_type, confidence = detector.detect_rhyme_type(word1, word2)
        print(f"{word1} - {word2}: {rhyme_type} (confidence: {confidence:.2f})")
    
    # Test text analysis
    test_text = "The cat sat on the mat with a hat"
    analysis = detector.analyze_rhyme_pattern(test_text)
    print(f"\nText analysis: {test_text}")
    print(f"Rhyme pattern: {analysis.rhyme_pattern}")
    print(f"Rhyme type: {analysis.rhyme_type}")
    print(f"Confidence: {analysis.confidence:.2f}")
    
    # Test feature extraction
    features = extract_rhyming_features(test_text)
    print(f"\nFeatures: {features}")
    
    # Test with a more complex rhyming text
    rhyming_text = "The night is bright with stars so far, my heart will start to beat"
    analysis2 = detector.analyze_rhyme_pattern(rhyming_text)
    print(f"\nRhyming text analysis: {rhyming_text}")
    print(f"Rhyme pattern: {analysis2.rhyme_pattern}")
    print(f"Rhyme type: {analysis2.rhyme_type}")
    print(f"Confidence: {analysis2.confidence:.2f}")

if __name__ == "__main__":
    test_rhyme_detection() 