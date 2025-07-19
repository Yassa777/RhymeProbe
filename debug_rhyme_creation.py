#!/usr/bin/env python3
"""
Debug script for rhyme creation and verification
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.dataset_creation import RhymeDatasetCreator

def test_rhyme_creation():
    """Test rhyme creation step by step."""
    creator = RhymeDatasetCreator()
    
    print("🧪 Testing rhyme creation...")
    
    # Test 1: Create a simple rhyming text
    print("\n1. Creating AABB rhyming text:")
    text = creator.create_rhyming_text("AABB", "nature_poem", "nature")
    print(f"Generated text:\n{text}")
    
    # Test 2: Verify the rhyming
    print(f"\n2. Verifying rhyming:")
    is_rhyming = creator.verify_rhyming(text, "AABB")
    print(f"Verification result: {is_rhyming}")
    
    # Test 3: Analyze the text
    print(f"\n3. Analyzing text:")
    analysis = creator.rhyme_detector.analyze_rhyme_pattern(text)
    print(f"Rhyme type: {analysis.rhyme_type}")
    print(f"Confidence: {analysis.confidence}")
    print(f"Rhyme pattern: {analysis.rhyme_pattern}")
    
    # Test 4: Check individual word pairs
    print(f"\n4. Checking word pairs:")
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for i, line in enumerate(lines):
        words = line.split()
        if words:
            last_word = words[-1].lower().strip(".,!?;:")
            print(f"Line {i}: ends with '{last_word}'")
    
    # Test 5: Check specific rhyme pairs for AABB
    if len(lines) >= 4:
        print(f"\n5. Checking AABB pattern:")
        word1 = lines[0].split()[-1].lower().strip(".,!?;:")
        word2 = lines[1].split()[-1].lower().strip(".,!?;:")
        word3 = lines[2].split()[-1].lower().strip(".,!?;:")
        word4 = lines[3].split()[-1].lower().strip(".,!?;:")
        
        print(f"Words: {word1}, {word2}, {word3}, {word4}")
        
        rhyme1, conf1 = creator.rhyme_detector.detect_rhyme_type(word1, word2)
        rhyme2, conf2 = creator.rhyme_detector.detect_rhyme_type(word3, word4)
        
        print(f"Lines 1-2: {word1}-{word2} = {rhyme1} (confidence: {conf1:.2f})")
        print(f"Lines 3-4: {word3}-{word4} = {rhyme2} (confidence: {conf2:.2f})")

if __name__ == "__main__":
    test_rhyme_creation() 