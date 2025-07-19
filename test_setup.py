#!/usr/bin/env python3
"""
Test script to verify the environment setup.
"""

import sys
import torch
import logging

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import transformers
        print("✓ transformers")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import sae_lens
        print("✓ sae_lens")
    except ImportError as e:
        print(f"✗ sae_lens: {e}")
        print("  Note: sae_lens is optional for initial testing")
    
    try:
        import pronouncing
        print("✓ pronouncing")
    except ImportError as e:
        print(f"✗ pronouncing: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ sentence_transformers")
    except ImportError as e:
        print(f"✗ sentence_transformers: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import spacy
        print("✓ spacy")
    except ImportError as e:
        print(f"✗ spacy: {e}")
        return False
    
    try:
        import nltk
        print("✓ nltk")
    except ImportError as e:
        print(f"✗ nltk: {e}")
        return False
    
    return True

def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("✗ No GPU available")
        print("  Note: Experiments will run on CPU (much slower)")
        return False

def test_model_loading():
    """Test loading the Gemma model."""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        print("✓ Tokenizer loaded successfully")
        
        # Test model loading (small test)
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2b",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✓ Model loaded successfully")
        else:
            print("⚠ Skipping model loading test (no GPU)")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_dataset():
    """Test dataset functionality."""
    print("\nTesting dataset functionality...")
    
    try:
        from src.data.dataset import RhymeDataset
        
        dataset = RhymeDataset()
        prompts = dataset.generate_rhyme_prompts(5)
        
        print(f"✓ Generated {len(prompts)} test prompts")
        
        # Test rhyme detection
        test_lines = ["The sun sets in the west", "As birds return to their nest"]
        rhyme_labels, rhyme_types, similarities = dataset.detect_rhymes(test_lines)
        
        print(f"✓ Rhyme detection working: {rhyme_labels}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_metrics():
    """Test metrics functionality."""
    print("\nTesting metrics functionality...")
    
    try:
        from src.metrics.rhyme_metrics import RhymeDetector
        
        detector = RhymeDetector()
        test_pairs = [("cat", "hat"), ("light", "bright"), ("moon", "spoon")]
        
        for word1, word2 in test_pairs:
            rhyme_type, confidence = detector.detect_rhyme_type(word1, word2)
            print(f"✓ {word1}-{word2}: {rhyme_type} (confidence: {confidence:.2f})")
        
        # Test text analysis
        test_text = "The cat sat on the mat with a hat"
        analysis = detector.analyze_rhyme_pattern(test_text)
        print(f"✓ Text analysis: {analysis.rhyme_type} (confidence: {analysis.confidence:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Rhyme Probe Environment Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("GPU", test_gpu),
        ("Model Loading", test_model_loading),
        ("Dataset", test_dataset),
        ("Metrics", test_metrics)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Environment is ready for experiments.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("You may still be able to run experiments with limited functionality.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 