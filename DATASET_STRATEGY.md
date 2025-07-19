# 🎯 Dataset Strategy for Rhyme Probe Experiments

## 📋 **Overview**

This document outlines the comprehensive dataset strategy for identifying and causally probing rhyming behavior in large language models using sparse autoencoders.

## 🎯 **Dataset Requirements**

### **Primary Objectives:**
1. **Rhyme Detection**: Create datasets that clearly distinguish between rhyming and non-rhyming text
2. **Pattern Diversity**: Include various rhyme patterns (AABB, ABAB, ABBA, AAAA)
3. **Topic Coverage**: Span multiple domains (nature, emotions, abstract concepts)
4. **Quality Control**: Ensure high-quality, verified rhyming/non-rhyming pairs
5. **Scalability**: Support experiments from small-scale (100 samples) to large-scale (10,000+ samples)

## 📊 **Dataset Architecture**

### **1. Rhyming Dataset**
- **Size**: 10,000 samples (configurable)
- **Patterns**: AABB, ABAB, ABBA, AAAA
- **Topics**: Nature, Emotions, Abstract concepts
- **Styles**: Nature poems, Emotional poems, Abstract poems
- **Verification**: Automated rhyme detection with confidence > 0.8

### **2. Non-Rhyming Dataset**
- **Size**: 10,000 samples (configurable)
- **Pattern**: No systematic rhyming
- **Topics**: Same as rhyming dataset
- **Styles**: Same as rhyming dataset
- **Verification**: Ensure rhyme density < 0.1

### **3. Test Dataset**
- **Size**: 2,000 samples (balanced)
- **Purpose**: Evaluation and generalization testing
- **Split**: 50% rhyming, 50% non-rhyming

## 🏗️ **Generation Strategy**

### **Template-Based Generation**

#### **Rhyme Patterns:**
```python
rhyme_patterns = {
    "AABB": [(0, 1), (2, 3)],  # First two lines rhyme, next two rhyme
    "ABAB": [(0, 2), (1, 3)],  # Alternating rhyme
    "ABBA": [(0, 3), (1, 2)],  # Envelope rhyme
    "AAAA": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # All rhyme
}
```

#### **Poem Templates:**
```python
templates = {
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
```

### **Rhyming Word Pairs**

#### **Nature Domain:**
- tree-free, sky-high, moon-soon, star-far
- light-bright, night-sight, wind-find, rain-pain
- sun-run, sea-free, wave-save, bird-word

#### **Emotions Domain:**
- love-dove, heart-start, soul-whole, tears-fears
- joy-boy, pain-rain, hope-rope, dream-seem
- feel-real, care-share, mind-find, life-strife

#### **Abstract Domain:**
- time-rhyme, space-place, world-unfurled, thought-bought
- word-heard, way-day, end-friend, start-heart
- path-bath, light-sight, dark-mark, true-blue

## 🔍 **Quality Control**

### **Automated Verification**

#### **Rhyming Verification:**
```python
def verify_rhyming(text: str, expected_pattern: str) -> bool:
    # Extract last words of each line
    # Check rhyme pattern using pronouncing library
    # Require confidence > 0.8 for verification
```

#### **Non-Rhyming Verification:**
```python
def verify_non_rhyming(text: str) -> bool:
    # Check consecutive lines for rhymes
    # Ensure rhyme density < 0.1
    # Require confidence < 0.5 for non-rhyming
```

### **Feature Extraction**

#### **Rhyme Metrics:**
- **Rhyme Density**: Percentage of rhyming word pairs
- **Perfect Rhyme Ratio**: Ratio of perfect rhymes to total rhymes
- **Slant Rhyme Ratio**: Ratio of slant rhymes to total rhymes
- **Assonance Ratio**: Ratio of assonance to total rhymes
- **Consonance Ratio**: Ratio of consonance to total rhymes
- **Rhyme Type Diversity**: Diversity of rhyme types used

## 📈 **Dataset Statistics**

### **Expected Metrics:**

#### **Rhyming Dataset:**
- **Rhyme Density**: 0.02-0.04 (2-4% of word pairs rhyme)
- **Perfect Rhyme Ratio**: > 0.95 (95%+ perfect rhymes)
- **Average Lines**: 4 lines per poem
- **Average Line Length**: 25-30 characters
- **Pattern Distribution**: Even distribution across AABB, ABAB, ABBA, AAAA

#### **Non-Rhyming Dataset:**
- **Rhyme Density**: < 0.01 (less than 1% accidental rhymes)
- **Perfect Rhyme Ratio**: < 0.1 (few perfect rhymes)
- **Average Lines**: 4 lines per poem
- **Average Line Length**: 25-30 characters
- **Pattern Distribution**: No systematic patterns

## 🚀 **Implementation Strategy**

### **Phase 1: Small-Scale Testing**
```bash
# Create debug datasets (10 samples each)
python src/data/dataset_creation.py
```

### **Phase 2: Medium-Scale Validation**
```bash
# Create validation datasets (1,000 samples each)
python run_experiments.py --debug --phase 1
```

### **Phase 3: Full-Scale Production**
```bash
# Create production datasets (10,000 samples each)
python run_experiments.py --phase 1
```

## 📊 **Dataset Analysis**

### **Comprehensive Statistics:**
```python
def analyze_dataset(dataset_path: str) -> Dict:
    return {
        "total_samples": len(samples),
        "avg_rhyme_density": np.mean(rhyme_densities),
        "std_rhyme_density": np.std(rhyme_densities),
        "avg_lines": np.mean(line_counts),
        "std_lines": np.std(line_counts),
        "avg_line_length": np.mean(line_lengths),
        "rhyme_patterns": set(patterns),
        "rhyme_types": set(rhyme_types),
        "topics": set(topics),
        "styles": set(styles)
    }
```

## 🔄 **Iterative Improvement**

### **Quality Metrics:**
1. **Rhyme Accuracy**: > 95% of rhyming samples pass verification
2. **Non-Rhyme Accuracy**: > 95% of non-rhyming samples pass verification
3. **Pattern Coverage**: All intended patterns represented
4. **Topic Balance**: Even distribution across topics
5. **Style Diversity**: Multiple poetic styles represented

### **Feedback Loop:**
1. Generate initial dataset
2. Analyze quality metrics
3. Adjust templates/vocabulary if needed
4. Regenerate with improvements
5. Repeat until quality thresholds met

## 📁 **File Organization**

```
data/
├── rhyme_dataset_YYYYMMDD_HHMMSS.json
├── non_rhyme_dataset_YYYYMMDD_HHMMSS.json
├── test_dataset_YYYYMMDD_HHMMSS.json
└── analysis/
    ├── rhyme_dataset_stats.json
    ├── non_rhyme_dataset_stats.json
    └── comparison_report.json
```

## 🎯 **Success Criteria**

### **Technical Success:**
- [ ] Rhyming dataset: > 95% verification rate
- [ ] Non-rhyming dataset: > 95% verification rate
- [ ] Pattern coverage: All 4 patterns represented
- [ ] Topic balance: Even distribution across 3 topics
- [ ] Style diversity: All 3 styles represented

### **Research Success:**
- [ ] Clear distinction between rhyming/non-rhyming
- [ ] Sufficient samples for statistical significance
- [ ] Diverse enough for generalization testing
- [ ] High-quality for human evaluation
- [ ] Reproducible generation process

## 🚀 **Next Steps**

1. **Run small-scale test** to verify generation quality
2. **Validate rhyme detection** on generated samples
3. **Scale up to medium size** for initial experiments
4. **Run full-scale generation** for production experiments
5. **Analyze and iterate** based on results

---

**🎯 Ready to generate your rhyme probe datasets!**

The infrastructure is in place, the strategy is comprehensive, and the quality control measures are robust. You're ready to create datasets that will enable powerful rhyme probe experiments. 