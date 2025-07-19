"""
Metrics and evaluation utilities for rhyme analysis.
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import pronouncing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class RhymeMetrics:
    """Metrics for evaluating rhyme quality and model performance."""
    
    def __init__(self):
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_rhyme_rate(self, lines: List[str]) -> float:
        """Calculate the percentage of consecutive lines that rhyme."""
        if len(lines) < 2:
            return 0.0
        
        rhyme_count = 0
        total_pairs = len(lines) - 1
        
        for i in range(len(lines) - 1):
            if self._lines_rhyme(lines[i], lines[i + 1]):
                rhyme_count += 1
        
        return rhyme_count / total_pairs
    
    def calculate_phonetic_similarity(self, lines: List[str]) -> List[float]:
        """Calculate phonetic similarity between consecutive lines."""
        similarities = []
        
        for i in range(len(lines) - 1):
            similarity = self._calculate_line_similarity(lines[i], lines[i + 1])
            similarities.append(similarity)
        
        return similarities
    
    def calculate_semantic_coherence(self, lines: List[str]) -> float:
        """Calculate semantic coherence using SBERT embeddings."""
        if len(lines) < 2:
            return 0.0
        
        # Get embeddings for all lines
        embeddings = self.sbert_model.encode(lines)
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def calculate_syntactic_validity(self, lines: List[str]) -> float:
        """Calculate syntactic validity score."""
        # This is a placeholder - you can implement more sophisticated
        # syntactic analysis using spaCy or NLTK
        return 0.8  # Placeholder
    
    def evaluate_patching_effect(self, original_text: str, patched_text: str) -> Dict:
        """Evaluate the effect of feature patching on rhyme quality."""
        original_lines = [line.strip() for line in original_text.split('\n') if line.strip()]
        patched_lines = [line.strip() for line in patched_text.split('\n') if line.strip()]
        
        original_metrics = self._calculate_all_metrics(original_lines)
        patched_metrics = self._calculate_all_metrics(patched_lines)
        
        # Calculate improvements
        improvements = {}
        for metric in original_metrics:
            if metric in patched_metrics:
                improvements[f"{metric}_improvement"] = (
                    patched_metrics[metric] - original_metrics[metric]
                )
        
        return {
            "original": original_metrics,
            "patched": patched_metrics,
            "improvements": improvements
        }
    
    def _lines_rhyme(self, line1: str, line2: str) -> bool:
        """Check if two lines rhyme."""
        words1 = line1.split()
        words2 = line2.split()
        
        if not words1 or not words2:
            return False
        
        last_word1 = words1[-1].lower().strip(".,!?;:")
        last_word2 = words2[-1].lower().strip(".,!?;:")
        
        return pronouncing.rhyme(last_word1, last_word2)
    
    def _calculate_line_similarity(self, line1: str, line2: str) -> float:
        """Calculate phonetic similarity between two lines."""
        words1 = line1.split()
        words2 = line2.split()
        
        if not words1 or not words2:
            return 0.0
        
        last_word1 = words1[-1].lower().strip(".,!?;:")
        last_word2 = words2[-1].lower().strip(".,!?;:")
        
        # Get pronunciations
        phones1 = pronouncing.phones_for_word(last_word1)
        phones2 = pronouncing.phones_for_word(last_word2)
        
        if phones1 and phones2:
            # Calculate similarity based on pronunciation
            return self._pronunciation_similarity(phones1[0], phones2[0])
        
        return 0.0
    
    def _pronunciation_similarity(self, phones1: str, phones2: str) -> float:
        """Calculate similarity between two pronunciations."""
        phones1_list = phones1.split()
        phones2_list = phones2.split()
        
        # Check for exact match
        if phones1_list == phones2_list:
            return 1.0
        
        # Check for similar ending
        if len(phones1_list) >= 2 and len(phones2_list) >= 2:
            if phones1_list[-2:] == phones2_list[-2:]:
                return 0.9
            elif phones1_list[-1] == phones2_list[-1]:
                return 0.7
        
        return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_all_metrics(self, lines: List[str]) -> Dict:
        """Calculate all metrics for a set of lines."""
        return {
            "rhyme_rate": self.calculate_rhyme_rate(lines),
            "avg_phonetic_similarity": np.mean(self.calculate_phonetic_similarity(lines)),
            "semantic_coherence": self.calculate_semantic_coherence(lines),
            "syntactic_validity": self.calculate_syntactic_validity(lines)
        }
    
    def statistical_significance_test(self, control_scores: List[float], 
                                    treatment_scores: List[float]) -> Dict:
        """Perform statistical significance test."""
        from scipy import stats
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_scores) - 1) * np.var(control_scores, ddof=1) +
                             (len(treatment_scores) - 1) * np.var(treatment_scores, ddof=1)) /
                            (len(control_scores) + len(treatment_scores) - 2))
        
        cohens_d = (np.mean(treatment_scores) - np.mean(control_scores)) / pooled_std
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "control_mean": np.mean(control_scores),
            "treatment_mean": np.mean(treatment_scores)
        } 