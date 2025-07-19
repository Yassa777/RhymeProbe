"""
Dataset management for rhyme probe experiments.
Handles prompt generation, rhyme annotation, and data loading.
"""

import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pronouncing
from transformers import AutoTokenizer
import random


@dataclass
class RhymeSample:
    """Single sample for rhyme analysis."""
    prompt: str
    completion: str
    lines: List[str]
    rhyme_labels: List[bool]
    rhyme_types: List[str]
    phonetic_similarities: List[float]
    metadata: Dict


class RhymeDataset:
    """Dataset for rhyme probe experiments."""
    
    def __init__(self, model_name: str = "google/gemma-2b"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.samples: List[RhymeSample] = []
        
    def generate_rhyme_prompts(self, n_samples: int = 3000) -> List[str]:
        """Generate prompts designed to produce rhyming and non-rhyming outputs."""
        prompts = []
        
        # Template-based prompt generation
        templates = [
            "Write a short poem about {topic} that {rhyme_instruction}",
            "Complete this {style} poem: {start_line}",
            "Create a {rhyme_scheme} poem about {topic}",
            "Write {lines} lines about {topic} that {rhyme_instruction}"
        ]
        
        topics = ["nature", "love", "technology", "cities", "oceans", "mountains", 
                 "friendship", "time", "dreams", "music", "art", "science"]
        
        rhyme_instructions = [
            "rhymes at the end of each line",
            "does not rhyme",
            "has an AABB rhyme scheme",
            "has an ABAB rhyme scheme"
        ]
        
        for _ in range(n_samples):
            template = random.choice(templates)
            topic = random.choice(topics)
            rhyme_instruction = random.choice(rhyme_instructions)
            
            if "rhyme_scheme" in template:
                rhyme_scheme = random.choice(["AABB", "ABAB", "ABBA"])
                prompt = template.format(rhyme_scheme=rhyme_scheme, topic=topic)
            elif "start_line" in template:
                start_lines = [
                    "The sun sets in the west",
                    "Beneath the starry night",
                    "In gardens full of light",
                    "Where dreams and hopes take flight"
                ]
                start_line = random.choice(start_lines)
                style = random.choice(["romantic", "modern", "classical"])
                prompt = template.format(start_line=start_line, style=style)
            elif "lines" in template:
                lines = random.choice([2, 4, 6, 8])
                prompt = template.format(lines=lines, topic=topic, 
                                       rhyme_instruction=rhyme_instruction)
            else:
                prompt = template.format(topic=topic, rhyme_instruction=rhyme_instruction)
            
            prompts.append(prompt)
        
        return prompts
    
    def detect_rhymes(self, lines: List[str]) -> Tuple[List[bool], List[str], List[float]]:
        """Detect rhymes in a list of lines."""
        rhyme_labels = []
        rhyme_types = []
        phonetic_similarities = []
        
        for i in range(len(lines) - 1):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()
            
            # Get last word of each line
            words1 = line1.split()
            words2 = line2.split()
            
            if not words1 or not words2:
                rhyme_labels.append(False)
                rhyme_types.append("no_words")
                phonetic_similarities.append(0.0)
                continue
            
            last_word1 = words1[-1].lower().strip(".,!?;:")
            last_word2 = words2[-1].lower().strip(".,!?;:")
            
            # Get pronunciations
            phones1 = pronouncing.phones_for_word(last_word1)
            phones2 = pronouncing.phones_for_word(last_word2)
            
            if phones1 and phones2:
                # Check for perfect rhyme
                if pronouncing.rhyme(last_word1, last_word2):
                    rhyme_labels.append(True)
                    rhyme_types.append("perfect_rhyme")
                    phonetic_similarities.append(1.0)
                else:
                    # Check for slant rhyme (similar ending sounds)
                    rhyme_score = self._calculate_slant_rhyme(phones1[0], phones2[0])
                    if rhyme_score > 0.7:
                        rhyme_labels.append(True)
                        rhyme_types.append("slant_rhyme")
                        phonetic_similarities.append(rhyme_score)
                    else:
                        rhyme_labels.append(False)
                        rhyme_types.append("no_rhyme")
                        phonetic_similarities.append(rhyme_score)
            else:
                rhyme_labels.append(False)
                rhyme_types.append("unknown_pronunciation")
                phonetic_similarities.append(0.0)
        
        return rhyme_labels, rhyme_types, phonetic_similarities
    
    def _calculate_slant_rhyme(self, phones1: str, phones2: str) -> float:
        """Calculate slant rhyme similarity between two pronunciations."""
        # Simple implementation - can be enhanced
        phones1_list = phones1.split()
        phones2_list = phones2.split()
        
        # Check if they end with similar sounds
        if len(phones1_list) >= 2 and len(phones2_list) >= 2:
            if phones1_list[-2:] == phones2_list[-2:]:
                return 0.9
            elif phones1_list[-1] == phones2_list[-1]:
                return 0.7
        
        return 0.0
    
    def add_sample(self, prompt: str, completion: str):
        """Add a sample to the dataset."""
        # Split completion into lines
        lines = [line.strip() for line in completion.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return  # Need at least 2 lines for rhyme analysis
        
        # Detect rhymes
        rhyme_labels, rhyme_types, phonetic_similarities = self.detect_rhymes(lines)
        
        # Create sample
        sample = RhymeSample(
            prompt=prompt,
            completion=completion,
            lines=lines,
            rhyme_labels=rhyme_labels,
            rhyme_types=rhyme_types,
            phonetic_similarities=phonetic_similarities,
            metadata={
                "num_lines": len(lines),
                "avg_line_length": sum(len(line) for line in lines) / len(lines),
                "rhyme_rate": sum(rhyme_labels) / len(rhyme_labels) if rhyme_labels else 0.0
            }
        )
        
        self.samples.append(sample)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert samples to pandas DataFrame."""
        data = []
        for sample in self.samples:
            data.append({
                "prompt": sample.prompt,
                "completion": sample.completion,
                "lines": sample.lines,
                "rhyme_labels": sample.rhyme_labels,
                "rhyme_types": sample.rhyme_types,
                "phonetic_similarities": sample.phonetic_similarities,
                "num_lines": sample.metadata["num_lines"],
                "avg_line_length": sample.metadata["avg_line_length"],
                "rhyme_rate": sample.metadata["rhyme_rate"]
            })
        return pd.DataFrame(data)
    
    def save(self, filepath: str):
        """Save dataset to file."""
        df = self.to_dataframe()
        df.to_json(filepath, orient='records', indent=2)
    
    def load(self, filepath: str):
        """Load dataset from file."""
        df = pd.read_json(filepath, orient='records')
        self.samples = []
        
        for _, row in df.iterrows():
            sample = RhymeSample(
                prompt=row["prompt"],
                completion=row["completion"],
                lines=row["lines"],
                rhyme_labels=row["rhyme_labels"],
                rhyme_types=row["rhyme_types"],
                phonetic_similarities=row["phonetic_similarities"],
                metadata={
                    "num_lines": row["num_lines"],
                    "avg_line_length": row["avg_line_length"],
                    "rhyme_rate": row["rhyme_rate"]
                }
            )
            self.samples.append(sample) 