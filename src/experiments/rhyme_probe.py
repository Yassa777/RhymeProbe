"""
Main experiment runner for rhyme probe experiments.
Orchestrates the entire pipeline from data generation to causal analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..data.dataset import RhymeDataset
from ..models.sae_extractor import SAEFeatureExtractor
from ..utils.metrics import RhymeMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RhymeProbeExperiment:
    """Main experiment class for rhyme probe analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "experiments"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.dataset = RhymeDataset(config.get("model_name", "google/gemma-2b"))
        self.feature_extractor = SAEFeatureExtractor(
            model_name=config.get("model_name", "google/gemma-2b"),
            sae_path=config.get("sae_path")
        )
        self.metrics = RhymeMetrics()
        
        # Experiment state
        self.results = {}
        
    def run_phase_1(self) -> Dict:
        """Phase 1: Dataset Creation & Rhyme Annotation."""
        logger.info("Starting Phase 1: Dataset Creation & Rhyme Annotation")
        
        # Generate prompts
        n_samples = self.config.get("n_samples", 3000)
        prompts = self.dataset.generate_rhyme_prompts(n_samples)
        
        # Generate completions (placeholder - you'll need to implement this)
        # For now, we'll use dummy completions
        completions = self._generate_completions(prompts)
        
        # Add samples to dataset
        for prompt, completion in zip(prompts, completions):
            self.dataset.add_sample(prompt, completion)
        
        # Save dataset
        dataset_path = self.output_dir / "rhyme_dataset.json"
        self.dataset.save(str(dataset_path))
        
        # Calculate basic statistics
        stats = self._calculate_dataset_stats()
        
        logger.info(f"Phase 1 complete. Dataset saved to {dataset_path}")
        logger.info(f"Dataset statistics: {stats}")
        
        return {
            "phase": "dataset_creation",
            "n_samples": len(self.dataset.samples),
            "stats": stats,
            "dataset_path": str(dataset_path)
        }
    
    def run_phase_2(self) -> Dict:
        """Phase 2: Feature Discovery."""
        logger.info("Starting Phase 2: Feature Discovery")
        
        # Load dataset if not already loaded
        if not self.dataset.samples:
            dataset_path = self.output_dir / "rhyme_dataset.json"
            if dataset_path.exists():
                self.dataset.load(str(dataset_path))
            else:
                raise FileNotFoundError("Dataset not found. Run Phase 1 first.")
        
        # Extract features and calculate importance
        layer_idx = self.config.get("layer_idx", -1)
        feature_importance = self.feature_extractor.get_feature_importance(
            self.dataset, layer_idx
        )
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Save feature importance
        importance_path = self.output_dir / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=2)
        
        # Get top features
        top_k = self.config.get("top_k_features", 100)
        top_features = sorted_features[:top_k]
        
        logger.info(f"Phase 2 complete. Top {top_k} features identified.")
        logger.info(f"Feature importance saved to {importance_path}")
        
        return {
            "phase": "feature_discovery",
            "feature_importance": feature_importance,
            "top_features": top_features,
            "importance_path": str(importance_path)
        }
    
    def run_phase_3(self) -> Dict:
        """Phase 3: Causal Probing."""
        logger.info("Starting Phase 3: Causal Probing")
        
        # Load feature importance if not already available
        if "feature_importance" not in self.results:
            importance_path = self.output_dir / "feature_importance.json"
            if importance_path.exists():
                with open(importance_path, 'r') as f:
                    feature_importance = json.load(f)
            else:
                raise FileNotFoundError("Feature importance not found. Run Phase 2 first.")
        else:
            feature_importance = self.results["feature_importance"]
        
        # Get top features for patching
        top_k = self.config.get("top_k_features", 100)
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        top_features = sorted_features[:top_k]
        
        # Run patching experiments
        patching_results = self._run_patching_experiments(top_features)
        
        # Save patching results
        patching_path = self.output_dir / "patching_results.json"
        with open(patching_path, 'w') as f:
            json.dump(patching_results, f, indent=2)
        
        logger.info(f"Phase 3 complete. Patching results saved to {patching_path}")
        
        return {
            "phase": "causal_probing",
            "patching_results": patching_results,
            "patching_path": str(patching_path)
        }
    
    def run_phase_4(self) -> Dict:
        """Phase 4: Generalization & Validation."""
        logger.info("Starting Phase 4: Generalization & Validation")
        
        # Load previous results
        patching_path = self.output_dir / "patching_results.json"
        if patching_path.exists():
            with open(patching_path, 'r') as f:
                patching_results = json.load(f)
        else:
            raise FileNotFoundError("Patching results not found. Run Phase 3 first.")
        
        # Run generalization tests
        generalization_results = self._run_generalization_tests(patching_results)
        
        # Run negative controls
        control_results = self._run_negative_controls()
        
        # Save generalization results
        gen_path = self.output_dir / "generalization_results.json"
        with open(gen_path, 'w') as f:
            json.dump({
                "generalization": generalization_results,
                "controls": control_results
            }, f, indent=2)
        
        logger.info(f"Phase 4 complete. Generalization results saved to {gen_path}")
        
        return {
            "phase": "generalization",
            "generalization_results": generalization_results,
            "control_results": control_results,
            "generalization_path": str(gen_path)
        }
    
    def run_full_experiment(self) -> Dict:
        """Run the complete experiment pipeline."""
        logger.info("Starting full rhyme probe experiment")
        
        start_time = datetime.now()
        
        # Run all phases
        self.results["phase_1"] = self.run_phase_1()
        self.results["phase_2"] = self.run_phase_2()
        self.results["phase_3"] = self.run_phase_3()
        self.results["phase_4"] = self.run_phase_4()
        
        # Save complete results
        results_path = self.output_dir / "complete_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Full experiment complete in {duration}")
        logger.info(f"Complete results saved to {results_path}")
        
        return self.results
    
    def _generate_completions(self, prompts: List[str]) -> List[str]:
        """Generate completions for prompts (placeholder)."""
        # This is a placeholder - you'll need to implement actual generation
        # using the language model
        completions = []
        for prompt in prompts:
            # For now, return dummy completions
            if "rhyme" in prompt.lower():
                completion = "The sun sets in the west\nAs birds return to their nest"
            else:
                completion = "The sun sets in the west\nAs birds fly home to rest"
            completions.append(completion)
        
        return completions
    
    def _calculate_dataset_stats(self) -> Dict:
        """Calculate basic dataset statistics."""
        if not self.dataset.samples:
            return {}
        
        rhyme_rates = [sample.metadata["rhyme_rate"] for sample in self.dataset.samples]
        line_counts = [sample.metadata["num_lines"] for sample in self.dataset.samples]
        
        return {
            "total_samples": len(self.dataset.samples),
            "avg_rhyme_rate": np.mean(rhyme_rates),
            "std_rhyme_rate": np.std(rhyme_rates),
            "avg_lines": np.mean(line_counts),
            "std_lines": np.std(line_counts)
        }
    
    def _run_patching_experiments(self, top_features: List[tuple]) -> Dict:
        """Run feature patching experiments."""
        # This is a placeholder for patching experiments
        # You'll need to implement actual patching logic
        
        results = {
            "feature_patching": {},
            "strength_sweep": {},
            "temporal_analysis": {}
        }
        
        # Test a few top features
        for i, (feature_name, importance) in enumerate(top_features[:10]):
            logger.info(f"Testing feature: {feature_name} (importance: {importance:.4f})")
            
            # Placeholder results
            results["feature_patching"][feature_name] = {
                "importance": importance,
                "rhyme_improvement": np.random.uniform(0, 0.2),  # Placeholder
                "p_value": np.random.uniform(0, 0.1)  # Placeholder
            }
        
        return results
    
    def _run_generalization_tests(self, patching_results: Dict) -> Dict:
        """Run generalization tests."""
        # Placeholder for generalization tests
        return {
            "different_styles": {"success_rate": 0.75},
            "multilingual": {"success_rate": 0.60},
            "out_of_distribution": {"success_rate": 0.65}
        }
    
    def _run_negative_controls(self) -> Dict:
        """Run negative control experiments."""
        # Placeholder for negative controls
        return {
            "random_features": {"rhyme_improvement": 0.02},
            "unrelated_features": {"rhyme_improvement": 0.01},
            "baseline": {"rhyme_improvement": 0.0}
        } 