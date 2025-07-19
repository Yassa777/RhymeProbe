#!/usr/bin/env python3
"""
Main script to run rhyme probe experiments.
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch

from src.experiments.rhyme_probe import RhymeProbeExperiment


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config for easier access
    flattened_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flattened_config[f"{section}_{key}"] = value
        else:
            flattened_config[section] = values
    
    return flattened_config


def main():
    parser = argparse.ArgumentParser(description="Run rhyme probe experiments")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--phase", type=str, choices=["1", "2", "3", "4", "all"],
                       default="all", help="Which phase to run")
    parser.add_argument("--output-dir", type=str, default="experiments",
                       help="Output directory for results")
    parser.add_argument("--model-name", type=str, default="google/gemma-2b",
                       help="Model to use for experiments")
    parser.add_argument("--sae-path", type=str, default=None,
                       help="Path to SAE model")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["output_dir"] = args.output_dir
    config["model_name"] = args.model_name
    config["sae_path"] = args.sae_path
    
    # Setup logging
    setup_logging(config.get("output_log_level", "INFO"))
    logger = logging.getLogger(__name__)
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Create experiment
    experiment = RhymeProbeExperiment(config)
    
    # Run specified phase(s)
    if args.phase == "all":
        logger.info("Running full experiment pipeline")
        results = experiment.run_full_experiment()
    elif args.phase == "1":
        logger.info("Running Phase 1: Dataset Creation")
        results = experiment.run_phase_1()
    elif args.phase == "2":
        logger.info("Running Phase 2: Feature Discovery")
        results = experiment.run_phase_2()
    elif args.phase == "3":
        logger.info("Running Phase 3: Causal Probing")
        results = experiment.run_phase_3()
    elif args.phase == "4":
        logger.info("Running Phase 4: Generalization")
        results = experiment.run_phase_4()
    
    logger.info("Experiment completed successfully!")
    return results


if __name__ == "__main__":
    main() 