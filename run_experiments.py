#!/usr/bin/env python3
"""
Rhyme Probe Experiment Runner
=============================

This script orchestrates the complete rhyme probe experiment pipeline:
1. Dataset creation
2. Sparse autoencoder training
3. Feature discovery and analysis
4. Causal probing experiments
5. Generalization testing
6. Results analysis and visualization
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import torch
import wandb
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset_creation import create_rhyme_dataset, create_non_rhyme_dataset
from src.models.sae_training import train_sparse_autoencoder
from src.experiments.feature_discovery import discover_rhyme_features
from src.experiments.causal_probing import run_causal_probing
from src.experiments.generalization import test_generalization
from src.utils.visualization import create_experiment_plots
from src.utils.metrics import compute_comprehensive_metrics

def setup_logging(config: Dict[str, Any], experiment_name: str) -> logging.Logger:
    """Set up logging configuration."""
    log_dir = Path(config['logging']['local']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def setup_wandb(config: Dict[str, Any], experiment_name: str) -> None:
    """Set up Weights & Biases logging."""
    if config['logging']['wandb']['entity'] is None:
        print("⚠️  Please set your W&B entity in the config file")
        return
    
    wandb.init(
        project=config['logging']['wandb']['project'],
        entity=config['logging']['wandb']['entity'],
        name=experiment_name,
        config=config,
        tags=["rhyme-probe", "interpretability", "sparse-autoencoder"]
    )

def create_output_directories(config: Dict[str, Any]) -> None:
    """Create output directories."""
    output_dirs = [
        config['output']['base_dir'],
        config['output']['models_dir'],
        config['output']['results_dir'],
        config['output']['plots_dir']
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def run_phase_1_dataset_creation(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, str]:
    """Phase 1: Create rhyming and non-rhyming datasets."""
    logger.info("🚀 Starting Phase 1: Dataset Creation")
    
    try:
        # Create rhyming dataset
        logger.info("Creating rhyming dataset...")
        rhyme_dataset_path = create_rhyme_dataset(
            num_samples=config['dataset']['rhyme_dataset']['num_samples'],
            min_length=config['dataset']['rhyme_dataset']['min_length'],
            max_length=config['dataset']['rhyme_dataset']['max_length'],
            rhyme_patterns=config['dataset']['rhyme_dataset']['rhyme_patterns'],
            rhyme_types=config['dataset']['rhyme_dataset']['rhyme_types']
        )
        
        # Create non-rhyming dataset
        logger.info("Creating non-rhyming dataset...")
        non_rhyme_dataset_path = create_non_rhyme_dataset(
            num_samples=config['dataset']['non_rhyme_dataset']['num_samples'],
            min_length=config['dataset']['non_rhyme_dataset']['min_length'],
            max_length=config['dataset']['non_rhyme_dataset']['max_length'],
            ensure_no_rhymes=config['dataset']['non_rhyme_dataset']['ensure_no_rhymes']
        )
        
        logger.info("✅ Phase 1 completed successfully")
        return {
            'rhyme_dataset': rhyme_dataset_path,
            'non_rhyme_dataset': non_rhyme_dataset_path
        }
        
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {e}")
        raise

def run_phase_2_sae_training(config: Dict[str, Any], dataset_paths: Dict[str, str], logger: logging.Logger) -> str:
    """Phase 2: Train sparse autoencoder."""
    logger.info("🚀 Starting Phase 2: Sparse Autoencoder Training")
    
    try:
        sae_model_path = train_sparse_autoencoder(
            model_name=config['model']['name'],
            rhyme_dataset_path=dataset_paths['rhyme_dataset'],
            non_rhyme_dataset_path=dataset_paths['non_rhyme_dataset'],
            d_model=config['sae']['d_model'],
            d_sae=config['sae']['d_sae'],
            l1_coefficient=config['sae']['l1_coefficient'],
            lr=config['sae']['lr'],
            batch_size=config['sae']['batch_size'],
            max_epochs=config['sae']['max_epochs'],
            warmup_steps=config['sae']['warmup_steps'],
            eval_frequency=config['sae']['eval_frequency'],
            save_frequency=config['sae']['save_frequency']
        )
        
        logger.info("✅ Phase 2 completed successfully")
        return sae_model_path
        
    except Exception as e:
        logger.error(f"❌ Phase 2 failed: {e}")
        raise

def run_phase_3_feature_discovery(config: Dict[str, Any], sae_model_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Phase 3: Discover rhyme-related features."""
    logger.info("🚀 Starting Phase 3: Feature Discovery")
    
    try:
        feature_results = discover_rhyme_features(
            sae_model_path=sae_model_path,
            model_name=config['model']['name'],
            num_features=config['sae']['feature_analysis']['num_features_to_analyze'],
            min_activation=config['sae']['feature_analysis']['min_activation_threshold'],
            max_activation=config['sae']['feature_analysis']['max_activation_threshold']
        )
        
        logger.info("✅ Phase 3 completed successfully")
        return feature_results
        
    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {e}")
        raise

def run_phase_4_causal_probing(config: Dict[str, Any], sae_model_path: str, feature_results: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Phase 4: Run causal probing experiments."""
    logger.info("🚀 Starting Phase 4: Causal Probing")
    
    try:
        probing_results = run_causal_probing(
            sae_model_path=sae_model_path,
            model_name=config['model']['name'],
            feature_results=feature_results,
            intervention_config=config['causal_probing']['intervention'],
            metrics=config['causal_probing']['metrics'],
            significance_config=config['causal_probing']['significance']
        )
        
        logger.info("✅ Phase 4 completed successfully")
        return probing_results
        
    except Exception as e:
        logger.error(f"❌ Phase 4 failed: {e}")
        raise

def run_phase_5_generalization(config: Dict[str, Any], sae_model_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Phase 5: Test generalization."""
    logger.info("🚀 Starting Phase 5: Generalization Testing")
    
    try:
        generalization_results = test_generalization(
            sae_model_path=sae_model_path,
            model_name=config['model']['name'],
            test_config=config['dataset']['test_dataset']
        )
        
        logger.info("✅ Phase 5 completed successfully")
        return generalization_results
        
    except Exception as e:
        logger.error(f"❌ Phase 5 failed: {e}")
        raise

def run_phase_6_analysis(config: Dict[str, Any], all_results: Dict[str, Any], logger: logging.Logger) -> None:
    """Phase 6: Comprehensive analysis and visualization."""
    logger.info("🚀 Starting Phase 6: Analysis and Visualization")
    
    try:
        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(all_results)
        
        # Create visualizations
        create_experiment_plots(
            results=all_results,
            metrics=metrics,
            output_dir=config['output']['plots_dir']
        )
        
        # Save results
        results_file = Path(config['output']['results_dir']) / f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info("✅ Phase 6 completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Phase 6 failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run rhyme probe experiments")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to config file")
    parser.add_argument("--phase", type=str, choices=["all", "1", "2", "3", "4", "5", "6"], default="all", help="Which phase to run")
    parser.add_argument("--experiment-name", type=str, help="Custom experiment name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set debug mode
    if args.debug:
        config['debug']['debug_mode'] = True
        config['debug']['small_scale']['enabled'] = True
    
    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{config['output']['experiment_name']}_{timestamp}"
    
    # Setup logging
    logger = setup_logging(config, experiment_name)
    logger.info(f"🎯 Starting Rhyme Probe Experiment: {experiment_name}")
    
    # Setup output directories
    create_output_directories(config)
    
    # Setup W&B
    if not args.debug:
        setup_wandb(config, experiment_name)
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU available: {torch.cuda.get_device_name()}")
    else:
        logger.warning("⚠️  No GPU available - experiments will be slow")
    
    try:
        all_results = {}
        
        # Phase 1: Dataset Creation
        if args.phase in ["all", "1"]:
            dataset_paths = run_phase_1_dataset_creation(config, logger)
            all_results['dataset_paths'] = dataset_paths
        
        # Phase 2: SAE Training
        if args.phase in ["all", "2"]:
            sae_model_path = run_phase_2_sae_training(config, all_results.get('dataset_paths', {}), logger)
            all_results['sae_model_path'] = sae_model_path
        
        # Phase 3: Feature Discovery
        if args.phase in ["all", "3"]:
            feature_results = run_phase_3_feature_discovery(config, all_results.get('sae_model_path', ''), logger)
            all_results['feature_results'] = feature_results
        
        # Phase 4: Causal Probing
        if args.phase in ["all", "4"]:
            probing_results = run_phase_4_causal_probing(
                config, 
                all_results.get('sae_model_path', ''), 
                all_results.get('feature_results', {}), 
                logger
            )
            all_results['probing_results'] = probing_results
        
        # Phase 5: Generalization
        if args.phase in ["all", "5"]:
            generalization_results = run_phase_5_generalization(config, all_results.get('sae_model_path', ''), logger)
            all_results['generalization_results'] = generalization_results
        
        # Phase 6: Analysis
        if args.phase in ["all", "6"]:
            run_phase_6_analysis(config, all_results, logger)
        
        logger.info("🎉 All phases completed successfully!")
        
        # Log final summary
        logger.info("📊 Experiment Summary:")
        logger.info(f"  - Experiment Name: {experiment_name}")
        logger.info(f"  - Output Directory: {config['output']['base_dir']}")
        logger.info(f"  - Results Directory: {config['output']['results_dir']}")
        logger.info(f"  - Plots Directory: {config['output']['plots_dir']}")
        
        if wandb.run:
            wandb.finish()
        
    except KeyboardInterrupt:
        logger.info("⚠️  Experiment interrupted by user")
        if wandb.run:
            wandb.finish()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        if wandb.run:
            wandb.finish()
        raise

if __name__ == "__main__":
    main() 