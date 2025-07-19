# 🎯 Pre-Experiment Checklist

## ✅ **Authentication & Access**

### Hugging Face Setup
- [ ] **Accept Gemma model terms**: https://huggingface.co/google/gemma-2b
- [ ] **Create HF token**: https://huggingface.co/settings/tokens
- [ ] **Test authentication locally**:
  ```bash
  python setup_huggingface_auth.py
  ```

### Weights & Biases Setup
- [ ] **Create W&B account**: https://wandb.ai
- [ ] **Get API key**: https://wandb.ai/settings
- [ ] **Update config**: Set `entity` in `configs/experiment_config.yaml`

## ✅ **Local Environment Testing**

### Dependencies
- [ ] **Install all packages**: `pip install -r requirements.txt`
- [ ] **Test imports**: `python test_setup.py`
- [ ] **Test rhyme detection**: `python src/metrics/rhyme_metrics.py`

### Configuration
- [ ] **Review experiment config**: `configs/experiment_config.yaml`
- [ ] **Set experiment parameters**:
  - Model size (start with gemma-2b for testing)
  - Dataset sizes (use small scale for initial tests)
  - SAE dimensions
  - Training epochs

## ✅ **GCP Instance Preparation**

### Instance Setup
- [ ] **Use Deep Learning VM template**: "gemma" with PyTorch
- [ ] **Instance specs**:
  - Machine type: a2-highgpu-1g
  - GPU: NVIDIA A100
  - Boot disk: 100GB SSD
  - Region: europe-west4 (Netherlands)

### File Transfer
- [ ] **Copy project files to instance**:
  ```bash
  gcloud compute scp --recurse . instance-name:~/gemma-rhyme-probe
  ```
- [ ] **Test file transfer**: Verify all files are present

### Environment Setup
- [ ] **Run setup script on instance**:
  ```bash
  chmod +x setup_deeplearning_vm.sh
  ./setup_deeplearning_vm.sh
  ```
- [ ] **Verify GPU access**: `nvidia-smi`
- [ ] **Test model loading**: Load Gemma model successfully

## ✅ **Experiment Planning**

### Dataset Strategy
- [ ] **Start small**: Use debug mode for initial tests
- [ ] **Dataset sizes**:
  - Debug: 100 samples each
  - Small: 1,000 samples each
  - Full: 10,000 samples each
- [ ] **Rhyme patterns**: AABB, ABAB, ABBA, AAAA
- [ ] **Rhyme types**: perfect, slant, assonance

### SAE Training Strategy
- [ ] **Architecture decisions**:
  - d_model: 2048 (Gemma-2b hidden size)
  - d_sae: 8192 (4x expansion)
  - L1 coefficient: 1e-3
- [ ] **Training parameters**:
  - Batch size: 32
  - Learning rate: 1e-4
  - Max epochs: 100
  - Early stopping patience: 5

### Causal Probing Strategy
- [ ] **Intervention methods**:
  - Activation patching
  - Feature ablation
- [ ] **Layers to probe**: 0-23 (all layers)
- [ ] **Heads to probe**: 0-7 (all heads)
- [ ] **Statistical significance**: α = 0.05

## ✅ **Monitoring & Logging**

### Experiment Tracking
- [ ] **W&B project**: "rhyme-probe-experiments"
- [ ] **Logging strategy**:
  - Model checkpoints
  - Training metrics
  - Feature activations
  - Causal probing results
- [ ] **Local backups**: Save results locally as well

### Error Handling
- [ ] **Graceful failures**: Handle model loading errors
- [ ] **Checkpointing**: Save progress regularly
- [ ] **Memory monitoring**: Watch GPU memory usage
- [ ] **Timeout handling**: Set reasonable timeouts

## ✅ **Validation & Testing**

### Small-Scale Tests
- [ ] **Debug mode test**: Run with minimal data
- [ ] **Single phase test**: Test each phase individually
- [ ] **Memory test**: Verify GPU memory is sufficient
- [ ] **Time estimation**: Estimate full experiment duration

### Quality Checks
- [ ] **Dataset quality**: Verify rhyme detection works
- [ ] **Model loading**: Test Gemma model access
- [ ] **SAE training**: Test small SAE training
- [ ] **Feature extraction**: Test feature discovery

## ✅ **Backup & Recovery**

### Data Backup
- [ ] **Local backup**: Keep copy of all code locally
- [ ] **Cloud backup**: Use GCS for large files
- [ ] **Version control**: Commit all changes to git

### Recovery Plan
- [ ] **Instance restart**: Know how to restart if needed
- [ ] **Checkpoint recovery**: Resume from checkpoints
- [ ] **Data recovery**: Recreate datasets if needed

## ✅ **Cost Management**

### Budget Planning
- [ ] **Cost estimation**: ~$2-3/hour for A100
- [ ] **Time estimation**: 8-24 hours for full experiment
- [ ] **Budget limit**: Set maximum spend limit
- [ ] **Monitoring**: Check costs regularly

### Optimization
- [ ] **Efficient training**: Use mixed precision
- [ ] **Memory optimization**: Gradient checkpointing
- [ ] **Early stopping**: Stop when converged
- [ ] **Resource monitoring**: Watch usage patterns

## 🚀 **Ready to Launch Checklist**

### Final Verification
- [ ] **All dependencies installed**
- [ ] **Authentication working**
- [ ] **Config files reviewed**
- [ ] **Small-scale test passed**
- [ ] **GPU instance ready**
- [ ] **Monitoring set up**
- [ ] **Backup strategy in place**

### Launch Commands
```bash
# 1. Start GPU instance
gcloud compute instances start instance-name --zone=europe-west4-a

# 2. SSH into instance
gcloud compute ssh instance-name --zone=europe-west4-a

# 3. Navigate to project
cd ~/gemma-rhyme-probe

# 4. Run debug test first
python run_experiments.py --debug --phase 1

# 5. Run full experiment
python run_experiments.py --experiment-name "rhyme_probe_v1_full"
```

## 📊 **Success Metrics**

### Technical Success
- [ ] **SAE training converges**
- [ ] **Rhyme features discovered**
- [ ] **Causal probing shows significance**
- [ ] **Generalization works**

### Research Success
- [ ] **Clear rhyme-related features found**
- [ ] **Causal relationships established**
- [ ] **Novel insights discovered**
- [ ] **Results are reproducible**

---

**🎯 You're ready to start your rhyme probe experiments!**

Remember:
- Start with debug mode
- Monitor costs and resources
- Save checkpoints frequently
- Document everything
- Have fun exploring! 🚀 