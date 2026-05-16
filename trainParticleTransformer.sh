#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=18G
#SBATCH -o slurm-%x-%j-%N.out

cd /home/joosep/tau/ml-tau-reco
uv run python3 src/trainParticleTransformer.py \
    max_num_files=-1 \
    num_dataloader_workers=4 \
    cache_size=4 \
    hydra.run.dir=.

echo "Training finished. Running builder inference..."
uv run python3 src/ParticleTransformerTauBuilder.py \
    --model ParticleTransformer_model_best.pt \
    --transform ParticleTransformer_FeatureStandardization.json \
    --input "data/test_dataset_*.pt" \
    --batch_size 1024 \
    --output inference_results.pt

echo "Builder inference finished. Generating ROC plot..."
uv run python3 src/plot_roc_ParticleTransformer.py \
    --input inference_results.pt \
    --output roc_particle_transformer.png
