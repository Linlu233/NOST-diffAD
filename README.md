# NOST-DiffAD

Implementation of **Normality-Oriented Semantic-Topology Score Diffusion for Visual Anomaly Detection** from `NOST-DiffAD.md`.

The code follows the document formulas for:

- semantic-spatial patch graph
- graph Laplacian
- attribute and structure branches
- gated dual-branch fusion
- normality score diffusion and score matching
- prototype compactness
- differentiable NMF constraint
- Laplacian graph smoothing
- patch energy terms
- conformal thresholding
- image and pixel metrics

## Setup

```bash
conda env create -f environment.yml
conda activate nostdiffad
```

If conda solving is slow for CUDA packages, use the verified pip-wheel route:

```bash
conda create -y -n nostdiffad python=3.10 pip
conda run -n nostdiffad python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
conda run -n nostdiffad python -m pip install pyyaml scikit-learn scipy tqdm pytest timm transformers -e .
conda activate nostdiffad
```

If the environment already exists:

```bash
pip install -e .
```

For automatic SAM part-mask generation:

```bash
pip install -e ".[masks]"
```

The default DINOv2 extractor uses Meta's official `facebookresearch/dinov2`
torch.hub release (`model.feature_extractor=dinov2_torchhub`). The Hugging Face
extractor remains available with `model.feature_extractor=dinov2`.

## Data Layout

The default loader expects MVTec-style folders:

```text
DATA_ROOT/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*_mask.png
```

For the graph term `beta_m I[m_i=m_j]`, provide SAM/SAM2 part mask ids separately:

```text
PART_MASK_ROOT/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect>/*.png
```

Generate SAM part masks from the image data:

```bash
python scripts/generate_part_masks.py \
  --data-root datasets/mvtec_ad \
  --output-root datasets/part_masks/sam_vit_b_mvtec_ad \
  --device cuda
```

Set `data.part_mask_root=/path/to/part_masks`. These masks are semantic part ids, not anomaly ground-truth masks. If part masks are unavailable, run with `--set graph.use_mask_topology=false graph.beta_m=0.0`.

Other industrial datasets can be run if converted to this layout, or by adding a dataset adapter in `src/nostdiffad/data.py`.
The loader treats `good` and `ok` directories as normal data, so the downloaded BTAD transformed layout can be used directly:

```bash
python scripts/generate_part_masks.py \
  --data-root datasets/btad/BTech_Dataset_transformed \
  --output-root datasets/part_masks/sam_vit_b_btad \
  --category 01 \
  --device cuda

python scripts/train.py --config configs/default.yaml \
  --set data.root=datasets/btad/BTech_Dataset_transformed \
  data.category='"01"' \
  data.part_mask_root=datasets/part_masks/sam_vit_b_btad
```

Official/original public dataset sources that can be fetched automatically:

```bash
python scripts/download_datasets.py --datasets visa btad kolektorsdd2 --root datasets --keep-archive
```

On mainland China servers, the official VisA AWS S3 link can be slow. The downloader uses `aria2c` for resumable multi-connection downloads when available; an archive with a sibling `.aria2` file is still incomplete.

Datasets with gated, evaluation-server, form, or browser-session access still require user-side access material: MVTec AD 2, MVTec LOCO AD, MVTec 3D-AD, MPDD, Real-IAD, and DAGM.

## Train

```bash
  python scripts/train.py --config configs/default.yaml \
  --set data.root=datasets/mvtec_ad data.category=bottle train.epochs=50
```

## Evaluate

```bash
  python scripts/evaluate.py --config configs/default.yaml \
  --set data.root=datasets/mvtec_ad data.category=bottle \
  eval.checkpoint=outputs/checkpoints/bottle_best.pt
```

## Experiment Matrix

Generate the commands required by the document's ablation, few-shot, robustness, and cross-category design:

```bash
python scripts/run_experiment_matrix.py --config configs/default.yaml --matrix configs/experiment_matrix.yaml --data-root datasets/mvtec_ad
python scripts/run_experiment_matrix.py --config configs/default.yaml --matrix configs/experiment_matrix.yaml --data-root datasets/mvtec_ad --cross-category
```

Tune dataset-level hyperparameters on a held-out tuning half of the test split before launching the formal
full-shot/no-robustness NOST-DiffAD pass. The queue generates missing SAM part masks first, runs each tuning
trial with early stopping, then writes `outputs/hparam_tuning/best_configs/<dataset>.yaml`:

```bash
nohup setsid /root/miniconda3/envs/nostdiffad/bin/python -u scripts/run_parallel_hparam_tuning.py \
  --output-root outputs/hparam_tuning \
  --max-parallel 2 \
  --skip-running \
  > outputs/logs/hparam_tuning_parallel.log 2>&1 < /dev/null &
echo $! > outputs/logs/hparam_tuning_parallel.pid
tail -f outputs/logs/hparam_tuning_parallel.log
```

Per-trial logs are written under `outputs/logs/hparam_tuning_trials/`. Increase `--max-parallel` only after
checking GPU memory and utilization. `--skip-running` lets a restarted scheduler avoid tasks already active in
another process. For multi-process scheduling on one GPU, add `--max-gpu-memory-used-mb`,
`--max-gpu-processes`, and `--launch-delay-seconds` so new trials are launched only while the GPU remains under
the chosen memory and process-count budget; use `--no-summary` for supplemental schedulers that share the same
result root.

After tuning finishes, run the official pass on the final held-out half of each test split. This uses the tuned
dataset configs, default 300 epoch budget, and early stopping:

```bash
python scripts/run_downloaded_official_experiments.py \
  --best-config-root outputs/hparam_tuning/best_configs \
  --write-bash outputs/logs/official_experiments_queue.sh \
  --skip-finished
nohup setsid bash outputs/logs/official_experiments_queue.sh > outputs/logs/official_experiments.log 2>&1 < /dev/null &
tail -f outputs/logs/official_experiments.log
```

KolektorSDD2's original flat split can be converted without modifying the archive:

```bash
python scripts/prepare_kolektorsdd2_mvtec.py \
  --source-root datasets/kolektorsdd2 \
  --output-root datasets/kolektorsdd2_mvtec \
  --category kolektorsdd2
```

Additional downloaded datasets that are not already in MVTec-style layout can be exposed as symlink views:

```bash
python scripts/prepare_dataset_views.py --datasets-root datasets
```

This creates `datasets/mvtec_ad_2_mvtec`, `datasets/mvtec_3d_rgb_mvtec`, `datasets/realiad_1024_mvtec`, and `datasets/visa_mvtec`
without copying image data. `scripts/run_parallel_hparam_tuning.py` and
`scripts/run_downloaded_official_experiments.py` will include MPDD, MVTec LOCO, MVTec AD 2, MVTec 3D RGB,
RealiAD, and VisA once their expected roots are present.

For a broad first-stage search over all currently available datasets, use a reduced high-value trial set before
expanding weak categories:

```bash
screen -dmS hparam_all_stage1 bash -lc 'cd /root/autodl-tmp/data && export PYTHONPATH=src:scripts && \
/root/miniconda3/envs/nostdiffad/bin/python -u scripts/run_parallel_hparam_tuning.py \
  --output-root outputs/hparam_all_datasets_stage1 \
  --log-root outputs/logs/hparam_all_datasets_stage1_trials \
  --epochs 80 --patience 15 --min-epochs 25 \
  --selection-metric selection_score_pro \
  --max-parallel 2 \
  --trial-names proto_only proto_only_topk001 proto_only_topk010 score_only score_proto_only \
    score_proto_topo_tiny topo_tiny graph_sigma25 lr3e-4 \
  --skip-running > outputs/logs/hparam_all_datasets_stage1.log 2>&1'
```

After the non-VisA stage-1 queue finishes, continue automatically with VisA tuning and its final-split official
run:

```bash
screen -dmS visa_after_stage1 bash -lc 'cd /root/autodl-tmp/data && ./scripts/wait_then_run_visa.sh > outputs/logs/visa_after_stage1.log 2>&1'
```

This repository does not include the listed benchmark datasets or external baseline implementations, so full benchmark execution requires providing dataset paths and baseline code/results.

## Formula Trace

See `FORMULA_TRACE.md` for the mapping from document equations to implementation files.
