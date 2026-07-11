# SDRL

PyTorch implementation of **Unsupervised Person Re-Identification with Diffusion Model via Semantic-Aware Disentanglement Representation Learning** (IEEE TCSVT).

## About

Unsupervised person re-identification (Re-ID) learns discriminative embeddings without manual identity labels. A major challenge is that appearance changes caused by **background clutter** and **inconsistent part semantics** across views can mislead clustering and pseudo-label assignment. SDRL addresses this by explicitly disentangling person semantics from background interference and enforcing consistency between global and local representations.

### Framework

SDRL builds on a part-based backbone and introduces four core components:

- **DAM (Disentanglement Aggregation Model)** — Separates each image into a person region and a background. Backgrounds are inpainted with LAMA; during training, persons from two images are composited onto each other's backgrounds to synthesize cross-camera enhanced views, enriching multi-view supervision without extra labels.

- **MSC (Multi-view Similarity Consistency)** — Aligns features between the original source view and the DAM-enhanced view. A distribution constraint (MMD on cross-/intra-camera distance statistics) and an instance constraint (InfoNCE over same-identity neighbors) jointly encourage identity-preserving, camera-invariant representations.

- **SSDM (Semantic Spatial Diffusion Model)** — Uses an STN to propose initial part geometry, then refines spatial transformer parameters with a diffusion model (cosine schedule + DDIM). Mask-conditioned features `U = F(x) ⊗ m` focus learning on person semantics. The model generates semantic part patches for fine-grained recognition.

- **SDC (Semantic Decoupled Contrastive loss)** — Applies decoupled contrastive learning on part features from consecutive diffusion steps, coupling `(θ_{t-1}, θ_t)` as positives while keeping negatives across other parts and timesteps. This stabilizes semantic part learning during diffusion training.

### Semantic consistency & pseudo-label refinement

SDRL measures **semantic consistency** between global and part-level neighborhood structures (via MMD over k-NN distance distributions). High-consistency parts receive stronger supervision in `L_ces`, which refines cluster pseudo-labels by ensembling part predictions—suppressing unreliable local semantics while preserving discriminative global structure.

### Objective

The overall training objective is:

`L_SDRL = L_ces + L_tri + λ_msc · L_msc + λ_sdc · L_sdc`

where `L_ces` is the refined cross-entropy loss, `L_tri` is a triplet loss on global features, and auxiliary SSDM diffusion loss is used during warm-up scheduling in practice.

## Install

```shell
git clone https://github.com/taoxuefong/SDRL-reid
cd SDRL-main
python setup.py develop
```

## Data

Place Re-ID datasets (e.g. Market-1501, MSMT17) under your data directory. If using DAM, also prepare per-image person masks and run `examples/precompute_dam_backgrounds.py` to generate inpainted backgrounds before training.

## Train

Run `examples/train_sdrl.py`. See `--help` for dataset, DAM, MSC, and SSDM options.

## Test

Run `examples/test.py` with a trained checkpoint. See `--help` for details.

## Citation

```bibtex
@article{tao2025sdrl,
  title={Unsupervised Person Re-Identification with Diffusion Model via Semantic-Aware Disentanglement Representation Learning},
  author={Tao, Xuefeng and Kong, Jun and Jiang, Min and Li, Jiayi and Mian, Ajmal},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```
