# 3LC x HACKBLOX Scene Classification Challenge: Solution & Data-Centric AI Report

**Team:** HackBlox AI Contenders  
**Task:** 6-Class Natural Scene Classification (`buildings: 0`, `forest: 1`, `glacier: 2`, `mountain: 3`, `sea: 4`, `street: 5`)  
**Platform:** 3LC Data-Centric AI Platform + PyTorch (CUDA)  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)  

---

## 1. Executive Summary & Philosophy

In traditional machine learning competitions, performance improvements are sought by modifying model architectures or downloading massive external foundation model weights. The **3LC x HackBlox Challenge** constrains the model architecture to a **fixed standard ResNet-18 initialized from scratch (no pretrained weights)**, challenging participants to achieve high test accuracy purely through **Data-Centric AI engineering**.

Starting with an initial training set of only **600 seed labeled images** (100 per class) and an unlabeled pool of **6,000 images**, our objective was to systematically curate, active-label, and balance the training distribution up to the competition limit of **exactly 3,000 active rows with `weight = 1`** (500 per class).

---

## 2. Iterative Data-Centric Workflow (5-Stage Pipeline)

### Summary of Iterations & Table Lineage

| Phase / Iteration | Active Samples (`weight = 1`) | Unlabeled Pool (`weight = 0`) | Validation Accuracy | Strategy & Focus |
| :--- | :---: | :---: | :---: | :--- |
| **Phase 1: Seed Baseline (`train`)** | 600 (100 / class) | 6,000 | 58.20% | Baseline training on seed labels. Initial 3D UMAP embedding extraction in 3LC. |
| **Phase 2: Confident Clustering (`train_0000`)** | 1,538 | 5,062 | 70.92% (+12.72%) | Active learning on high-confidence (>0.80) clear clusters (forest, street, buildings). |
| **Phase 3: Boundary Disambiguation (`train_0001`)** | 2,850 | 3,750 | 78.25% (+20.05%) | Hard boundary disambiguation across `glacier`, `mountain`, and `sea` with cosine LR schedule. |
| **Phase 4: 8-View TTA Margin Curation (`train_0003`)** | **3,000 / 3,000** | 3,600 | **81.58%** (+23.38%) | 8-view multi-scale TTA pseudo-labeling with top-1/top-2 margin filtering. Exact 500 samples/class. |
| **Phase 5: Consensus Noise Pruning (`train_0004`)** | **3,000 / 3,000** | 3,600 | **83.75%** (+25.55%) | Multi-model consensus noise pruning of 360 lowest-confidence active rows, replaced with top-margin clean pool candidates. Multi-seed SWA ensemble. |

```
Dataset Lineage Graph (3LC Tables):
[Raw data/train + data/val]
       |
       v (register_tables.py)
[Table: train (Revision 0) - 600 labeled / 6000 undefined]
       |
       v (Phase 2: High Confidence Cluster Curation)
[Table: train_0000 (Revision 1) - 1538 labeled / 5062 undefined]
       |
       v (Phase 3: Class Rebalance & Boundary Curation)
[Table: train_0001 (Revision 2) - 2850 labeled / 3750 undefined]
       |
       v (Phase 4: 8-View TTA Margin-Filtered Curation)
[Table: train_0003 (Revision 3) - 3000 labeled / 3600 undefined]
       |
       v (Phase 5: Multi-Model Consensus Noise Pruning & Re-Curation)
[Table: train_0004 (Revision 4) - 3000 labeled / 3600 undefined]  <-- FINAL PRUNED & SANITIZED TABLE (Exact 500/class)
```

---

## 3. Key Data-Centric Insights from 3LC Embeddings

Using the **3LC Dashboard** and 3D UMAP feature projections, we analyzed the failure modes of the scratch ResNet-18:

1. **The Glacier vs Mountain Overlap:**
   * *Finding:* Mountain scenes with heavy snow pack strongly co-clustered with pure glaciers.
   * *Action:* Filtered the boundary cluster in 3LC, computed the top-1/top-2 softmax margin, and promoted only images with margin > 0.80.
2. **Sea vs Glacier Horizon Ambiguity:**
   * *Finding:* Images of sea horizons with white wave crests had lower initial confidence (<0.75).
   * *Action:* Sourced targeted sea images during Iteration 3 & 4, bringing total sea representation to an exact parity of 500 images.
3. **Consensus Noise Pruning:**
   * *Finding:* ~12% of initially labeled boundary images showed high multi-model disagreement.
   * *Action:* Pruned the 360 noisy samples, replacing them with highest-margin clean candidates from the undefined pool in `train_0004`.

---

## 4. Multi-Seed Model & Test-Time Augmentation (TTA) Ensemble

To maximize generalization on the hidden 1,800 test images, we combined multi-seed model ensembling (Seeds 42, 101, 777) with 8-view Test-Time Augmentation (24 forward passes per sample):
1. **Standard View:** Center-crop (224x224).
2. **Horizontal Flip View:** 224x224 flipped horizontally.
3. **Multi-Scale Zoom View 1 (240px):** Resized to 240x240 and center-cropped to 224x224.
4. **Multi-Scale Zoom Flip 1:** 240px zoomed + horizontally flipped.
5. **Multi-Scale Zoom View 2 (256px):** Resized to 256x256 and center-cropped to 224x224.
6. **Multi-Scale Zoom Flip 2:** 256px zoomed + horizontally flipped.
7. **Corner Aspect Crop View:** Random perspective crop resized to 224x224.
8. **Corner Aspect Crop Flip View:** Perspective crop + horizontal flip.

Softmax probabilities across all models and views were aggregated, producing calibrated confidence scores and high-precision class predictions in [`submission.csv`](file:///c:/Users/Aakarsh/Desktop/Hackblox/submission.csv).

---

## 5. Competition Rules Compliance Verification

- **Model Architecture:** Strictly ResNet-18 (`torchvision.models.resnet18(weights=None)` with only `resnet.fc = nn.Linear(512, 6)`).
- **Pretrained Weights:** None. Random initialization trained strictly from scratch.
- **External Data:** Zero external images used. Only the provided `data/train` and `data/val` sets.
- **Budget Cap:** Final active dataset contains **exactly 3,000 samples with `weight = 1`** (500 samples per class), fully complying with the <= 3,000 limit.
- **3LC Project:** Full table lineage and per-sample metrics packaged in `3LC_Intel-Scene_Project.zip`.

---

## 6. How to Reproduce

```bash
3lc service
python register_tables.py
python prune_and_recurate.py
python train_multiseed.py
python predict_ensemble.py
```
