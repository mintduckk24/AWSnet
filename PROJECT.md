# Project: AWSnet MyoPS380 Adaptation (`myops380_train304_test76_last_v1`)

## 1. Overview & Citation
This project adapts the official **AWSnet** (Auto-Weighted Supervision Attention Network) architecture to the **MyoPS380** standardized benchmark protocol (`myops380_train304_test76_last_v1`), strictly matching the reference benchmark established in **SCAR** (`https://github.com/thanhquan123hi1/SCAR.git`) and the requirements of `Promt_train_test_no_val.txt`.

- **Paper**: Wang, Kai-Ni, et al. *"AWSnet: An auto-weighted supervision attention network for myocardial scar and edema segmentation in multi-sequence cardiac magnetic resonance images."* Medical Image Analysis 77 (2022): 102362.
- **Model ID**: `awsnet`
- **Official Checkpoint Target**: `<save_dir>/awsnet_seed<seed>.pt` (e.g. `awsnet_seed1234.pt`)
- **Metric Protocol**: `myops380_voxel_v3`

---

## 2. Compatibility Audit & Architectural Adaptation

### A. Network Architecture (`AttU_Net`)
- **2D Slice Formulation**: MyoPS380 comprises 2D short-axis slices ($128 \times 128$) across 3 aligned CMR sequences: `bSSFP`, `LGE`, and `T2w`. `AttU_Net` is natively a 2D Attention U-Net with 5 encoder stages and 4 decoder stages with additive Attention Gates (`AttentionBlock`).
- **Input Channels Adaptation**: In the original challenge submission, the authors pre-multiplied images with an LV mask and passed only 2 modalities (DE/LGE and T2), commenting out the 3-modality input (`# input_img = torch.cat([img0, img1, img2], 1)`). For the standardized benchmark, the model interface is updated to accept all 3 aligned modalities (`bSSFP`, `LGE`, `T2w`) via `in_channels=3`, accepting both a concatenated tensor `(B, 3, 128, 128)` and separate tensors `(c0, lge, t2)` without information loss or external ROI priors.
- **Output Classes & Deep Supervision**: The original challenge code targeted 3 classes within an LV mask. Under the standardized benchmark, `out_channels=4` produces unnormalized logits for the 4 canonical classes:
  - `0`: Background
  - `1`: Normal Myocardium
  - `2`: Edema (edema-exclusive)
  - `3`: Scar
- **Bug Fix in Author Codebase**: In the upstream `ASAN_training/Network/Models.py`, `NestedUNet` referenced `conv_block_nested` which was never defined (causing a `NameError`). This block has been properly implemented to ensure syntax and import integrity.

### B. Loss Objective & Auto-Weighted Supervision (AWS) Analysis
- **Native Loss Formulation**: AWSnet's native objective is `LossPRSnet_dicece`, defined in `loss_deepsupvision.py`:
  $$\mathcal{L}_{\text{total}} = \sum_{i=1}^4 w_i \left(0.2 \times \mathcal{L}_{\text{CE}, i} + 1.0 \times \mathcal{L}_{\text{Dice}, i}\right)$$
  across 4 multi-scale decoder stages (1/8 resolution, 1/4 resolution, 1/2 resolution, and full resolution).
- **Auto-Weighting Adaptation under No-Val Protocol**:
  - In the original paper, the authors used REINFORCE policy gradients with a multi-path bandit (`Mixedpath(3)`) to sample 10 weight candidates per epoch, training 10 model copies, evaluating validation Dice on a validation split, computing reward $(v\_dice + 0.04)^3$, and selecting the best validation model.
  - However, `Promt_train_test_no_val.txt` **strictly prohibits** validation splits, validation loaders, and validation-based model selection (`no_val=True`, fixed 300 epochs budget, checkpoint selection strictly at epoch 300).
  - Therefore, for benchmark compliance without data leakage or artificial validation splits, the deep supervision weights are locked to the baseline's canonical default:
    $$w = [0.125, 0.25, 0.5, 1.0]$$
    (or normalized softmax $[0.2014, 0.2014, 0.2014, 0.3957]$).
  - The loss implementation is vectorized and device-agnostic, eliminating the upstream hardcoded `.cuda()` and `.cpu()` calls.

### C. Data Augmentation & Input Pipeline
- **Dataset**: Exactly 304 training patients (1,972 slices) from locked `train.txt`. Zero validation split (`val.txt` forbidden).
- **Augmentation**: SCAR's exact `RandomGenerator` (I-MMSeg implementation):
  1. Synchronized rot90/flip (50%), free rotation $\pm 20^\circ$ (25%), Gaussian noise $\sigma=0.2$ (12.5%).
  2. np.roll integer shift $\pm 5$px (50%).
  3. Shared brightness factor $\mathcal{U}(0.8, 1.2)$ clipped to 255 (50%).
  4. Shared gamma $\mathcal{U}(0.8, 1.2)$ power formula (50%).
  5. Random crop from top/left up to $H/5, W/5$ (50%).
  6. Cubic zoom resize for images, nearest zoom for masks if dimensions differ from $(128, 128)$.
  7. Ingestion of $[0.0, 1.0]$ preprocessed cache without double division by 255.0.
  8. Canonical label mapping: legacy cache $\{0:\text{bg}, 1:\text{myo}, 2:\text{scar}, 3:\text{edema}\} \to \text{canonical } \{0:\text{bg}, 1:\text{myo}, 2:\text{edema}, 3:\text{scar}\}$ via LUT `[0, 1, 3, 2]`.

### D. Training Budget & Optimization
- **Epoch Budget**: Exactly 300 epochs (`max_epochs = 300`), 124 batches per epoch (last batch has 4 samples), `drop_last = False`, exactly 37,200 total updates.
- **Effective Batch Size**: Target effective batch size 16 (`batch_size = 16, accum_steps = 1`). In VRAM-constrained environments, sample-weighted gradient accumulation is supported (`batch_size * accum_steps = 16`) by accumulating `loss * cur_batch_size` and dividing by actual `group_samples` at boundaries.
- **Precision Policy**: Standardized to SCAR's `amp=auto` protocol (CUDA prefers BF16, else FP16 with GradScaler; CPU auto disables AMP). Scaler unscaling and skipped step tracking (`amp_skipped_updates`, `successful_updates`) are strictly enforced.
- **Optimizer**: AdamW with `base_lr = 0.0003333`, `weight_decay = 0.0001`, `betas = (0.9, 0.999)`, `eps = 1e-8`.
- **Scheduler**: Step-level Polynomial LR Decay (`power = 0.9`, protocol `im_mseg_adamw_poly_v1`), advancing only on successful optimizer steps:
  $$\text{lr} = \text{base\_lr} \times \left(\max\left(0.0, 1.0 - \frac{\max(0, u - 1)}{N}\right)\right)^{0.9}$$
- **Checkpointing**: Atomic saving to `checkpoints/awsnet_seed<seed>.pt` at the end of each epoch. Completed epoch 300 is strictly required for official evaluation (`training_complete = True`). Checkpoints store model weights, optimizer/scheduler/scaler states, RNG states, update telemetry, and manifest hashes.

### E. Inference & Shared Evaluator
- **Test Set**: Exactly 76 test volumes from locked `test_vol.txt`.
- **Inference**: Single-pass forward in `eval()` mode slice-by-slice, assembled into 3D volume $(128, 128, D)$ with canonical classes $\{0, 1, 2, 3\}$. Zero morphological postprocessing (no LCC, no connected-component filtering).
- **Export Format**: `test_predictions/<CaseID>_pred.npz` with key `'prediction'` as uint8 array $(128, 128, D)$.
- **Shared Evaluator**: Directly executes SCAR's `surface_distance.py` under protocol `myops380_voxel_v3`, generating `metrics.json` and `per_case.csv` with equal patient weighting and standard empty-mask policy.

---

## 3. Directory Layout

```
AWSnet/
├── dataset/
│   ├── data_augmentation.py           # SCAR / I-MMSeg exact RandomGenerator & ResizeGenerator
│   ├── myops380_dataset.py            # MyoPS380TrainDataset & MyoPS380TestVolumeDataset
│   ├── splits/
│   │   └── myops380_no_val/           # Locked manifests (train.txt: 1972 slices, test_vol.txt: 76 cases)
│   └── __init__.py
├── models/
│   ├── awsnet.py                      # Canonical AWSnet (AttU_Net) model implementation
│   └── __init__.py
├── utils/
│   ├── criterions.py                  # Native LossPRSnet_dicece & MultiDiceLoss (vectorized & device-agnostic)
│   ├── lr_scheduler.py                # AdamW builder & Polynomial LR decay (im_mseg_adamw_poly_v1)
│   ├── paths.py                       # Robust dataset/split/checkpoint path resolution
│   ├── surface_distance.py            # Bundled SCAR evaluator (myops380_voxel_v3)
│   └── __init__.py
├── train.py                           # Production 300-epoch training engine (no-val protocol)
├── predict.py                         # Single-pass 3D volume inference engine
├── evaluate.py                        # Official shared evaluation engine
├── tests/
│   ├── test_awsnet.py                 # Architecture, parameter count, forward/backward tests
│   ├── test_edge_cases.py             # Empty masks, canonicalization, scheduler, manifest integrity
│   └── test_e2e.py                    # End-to-end training, resume, inference & evaluation pipeline
├── ASAN_training/                     # Upstream code with bug fixes applied
└── PROJECT.md                         # This technical specification
```

---

## 4. Usage Commands

### A. Run Training (300 Epochs, Zero Validation)
```bash
python train.py --batch_size 16 --accum_steps 1 --max_epochs 300 --base_lr 0.0003333 --seed 1234
```
For lower VRAM environments (maintaining effective batch size 16):
```bash
python train.py --batch_size 4 --accum_steps 4 --max_epochs 300 --base_lr 0.0003333 --seed 1234
```

### B. Resume Training
```bash
python train.py --resume checkpoints/awsnet_seed1234.pt --batch_size 16 --accum_steps 1 --max_epochs 300
```

### C. Run Inference (Generate Test Predictions)
```bash
python predict.py --checkpoint checkpoints/awsnet_seed1234.pt
```

### D. Run Shared Evaluation on Test Set (Epoch 300 Checkpoint)
```bash
python evaluate.py --checkpoint checkpoints/awsnet_seed1234.pt
```

### E. Run Full Test Suite
```bash
pytest tests
# Or:
python -m unittest discover -s tests
```
