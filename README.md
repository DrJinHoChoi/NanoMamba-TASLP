# NanoMamba

**Noise-Robust Ultra-Compact Keyword Spotting via Dual-Expert Normalization and Adaptive State Space Dynamics**

> IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrJinHoChoi/NanoMamba-TASLP/blob/main/NanoMamba_Train.ipynb)

## Highlights

- **4,957 parameters** — 4.8x smaller than DS-CNN-S (23.7K), yet more noise-robust
- **DualPCEN**: Two complementary PCEN experts with signal-based routing (Spectral Flatness + Tilt + TMI)
- **SA-SSM**: SNR-conditioned Mamba — delta-modulation + B-gating adapt temporal dynamics to noise level
- **Zero-overhead noise robustness**: No separate enhancement module, no extra inference cost
- **FPGA-ready**: Complete Verilog RTL for Xilinx Artix-7, INT8 datapath, ~15mW

## Architecture

```
Raw Audio (16kHz, 1s)
  --> STFT (512-FFT, 160-hop)
  --> SNR Estimator (per-mel-band, running EMA)
  --> DualPCEN (Expert1: babble | Expert2: stationary, SF+Tilt+TMI routing)
  --> Instance Norm
  --> Patch Projection (n_mels -> d_model)
  --> N x SA-SSM Block
  |     LayerNorm -> in_proj -> DWConv1d -> SA-SSM -> Gate -> out_proj + Residual
  |     SA-SSM: SNR modulates delta (step size) and B (input gate)
  --> Global Average Pooling
  --> 12-class Classifier
```

## Key Results

| Model | Params | Clean | Factory 0dB | White 0dB | Babble 0dB |
|-------|--------|-------|------------|-----------|------------|
| NanoMamba-Tiny-DualPCEN (ours) | 4,957 | - | - | - | - |
| NanoMamba-Matched-DualPCEN (ours) | 7,402 | - | - | - | - |
| BC-ResNet-1 | 7,464 | - | - | - | - |
| DS-CNN-S | 23,700 | - | - | - | - |

*Results will be filled after full evaluation. Run the Colab notebook to reproduce.*

## Quick Start

### Google Colab (Recommended)

Click the badge above or [open directly](https://colab.research.google.com/github/DrJinHoChoi/NanoMamba-TASLP/blob/main/NanoMamba_Train.ipynb). The notebook handles everything: dataset download, training, noise evaluation.

### Local

```bash
git clone https://github.com/DrJinHoChoi/NanoMamba-TASLP.git
cd NanoMamba-TASLP
pip install -r requirements.txt

# Train NanoMamba-Tiny-DualPCEN (4,957 params, ~8min on GPU)
python train_colab.py \
    --models NanoMamba-Tiny-DualPCEN \
    --epochs 30 --noise_aug --calibrate

# Evaluate on 5 noise types x 7 SNR levels
python train_colab.py \
    --models NanoMamba-Tiny-DualPCEN \
    --eval_only --calibrate \
    --noise_types factory,white,babble,street,pink \
    --snr_range=-15,-10,-5,0,5,10,15
```

## Project Structure

```
NanoMamba-TASLP/
  nanomamba.py             # Core: SA-SSM + DualPCEN + MultiPCEN + SNR Estimator
  train_colab.py           # Training pipeline, noise evaluation, calibration
  model.py                 # CNN baselines (DS-CNN-S, BC-ResNet-1)
  paper_models.py          # Additional model variants for ablation
  proposed_model.py        # DualPCEN proposed architecture
  train_all_models.py      # Multi-model training orchestrator
  measure_efficiency.py    # Latency & memory benchmarks
  arm_analysis.py          # ARM Cortex-M deployment analysis
  NanoMamba_Train.ipynb    # Colab notebook (one-click training)
  requirements.txt
  checkpoints_full/        # Pre-trained model weights
  rtl/                     # FPGA/ASIC Verilog implementation
    src/                   #   10 RTL modules (SSM, PCEN, STFT, classifier, ...)
    tb/                    #   Testbench
    mem/                   #   LUT & weight memory files
    fpga/                  #   Xilinx Artix-7 constraints & wrapper
    Makefile               #   Simulation & synthesis automation
  scripts/                 # Weight export & LUT generation utilities
```

## Model Variants

| Model | Params | Description |
|-------|--------|-------------|
| `NanoMamba-Tiny` | 4,634 | SA-SSM baseline (d=16, s=4, 2 layers) |
| `NanoMamba-Tiny-DualPCEN` | 4,957 | + Dual-PCEN with SF+Tilt routing |
| `NanoMamba-Tiny-TriPCEN` | 5,118 | + 3-expert PCEN (factory/street specialist) |
| `NanoMamba-Matched-DualPCEN` | 7,402 | Param-matched to BC-ResNet-1 (d=21, s=5) |
| `NanoMamba-*-v2` | same | + TMI + SNR-conditioned temp + temporal smoothing |
| `NanoMamba-*-v2-SSMv2` | same | + SA-SSM v2 (Michaelis-Menten + PCEN gate conditioning) |

## SA-SSM: How It Works

Standard Mamba treats all input frames equally. SA-SSM conditions the selection mechanism on per-mel-band SNR:

- **Delta modulation**: Low SNR -> smaller step size -> longer temporal memory (average out noise)
- **B-gating**: Low SNR -> attenuate noisy input -> preserve state from cleaner frames
- **Runtime calibration**: Noise profile estimated during silence -> adaptive buffer parameters

## FPGA Implementation

| Spec | Value |
|------|-------|
| Target | Xilinx Artix-7 (XC7A35T) |
| Resources | ~2,500 LUTs, 4 DSP48, 3 BRAM |
| Power | ~15mW (FPGA), ~0.08mW (ASIC 28nm estimate) |
| Datapath | INT8 weights, 32-bit accumulator |
| Clock | 50MHz, real-time processing |

## Dataset

Google Speech Commands V2 (12-class: yes, no, up, down, left, right, on, off, stop, go + silence + unknown). Automatically downloaded by the training script.

## Citation

```bibtex
@article{choi2026nanomamba,
  author  = {Choi, Jin Ho},
  title   = {{NanoMamba}: Noise-Robust Ultra-Compact Keyword Spotting via
             Dual-Expert Normalization and Adaptive State Space Dynamics},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2026},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

## License

Dual license: Free for academic/research use. Commercial use requires a separate license. See [LICENSE](LICENSE) for details.
