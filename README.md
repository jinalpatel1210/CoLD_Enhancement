# CoLD++: Improving Collaborative Label Denoising Framework for Network Intrusion Detection using Label Correction

This repository contains **CoLD++**, an enhanced implementation of the **CoLD (Collaborative Label Denoising)** framework for learning under noisy labels in Network Intrusion Detection Systems (NIDS).

The repository extends the original CoLD framework by introducing additional label recovery strategies that improve downstream classification performance while preserving the original denoising pipeline.

---

## Framework Overview

<p align="center">
  <img src="figures/CoLD++_figure.png" width="950">
</p>

<p align="center">
<b>Figure 1.</b> Overview of the proposed CoLD++ framework.
</p>

---

## Features

The proposed CoLD++ framework introduces two lightweight enhancements to the original CoLD recovery stage:

### 1. Diversified Sample Selection

Instead of randomly selecting recovered samples, CoLD++ performs diversity-aware selection in the latent feature space.

- Preserves the original recovery budget.
- Improves feature-space coverage.
- Produces a more representative recovered subset.
- Provides consistent improvements over random recovery.

---

### 2. Adaptive Maximum Probability Threshold

A confidence-based recovery strategy where the recovery threshold varies according to the estimated noise level.

- Low noise → lower confidence threshold.
- High noise → stricter confidence threshold.
- Prevents recovery of unreliable samples.
- Improves robustness of the recovery stage.

---

## Repository Structure

```
CoLD_Enhancement/
│
├── config/
│   └── default.yaml
│
├── figures/
│   └── CoLD++_figure.png
│
├── scripts/
│   ├── classify.py
│   ├── clustering.py
│   ├── config.py
│   ├── data.py
│   ├── denoise.py
│   ├── encoder.py
│   ├── feature_reorder.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── pipeline.py
│   ├── recovery.py
│   ├── reliability.py
│   ├── utils.py
│    
├── prepare_data.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Dataset

Experiments were conducted on:

- **MALTLS-22**

The dataset should be placed under

```
data/
```

following the directory structure used by the original CoLD implementation.

---

## Installation

Clone the repository

```bash
git clone https://github.com/jinalpatel1210/CoLD_Enhancement.git

cd CoLD_Enhancement
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Code

Run the complete pipeline

```bash
python main.py
```

Configuration parameters can be modified through

```
config/default.yaml
```

---

## Experimental Results

The proposed enhancements were evaluated under both **symmetric** and **asymmetric** label noise.

The additional experiments investigate:

- Diversified sample selection
- Adaptive maximum probability threshold

Both methods consistently improve or maintain the performance of the original CoLD framework while introducing only minimal computational overhead.

---

## Citation

If you use this repository, please cite the original CoLD paper:

```bibtex
@inproceedings{yang2026cold,
  title={CoLD: Collaborative Label Denoising Framework for Network Intrusion Detection},
  author={Yang, Shuo and others},
  booktitle={Proceedings of the Network and Distributed System Security Symposium (NDSS)},
  year={2026}
}
```

---

## Acknowledgements

This work is based on the original **CoLD** framework proposed by the authors of the NDSS 2026 paper.

The enhancements implemented in this repository focus on improving the label recovery stage while preserving the original collaborative denoising pipeline.
