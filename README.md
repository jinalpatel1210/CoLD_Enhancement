# CoLD++: Improving Collaborative Label Denoising Framework for Network Intrusion Detection using Label Correction

This repository contains **CoLD++**, an enhanced implementation of the **CoLD (Collaborative Label Denoising)** framework for learning under noisy labels in Network Intrusion Detection Systems (NIDS).

Unlike the original CoLD framework, which permanently removes samples identified as noisy, **CoLD++ introduces a selective label correction and recovery mechanism**. The proposed framework recovers reliable discarded samples using collaborative agreement and classifier-based verification, assigns corrected labels, and reintegrates them into the training set while preserving the original collaborative denoising pipeline.

---

# Framework Overview

<p align="center">
  <img src="figure/CoLD++_figure.png" width="900">
</p>

<p align="center">
<b>Figure 1.</b> Overview of the proposed CoLD++ framework.
</p>

---

# Main Enhancement

The proposed **CoLD++** framework extends the original **CoLD** framework by introducing a **multi-stage label recovery mechanism** that recovers reliable samples discarded during the original denoising process instead of permanently removing them.

As illustrated in Figure 1, the proposed framework consists of the following stages.

### 1. Original CoLD Denoising

The original CoLD framework is first executed without any modification. Multi-view clustering together with GMM-based noise detection separates the training data into:

- Kept (clean) samples
- Discarded (potentially noisy) samples

These kept samples are then used to train the downstream classifier.

---

### 2. Candidate Recovery using Agreement and Classifier Matching

Discarded samples are reconsidered through a two-stage verification process.

#### (a) Majority Agreement

Each discarded sample receives labels from all collaborative views. An agreement score is computed based on the number of views predicting the same label. Only samples whose agreement score exceeds a predefined threshold are selected as recovery candidates.

#### (b) Classifier Verification

The downstream classifier trained on the purified dataset predicts labels for all candidate samples.

#### (c) Label Matching

A candidate sample is recovered only when the classifier prediction matches the majority agreement label. This dual verification reduces the probability of recovering incorrectly labeled samples.

---

### 3. Top-k Sample Selection

Among the verified recovery candidates, only the top-*k* samples are selected for recovery. This limits the number of recovered samples while ensuring that only highly reliable candidates are added back to the training set.

---

### 4. Retraining and Evaluation

The recovered samples are merged with the original clean samples to construct the final training dataset.

The downstream classifier is retrained on this updated dataset and evaluated on the clean test set using Macro-F1 score.

This recovery strategy preserves informative samples that would otherwise be discarded by the original CoLD framework, thereby improving the robustness of learning under noisy-label conditions.

---

# Additional Experimental Analysis

Apart from the proposed CoLD++ framework, the repository also includes two additional experimental studies:

- **Diversified Sample Selection**
- **Adaptive Maximum Probability Threshold**

These experiments are provided for analysis purposes only and are **not part of the proposed CoLD++ framework**.

---

# Repository Structure

```text
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
│   └── utils.py
│
├── prepare_data.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Dataset

Experiments were conducted on the following dataset:

- **MALTLS-22**

The processed dataset should follow the same directory structure as the original CoLD implementation.

```text
data/
    MALTLS-22/
        X_train.npy
        y_train.npy
        X_test.npy
        y_test.npy
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/jinalpatel1210/CoLD_Enhancement.git
cd CoLD_Enhancement
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Code

Execute the complete pipeline using

```bash
python main.py
```

Configuration parameters can be modified through

```text
config/default.yaml
```

---

# Experimental Results

The proposed CoLD++ framework was evaluated under both **symmetric** and **asymmetric** label noise settings on the **MALTLS-22** dataset.

The selective label recovery strategy consistently improves or maintains the downstream Macro-F1 score by preserving informative training samples that would otherwise be discarded by the original CoLD framework.

The repository also includes two additional experimental analyses:

- Diversified Sample Selection
- Adaptive Maximum Probability Threshold

---

# Citation

If you use this repository, please cite the original CoLD paper.

```bibtex
@inproceedings{yang2026cold,
  title={CoLD: Collaborative Label Denoising Framework for Network Intrusion Detection},
  author={Yang, Shuo and others},
  booktitle={Proceedings of the Network and Distributed System Security Symposium (NDSS)},
  year={2026}
}
```

---

# Acknowledgements

This work is built upon the original **CoLD** framework proposed by the authors of the NDSS 2026 paper.

We sincerely acknowledge the original authors for making their implementation publicly available.

The proposed **CoLD++** framework extends the original work through a selective label recovery mechanism that performs label correction instead of permanently removing noisy samples.

---

# License

This project follows the same license as the original CoLD repository.
