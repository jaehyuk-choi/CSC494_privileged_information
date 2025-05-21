# CSC494_privileged_information
# CSC494 – Leveraging LLMs to Generate Privileged Information for Clinical Predictive Tasks

This repository contains the implementation of a research project exploring how large language models (LLMs) can enhance clinical predictive models by generating auxiliary (privileged) information during training. This follows the Learning Using Privileged Information (LUPI) framework.

---

## 🧠 Project Summary

The goal is to improve clinical predictions in data-scarce settings using LLMs to generate useful side information during training — but not at inference time. Various architectures and learning paradigms (direct, multitask, multiview, pairwise similarity) are tested across different datasets and LLM configurations (8B, 70B).

---

## 📁 Directory Structure

```bash
CSC494_PRIVILEGED_INFORMATION/
├── data/                      # Data preprocessing modules for each learning pattern
│   ├── baseline_data.py
│   ├── direct_data.py
│   ├── multitask_data.py
│   ├── multiview_data.py
│   └── pairwise_data.py
│
├── model/                     # Models for each integration pattern
│   ├── direct_model/
│   │   ├── combined_multitask.py
│   │   ├── decoupled_residual.py
│   │   ├── decoupled_residual_no_decomp.py
│   │   ├── direct_pattern_no_decomp.py
│   │   └── direct_pattern_residual.py
│   ├── multitask_model/
│   │   └── multitask_models.py
│   ├── multiview_model/
│   │   └── multiview_models.py
│   └── pairwise_model/
│       ├── continuous.py
│       ├── soft.py
│       ├── baseline_model.py
│       └── multitask_mlp_simul_model.py
│
├── prompting/                 # LLM prompt results and pairs
│   ├── augmented_data_*.csv
│   ├── pairs.json
│   └── run_generation.slurm
│
├── img/                       # Figures and visualizations (if used in notebook/paper)
│
├── explore_data.py           # Initial EDA and dataset overview
├── utils.py                  # Shared utilities
│
├── direct.py                 # Training script: Direct pattern
├── multitask.py              # Training script: Multitask pattern
├── multiview.py              # Training script: Multiview pattern
├── pairwise.py               # Training script: Pairwise similarity pattern
├── ehr.py                    # Dataset wrapper for MIMIC
│
├── final_results.csv         # Consolidated evaluation results
├── multitask_results.csv
├── multiview_results.csv
├── multiview_multitask_results.csv
├── results_pairwise_Cont.csv
├── results_pairwise_Soft.csv
│
└── README.md
