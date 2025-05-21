# Leveraging LLMs to Generate Privileged Information for Clinical Predictive Tasks

This repository accompanies the research project **"Leveraging LLMs to Generate Privileged Information for Clinical Predictive Tasks"** by **Jaehyuk Choi (April 2025)**. The project explores how large language models (LLMs) can enhance predictive modeling in healthcare by generating synthetic, context-rich auxiliary data used only during training.

---

## 📌 Overview

Predictive modeling in clinical settings is often limited by small, imbalanced datasets. Rather than using LLMs as direct predictors, this project investigates their utility in generating **privileged information** — synthetic features used exclusively at training time — inspired by the **LUPI framework** (Vapnik) and follow-up works (e.g., Jonschkowski et al., 2016).

---

## ⚙️ Methodology

1. **Synthetic Data Generation**  
   LLMs (LLaMA-8B, LLaMA-70B) are prompted to simulate clinical reasoning and generate side information.

2. **Pattern-Based Integration**  
   LLM-generated data is integrated through four training paradigms:
   - **Direct**
   - **Multi-task**
   - **Multi-view**
   - **Pairwise similarity**

3. **Training & Evaluation**  
   Models are trained using various strategies including:
   - Simultaneous training
   - Decoupled optimization
   - Pretrain → Finetune

---

## 🧪 Datasets

- **UCI Diabetes Dataset**  
  Predicts diabetes onset from patient lifestyle and demographic factors.

- **MIMIC-IV Dataset**  
  Predicts in-hospital mortality using ICU time-series data.

---

## 🔬 Learning Patterns

| Pattern     | Description |
|-------------|-------------|
| **Direct**  | LLM-generated scalar is directly concatenated or passed through a decompression layer. |
| **Multi-task** | Shared encoder with auxiliary supervision heads. |
| **Multi-view** | Dual encoders for original and privileged views. |
| **Pairwise Similarity** | Trains on LLM-derived similarity scores between instances. |

- Hybrid strategies (e.g., Direct + Multi-task) are also implemented.

---
