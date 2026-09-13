# Batten Disease Prediction Using AI

AI-based research prototype for Batten Disease (Neuronal Ceroid Lipofuscinosis) prediction using clinical information and MRI image analysis.

## Overview

Batten Disease, also known as Neuronal Ceroid Lipofuscinosis (NCL), is a group of rare inherited neurodegenerative disorders.

This project develops an AI-based research prototype that analyzes:

- Clinical and patient-related information
- Brain MRI images

The system uses machine learning for clinical subtype prediction and deep learning for MRI-based Batten/Non-Batten classification.

The application also provides a Grad-CAM visualization to improve the interpretability of the MRI model's prediction.

> **Disclaimer:** This project is an academic/research prototype and is not a clinically validated diagnostic system. It must not be used as a substitute for professional medical diagnosis or treatment.

---

## Key Features

- Clinical data-based NCL subtype prediction
- MRI-based Batten/Non-Batten classification
- EfficientNet-B0 transfer learning
- Fine-tuning of the MRI model
- Grad-CAM-based MRI attention visualization
- Dynamic AI reasoning
- Patient information input
- Nutrition and daily-care guidance
- Streamlit-based interactive dashboard

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │      User Input      │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ Clinical Data    │        │    MRI Image     │
       │                  │        │                  │
       │ Gender           │        │ Upload MRI       │
       │ Phenotype        │        │                  │
       │ Age at Onset     │        └────────┬─────────┘
       │ Histology        │                 │
       │ Country          │                 ▼
       └────────┬─────────┘        ┌──────────────────┐
                │                  │ Preprocessing    │
                ▼                  │ 224 × 224 RGB   │
       ┌──────────────────┐        └────────┬─────────┘
       │ Clinical ML      │                 │
       │ Model            │                 ▼
       │                  │        ┌──────────────────┐
       │ Random Forest    │        │ EfficientNet-B0  │
       └────────┬─────────┘        │ MRI Classifier   │
                │                  └────────┬─────────┘
                ▼                           │
       ┌──────────────────┐                 ▼
       │ NCL Subtype      │        ┌──────────────────┐
       │ Prediction       │        │ Batten /        │
       │ CLN1 / CLN2 /    │        │ Non-Batten      │
       │ CLN3             │        │ Prediction      │
       └────────┬─────────┘        └────────┬─────────┘
                │                           │
                │                           ▼
                │                  ┌──────────────────┐
                │                  │    Grad-CAM      │
                │                  │ Attention Map    │
                │                  └────────┬─────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                   ┌──────────────────────┐
                   │   Final Assessment   │
                   │                      │
                   │ Prediction +         │
                   │ Clinical Context +  │
                   │ Explainability       │
                   └──────────────────────┘