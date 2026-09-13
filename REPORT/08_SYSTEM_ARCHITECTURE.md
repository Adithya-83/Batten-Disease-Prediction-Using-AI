\# 8. SYSTEM ARCHITECTURE



\## 8.1 Overview



The proposed Batten Disease Prediction Using Artificial Intelligence system follows a modular architecture that combines clinical-data analysis, MRI-based image classification, explainable AI, and a web-based user interface.



The system is designed as a research prototype for supporting the analysis of information associated with Batten disease and neuronal ceroid lipofuscinosis (NCL). It does not replace professional medical diagnosis.



The architecture consists of the following major components:



1\. Patient Information Input

2\. Clinical Data Processing

3\. Clinical Machine Learning Model

4\. MRI Image Input

5\. MRI Image Preprocessing

6\. Deep Learning MRI Classifier

7\. Grad-CAM Explainability Module

8\. Assessment and Reasoning Module

9\. Supportive Nutrition and Daily Care Module

10\. Streamlit User Interface



---



\## 8.2 High-Level Architecture



The overall system architecture can be represented as:



```text

&nbsp;                 ┌──────────────────────────┐

&nbsp;                 │      User / Researcher   │

&nbsp;                 └────────────┬─────────────┘

&nbsp;                              │

&nbsp;                              ▼

&nbsp;                 ┌──────────────────────────┐

&nbsp;                 │   Streamlit Web Interface │

&nbsp;                 └────────────┬─────────────┘

&nbsp;                              │

&nbsp;               ┌──────────────┴──────────────┐

&nbsp;               │                             │

&nbsp;               ▼                             ▼

&nbsp;     ┌───────────────────┐         ┌───────────────────┐

&nbsp;     │ Patient Information│         │    MRI Upload     │

&nbsp;     │      Input         │         │      Input        │

&nbsp;     └─────────┬─────────┘         └─────────┬─────────┘

&nbsp;               │                             │

&nbsp;               ▼                             ▼

&nbsp;     ┌───────────────────┐         ┌───────────────────┐

&nbsp;     │ Clinical Data     │         │ MRI Preprocessing │

&nbsp;     │ Processing        │         │ 224 × 224 RGB     │

&nbsp;     └─────────┬─────────┘         └─────────┬─────────┘

&nbsp;               │                             │

&nbsp;               ▼                             ▼

&nbsp;     ┌───────────────────┐         ┌───────────────────┐

&nbsp;     │ Random Forest     │         │ EfficientNet-B0   │

&nbsp;     │ Clinical Model    │         │ MRI Classifier    │

&nbsp;     └─────────┬─────────┘         └─────────┬─────────┘

&nbsp;               │                             │

&nbsp;               ▼                             ▼

&nbsp;     ┌───────────────────┐         ┌───────────────────┐

&nbsp;     │ CLN Subtype       │         │ Batten /          │

&nbsp;     │ Prediction        │         │ Non-Batten       │

&nbsp;     └─────────┬─────────┘         └─────────┬─────────┘

&nbsp;               │                             │

&nbsp;               │                             ▼

&nbsp;               │                   ┌───────────────────┐

&nbsp;               │                   │    Grad-CAM       │

&nbsp;               │                   │ Explainability    │

&nbsp;               │                   └─────────┬─────────┘

&nbsp;               │                             │

&nbsp;               └──────────────┬──────────────┘

&nbsp;                              ▼

&nbsp;                 ┌──────────────────────────┐

&nbsp;                 │ Assessment \& Reasoning   │

&nbsp;                 │        Module            │

&nbsp;                 └────────────┬─────────────┘

&nbsp;                              │

&nbsp;               ┌──────────────┴──────────────┐

&nbsp;               │                             │

&nbsp;               ▼                             ▼

&nbsp;     ┌───────────────────┐         ┌───────────────────┐

&nbsp;     │ Patient Summary   │         │ Supportive Care   │

&nbsp;     │ and AI Reasoning  │         │ Recommendations   │

&nbsp;     └───────────────────┘         └───────────────────┘

