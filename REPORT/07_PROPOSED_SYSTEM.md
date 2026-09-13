\# 6. PROPOSED SYSTEM



\## 6.1 Overview



The proposed system is an Artificial Intelligence-based research prototype designed to analyze Batten Disease-related clinical information and brain MRI images.



The system uses two specialized AI pipelines:



1\. A clinical machine learning pipeline for NCL subtype prediction.

2\. An MRI deep learning pipeline for Batten versus Non-Batten classification.



A Grad-CAM explainability component is integrated into the MRI pipeline to provide a visual representation of regions that contributed more strongly to the neural network's classification.



The complete workflow is deployed through an interactive Streamlit application.



---



\## 6.2 Proposed System Architecture



The proposed architecture is:



```text

&nbsp;                    USER

&nbsp;                      │

&nbsp;           ┌──────────┴──────────┐

&nbsp;           │                     │

&nbsp;           ▼                     ▼

&nbsp;   Clinical Information       MRI Image

&nbsp;           │                     │

&nbsp;           ▼                     ▼

&nbsp;   Clinical Preprocessing    Image Preprocessing

&nbsp;           │                     │

&nbsp;           ▼                     ▼

&nbsp;      Random Forest         EfficientNet-B0

&nbsp;           │                     │

&nbsp;           ▼                     ▼

&nbsp;     NCL Subtype             Batten /

&nbsp;     CLN1 / CLN2 / CLN3      Non-Batten

&nbsp;                                 │

&nbsp;                                 ▼

&nbsp;                              Grad-CAM

&nbsp;                                 │

&nbsp;           ┌─────────────────────┘

&nbsp;           │

&nbsp;           ▼

&nbsp;      Streamlit Application

&nbsp;           │

&nbsp;           ▼

&nbsp;     AI-Assisted Assessment

&nbsp;           │

&nbsp;      ┌────┴────┐

&nbsp;      ▼         ▼

&nbsp;   Reasoning  Supportive

&nbsp;              Guidance

