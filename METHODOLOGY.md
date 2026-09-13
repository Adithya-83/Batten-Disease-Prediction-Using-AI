\# Methodology and Algorithms



\## 1. Methodology Overview



The proposed system uses two machine learning pipelines to analyze different types of information:



1\. Clinical data analysis

2\. MRI image analysis



The clinical pipeline uses traditional machine learning for NCL subtype prediction.



The MRI pipeline uses deep learning with EfficientNet-B0 for Batten versus Non-Batten image classification.



Grad-CAM is used to provide visual interpretability for the MRI prediction.



---



\# 2. Overall Methodology



```text

Patient Information + MRI Image

&nbsp;             │

&nbsp;      ┌──────┴──────┐

&nbsp;      │             │

&nbsp;      ▼             ▼

Clinical Pipeline   MRI Pipeline

&nbsp;      │             │

&nbsp;      ▼             ▼

Random Forest     EfficientNet-B0

&nbsp;      │             │

&nbsp;      ▼             ▼

NCL Subtype       Batten / Non-Batten

&nbsp;      │             │

&nbsp;      │             ▼

&nbsp;      │          Grad-CAM

&nbsp;      │             │

&nbsp;      └──────┬──────┘

&nbsp;             ▼

&nbsp;      Streamlit Dashboard

&nbsp;             │

&nbsp;             ▼

&nbsp;      AI-Assisted Assessment

