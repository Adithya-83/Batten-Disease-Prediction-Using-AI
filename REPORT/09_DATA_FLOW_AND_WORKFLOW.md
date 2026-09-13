\# 9. DATA FLOW AND WORKFLOW



\## 9.1 Overview



The Batten Disease Prediction Using Artificial Intelligence system follows a sequential data-processing workflow beginning with patient information and MRI input and ending with an AI-generated research assessment.



The workflow consists of two major analytical paths:



1\. Clinical data analysis

2\. MRI image analysis



The outputs of these paths are presented together through the assessment and reasoning interface.



---



\## 9.2 Overall Data Flow



The overall data flow is:



```text

Patient Information ──────┐

&nbsp;                         │

&nbsp;                         ▼

&nbsp;                 Clinical Processing

&nbsp;                         │

&nbsp;                         ▼

&nbsp;                 Random Forest Model

&nbsp;                         │

&nbsp;                         ▼

&nbsp;                   CLN Subtype

&nbsp;                         │

&nbsp;                         │

&nbsp;                         ├──────────────┐

&nbsp;                         │              │

&nbsp;                         │              ▼

&nbsp;                         │       Assessment Module

&nbsp;                         │              ▲

&nbsp;                         │              │

MRI Image ────────────────┐              │

&nbsp;                         ▼              │

&nbsp;                  MRI Preprocessing     │

&nbsp;                         │              │

&nbsp;                         ▼              │

&nbsp;                  EfficientNet-B0       │

&nbsp;                         │              │

&nbsp;                         ▼              │

&nbsp;                Batten/Non-Batten       │

&nbsp;                         │              │

&nbsp;                         ▼              │

&nbsp;                      Grad-CAM          │

&nbsp;                         │              │

&nbsp;                         └──────────────┘

&nbsp;                                │

&nbsp;                                ▼

&nbsp;                      Final Research Result

&nbsp;                                │

&nbsp;               ┌────────────────┼────────────────┐

&nbsp;               ▼                ▼                ▼

&nbsp;         AI Reasoning      MRI Attention     Supportive

&nbsp;                             Map             Information

