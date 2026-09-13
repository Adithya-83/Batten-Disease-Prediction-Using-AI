\# Data Flow



\## 1. Overview



The Batten Disease Prediction Using AI system follows two parallel data-processing pipelines:



1\. Clinical Data Pipeline

2\. MRI Image Pipeline



The outputs of these pipelines are presented together through the Streamlit application.



---



\## 2. Clinical Data Flow



The clinical data originates from the NCL patient and mutation datasets.



```text

Raw Clinical Data

&nbsp;     ↓

CLN1 / CLN2 / CLN3 Data

&nbsp;     ↓

Data Cleaning

&nbsp;     ↓

Data Preprocessing

&nbsp;     ↓

Processed Clinical Dataset

&nbsp;     ↓

Feature Preparation

&nbsp;     ↓

Random Forest Model

&nbsp;     ↓

NCL Subtype Prediction

&nbsp;     ↓

CLN1 / CLN2 / CLN3

