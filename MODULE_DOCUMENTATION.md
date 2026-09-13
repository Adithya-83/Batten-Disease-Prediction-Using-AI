\# Module Documentation



\## 1. Overview



The Batten Disease Prediction Using AI project is organized into multiple modules responsible for data processing, model training, model evaluation, explainability, and application deployment.



The major modules are:



1\. Clinical Data Processing Module

2\. MRI Data Preparation Module

3\. Clinical Machine Learning Module

4\. MRI Deep Learning Module

5\. MRI Evaluation Module

6\. Grad-CAM Explainability Module

7\. Streamlit Application Module

8\. Patient Assessment Module

9\. Supportive Guidance Module



---



\# 2. Clinical Data Processing Module



\## Purpose



The Clinical Data Processing Module prepares the clinical and mutation information used by the clinical machine learning model.



\## Input



The module uses raw CLN1, CLN2 and CLN3 patient and mutation datasets.



The raw files are organized under:



```text

data/raw/

├── cln1/

├── cln2/

└── cln3/

