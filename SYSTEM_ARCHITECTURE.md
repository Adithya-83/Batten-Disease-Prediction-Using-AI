\# System Architecture



\## 1. Overview



The Batten Disease Prediction Using AI system is designed as a multimodal research prototype that processes both clinical information and brain MRI images.



The system contains four major layers:



1\. Input Layer

2\. Processing and Model Layer

3\. Output Layer

4\. Infrastructure and Deployment Layer



---



\## 2. Input Layer



The system accepts two categories of input.



\### 2.1 Clinical Information



The user provides patient-related information including:



\- Gender

\- Clinical presentation / phenotype

\- Age at onset

\- Histology

\- Country of origin



This information is passed to the clinical machine learning model.



\### 2.2 Brain MRI Image



The user uploads a brain MRI image in a supported image format.



The MRI image is passed through the MRI preprocessing pipeline before being analyzed by the deep learning model.



---



\## 3. Processing and Model Layer



The processing layer contains two independent analysis pipelines.



\### 3.1 Clinical Data Pipeline



```text

Clinical Information

&nbsp;       ↓

Data Cleaning

&nbsp;       ↓

Feature Processing

&nbsp;       ↓

Machine Learning Model

&nbsp;       ↓

NCL Subtype Prediction

