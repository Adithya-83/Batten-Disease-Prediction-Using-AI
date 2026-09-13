\# 2. PROBLEM STATEMENT



\## 2.1 Problem Definition



Batten Disease, or Neuronal Ceroid Lipofuscinosis (NCL), is a rare group of inherited neurodegenerative disorders with multiple genetic subtypes and variable clinical manifestations.



The analysis of Batten Disease-related information may involve both structured clinical information and brain MRI findings. However, the rarity of the disease and the complexity of clinical and imaging patterns make the development of automated analysis systems challenging.



Traditional rule-based approaches may have difficulty identifying complex relationships within clinical data and visual patterns within MRI images.



Therefore, there is a need to investigate an Artificial Intelligence-based approach capable of processing different types of Batten Disease-related information and providing an interpretable research-oriented assessment.



---



\## 2.2 Existing Challenge



The project addresses the following challenges:



\### 1. Clinical Data Complexity



Clinical and mutation datasets contain multiple patient-related and genetic attributes.



Identifying relationships between these attributes and NCL subtypes can be difficult using manual analysis alone.



\### 2. MRI Image Complexity



Brain MRI images contain complex visual information.



Manually identifying patterns associated with Batten/NCL-related imaging characteristics can be challenging, particularly when working with limited and heterogeneous research-derived images.



\### 3. Limited Availability of Data



Batten Disease is rare, resulting in limited availability of large, standardized datasets suitable for machine learning and deep learning.



\### 4. Interpretability



A deep learning prediction alone does not explain which regions of an MRI influenced the model.



An explainability mechanism is therefore desirable for an AI research prototype.



\### 5. Integration



Clinical information and MRI information represent different data modalities.



A practical research system should be able to process these sources through appropriate specialized models and present their outputs in a unified interface.



---



\## 2.3 Proposed Problem



The problem addressed by this project can be stated as:



> \*\*To develop an Artificial Intelligence-based research prototype that analyzes Batten Disease-related clinical information and brain MRI images, predicts NCL subtype from clinical information, classifies MRI images as Batten or Non-Batten, and provides an interpretable visualization of the MRI model's prediction.\*\*



---



\## 2.4 Problem Decomposition



The overall problem is divided into two primary machine learning tasks.



\### Task 1 — Clinical Classification



Given processed clinical and mutation-related features associated with NCL:



```text

Input:

Clinical Information



Output:

CLN1 / CLN2 / CLN3

