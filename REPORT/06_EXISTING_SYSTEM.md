\# 5. EXISTING SYSTEM



\## 5.1 Overview



The analysis of Batten Disease and Neuronal Ceroid Lipofuscinoses (NCLs) traditionally relies on clinical assessment, genetic information, neurological examination, and medical imaging.



Research literature has demonstrated the usefulness of MRI for investigating structural and disease-related changes associated with NCL.



However, the available approaches generally require interpretation of multiple sources of information by qualified healthcare professionals and do not provide the specific integrated workflow implemented in this project.



---



\## 5.2 Conventional Clinical Assessment



Clinical evaluation of suspected Batten Disease may involve information such as:



\- Patient history

\- Age at onset

\- Neurological manifestations

\- Clinical phenotype

\- Genetic and mutation information

\- Medical examination

\- Brain imaging



The interpretation of these factors requires appropriate clinical expertise.



---



\## 5.3 MRI-Based Assessment



MRI is an important neuroimaging modality for investigating NCL-related structural changes.



Research has reported MRI findings associated with different NCL subtypes.



However, interpretation of MRI findings is generally performed by trained medical professionals.



The identification of subtle or complex visual patterns may be challenging when performed manually.



---



\## 5.4 Genetic and Clinical Data Analysis



NCL subtypes are associated with different genetic and clinical characteristics.



Clinical and mutation databases can contain large amounts of structured information.



Manual analysis of such information can be time-consuming and may make it difficult to identify relationships between multiple variables.



Machine learning provides a computational approach for analyzing structured clinical information.



---



\## 5.5 Limitations of Conventional Approaches



The following limitations motivate the development of an AI-based research prototype.



\### 1. Manual Analysis



Clinical and imaging information requires expert interpretation.



\### 2. Multiple Data Sources



Patient information, genetic information, and MRI findings may need to be considered together during evaluation.



\### 3. Complex MRI Patterns



MRI images contain complex visual information that may be difficult to analyze using simple manually defined rules.



\### 4. Limited Disease Data



Batten Disease is rare, which limits the availability of large standardized datasets.



\### 5. Lack of Automated Image Interpretation



Traditional workflows do not inherently provide an automated deep learning classification of the MRI image.



\### 6. Limited Model Interpretability



A conventional automated classifier may provide a prediction without showing which image regions influenced the result.



\### 7. Lack of Unified Interactive Workflow



Clinical information, MRI analysis, explainability, and supportive information are not necessarily presented through a single research-oriented interface.



---



\## 5.6 Existing System Summary



The conventional approach can be represented as:



```text

Patient

&nbsp;  ↓

Clinical Evaluation

&nbsp;  ↓

Genetic / Mutation Analysis

&nbsp;  ↓

MRI Examination

&nbsp;  ↓

Expert Interpretation

&nbsp;  ↓

Clinical Assessment

