\# Limitations and Future Scope



\## 1. Limitations



Although the Batten Disease Prediction Using AI project demonstrates the feasibility of combining clinical machine learning and MRI-based deep learning, several limitations must be considered.



---



\## 1.1 Limited MRI Dataset



The MRI dataset contains only 123 images:



\- 41 Batten images

\- 82 Non-Batten images



This is relatively small for training a deep learning model.



A larger dataset would be required to establish more reliable generalization.



---



\## 1.2 Research-Derived Batten MRI Images



The Batten MRI images were collected from published research literature.



These images were therefore not obtained from a single standardized clinical imaging repository.



Differences in:



\- MRI acquisition protocols

\- Scanner hardware

\- Image resolution

\- Image orientation

\- Patient characteristics

\- Figure presentation



may affect model performance.



---



\## 1.3 Limited Representation of Clinical Diversity



The available MRI dataset does not provide sufficient representation of the full clinical diversity of Batten Disease.



Batten Disease includes multiple NCL subtypes and can demonstrate variation in disease stage and imaging characteristics.



The current dataset cannot fully represent this variation.



---



\## 1.4 Internal Validation



The reported MRI performance was obtained using an internal stratified validation split.



Final validation results:



| Metric | Result |

|---|---:|

| Accuracy | 88.00% |

| Precision | 85.71% |

| Recall | 75.00% |

| F1-score | 80.00% |



These results should not be interpreted as performance on an independent clinical test population.



---



\## 1.5 No External Validation



The model has not been evaluated on an independent external dataset from another institution or imaging center.



External validation would be necessary to assess whether the model generalizes to unseen clinical environments.



---



\## 1.6 Clinical Dataset Limitation



The clinical dataset is based primarily on NCL/Batten-associated patient and mutation records.



It is not a balanced healthy-versus-Batten clinical dataset.



Therefore, the clinical model is designed to provide NCL subtype information rather than a calibrated probability of Batten Disease.



---



\## 1.7 Separate Model Responsibilities



The current system uses two specialized models:



```text

Clinical Model

&nbsp;     ↓

NCL Subtype Prediction



MRI Model

&nbsp;     ↓

Batten / Non-Batten Classification

