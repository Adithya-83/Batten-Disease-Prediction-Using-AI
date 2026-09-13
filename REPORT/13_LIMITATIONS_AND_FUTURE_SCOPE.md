\# 13. LIMITATIONS AND FUTURE SCOPE



\## 13.1 Overview



The Batten Disease Prediction Using Artificial Intelligence system was developed as an academic and research prototype. Although the system successfully integrates clinical machine learning, MRI-based deep learning, Grad-CAM explainability, and a Streamlit interface, several limitations must be considered when interpreting its results.



These limitations primarily arise from dataset size, data diversity, validation methodology, clinical representation, and the absence of external clinical validation.



---



\## 13.2 Dataset Limitations



The MRI dataset used in the project is relatively small.



The final MRI dataset contains:



\- 41 Batten MRI images

\- 82 normal MRI images

\- 123 total images



The limited number of images restricts the ability of the deep learning model to learn the full variability of MRI appearances associated with Batten disease.



The Batten MRI images were obtained from research-literature figures rather than from a standardized clinical imaging repository.



Consequently, variations may exist in:



\- MRI acquisition protocols

\- Scanner characteristics

\- Image resolution

\- Image orientation

\- Image presentation

\- Anatomical coverage

\- Figure quality



These variations may affect model generalization.



---



\## 13.3 Limited Representation of Non-Batten Conditions



The non-Batten class used in the MRI model consists of normal MRI images.



It does not represent the complete range of neurological conditions that may appear in real clinical practice.



For example, the current model was not trained as a comprehensive classifier distinguishing Batten disease from:



\- Other neurodegenerative disorders

\- Brain tumors

\- Multiple sclerosis

\- Other metabolic disorders

\- Other pediatric neurological diseases

\- Other causes of cerebral atrophy



Therefore, the Non-Batten result should not be interpreted as confirmation that an MRI is clinically normal.



---



\## 13.4 Clinical Dataset Limitation



The clinical dataset was derived from NCL-related patient and mutation information associated with CLN1, CLN2, and CLN3.



The available data is primarily composed of disease-associated cases rather than a balanced Batten-versus-healthy clinical population.



Therefore, the clinical Random Forest model is used to provide NCL subtype context.



Its output should not be interpreted as a validated binary probability of Batten disease.



---



\## 13.5 Small Validation Set



The final MRI model was evaluated using an internal validation set of 25 images.



The measured results were:



\- Accuracy: 88.00%

\- Precision: 85.71%

\- Recall: 75.00%

\- F1-score: 80.00%



Although these results are encouraging for a research prototype, a validation set of this size is insufficient to establish reliable clinical performance.



A small validation set can produce unstable estimates of performance and may not adequately represent the variability present in unseen clinical data.



---



\## 13.6 Lack of External Validation



The current system has not been externally validated using an independent clinical MRI dataset.



External validation is necessary to determine whether the model can generalize beyond the images used during development.



Future studies should evaluate the trained model on datasets collected independently from different:



\- Hospitals

\- MRI scanners

\- Patient populations

\- Imaging protocols

\- Geographic regions



---



\## 13.7 Lack of Prospective Clinical Validation



The application has not undergone prospective clinical evaluation.



The current testing verifies software functionality and internal model behavior rather than real-world clinical effectiveness.



Before clinical use could be considered, prospective studies involving appropriate clinical experts and ethically collected patient data would be necessary.



---



\## 13.8 Model Calibration Limitation



The MRI model produces a sigmoid score that is used to calculate a complementary Batten score.



However, this score has not been clinically calibrated.



Therefore, a value such as 0.80 should not be interpreted as meaning that there is an 80% clinical probability that the patient has Batten disease.



The value represents the model's output under the implemented classification setup.



Similarly, the clinical model's probability output represents subtype-classification confidence rather than a calibrated disease probability.



---



\## 13.9 Multimodal Fusion Limitation



The application presents clinical and MRI results together, but the current system does not contain a formally trained multimodal fusion model.



The two model responsibilities are intentionally separated:



```text

Clinical Model

&nbsp;     ↓

NCL Subtype Context



MRI Model

&nbsp;     ↓

Batten/Non-Batten Classification

