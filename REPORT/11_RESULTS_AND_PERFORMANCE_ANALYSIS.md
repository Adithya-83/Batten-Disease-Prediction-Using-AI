\# 11. RESULTS AND PERFORMANCE ANALYSIS



\## 11.1 Overview



The developed Batten Disease Prediction Using Artificial Intelligence system was evaluated at both the model and application levels.



The evaluation focused on:



\- Clinical machine learning model selection

\- MRI deep learning classification

\- MRI validation performance

\- Confusion matrix analysis

\- Precision, recall, and F1-score

\- Grad-CAM explainability

\- Streamlit application functionality

\- Input and output behavior



The reported model results represent internal research-prototype evaluation and should not be interpreted as clinical validation.



---



\## 11.2 Clinical Machine Learning Results



Multiple machine learning algorithms were evaluated for the structured clinical-data component.



The evaluated algorithms included:



\- Logistic Regression

\- Decision Tree

\- Random Forest



Random Forest provided the strongest observed performance among the evaluated algorithms and was selected as the final clinical model.



The selected model is stored as:



```text

models/batten\_disease\_best\_model.pkl

