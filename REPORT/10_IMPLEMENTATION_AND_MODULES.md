\# 10. IMPLEMENTATION AND MODULES



\## 10.1 Overview



The Batten Disease Prediction Using Artificial Intelligence system is implemented as a modular research prototype using Python, machine learning, deep learning, computer vision, and Streamlit.



The implementation is divided into independent functional modules so that clinical analysis, MRI analysis, explainability, user interaction, and supportive information can be maintained separately.



The major implementation modules are:



1\. Application Configuration Module

2\. Clinical Data Module

3\. Clinical Prediction Module

4\. MRI Preprocessing Module

5\. MRI Prediction Module

6\. Grad-CAM Explainability Module

7\. Assessment and Reasoning Module

8\. Nutrition and Daily Care Module

9\. Streamlit Interface Module

10\. Model and Data Storage Module



---



\## 10.2 Application Configuration Module



The application begins by defining the project base directory and paths to the required resources.



The main resources include:



\- Clinical machine learning model

\- MRI deep learning model

\- Processed clinical dataset



The application uses relative paths derived from the project directory rather than relying on machine-specific absolute paths.



The primary model files are:



```text

models/batten\_disease\_best\_model.pkl

models/batten\_mri\_efficientnet.keras

