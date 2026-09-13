\# 14. CONCLUSION



\## 14.1 Project Conclusion



The project titled \*\*“Batten Disease Prediction Using Artificial Intelligence”\*\* was developed as an academic research prototype to investigate the application of artificial intelligence techniques to clinical and MRI-related information associated with Batten disease and neuronal ceroid lipofuscinoses (NCL).



The system integrates structured clinical-data analysis, deep learning-based MRI classification, explainable artificial intelligence, and an interactive Streamlit interface into a single application.



---



\## 14.2 Clinical Machine Learning Component



The clinical component was developed using structured information associated with CLN1, CLN2, and CLN3.



Multiple machine learning algorithms were evaluated, including:



\- Logistic Regression

\- Decision Tree

\- Random Forest



Random Forest was selected as the final clinical model based on the observed model comparison.



The model provides NCL subtype information such as CLN1, CLN2, and CLN3.



This output is used as clinical context within the application rather than as a direct probability of Batten disease.



---



\## 14.3 MRI Deep Learning Component



The MRI component uses EfficientNet-B0 with transfer learning and selective fine-tuning.



The model was trained using:



\- 41 Batten MRI images

\- 82 normal MRI images



The MRI images were standardized to 224 × 224 RGB inputs before classification.



The final MRI model achieved the following internal validation results:



| Metric | Result |

|---|---:|

| Accuracy | 88.00% |

| Precision | 85.71% |

| Recall | 75.00% |

| F1-score | 80.00% |



These results demonstrate the feasibility of the proposed approach on the available research dataset.



However, the dataset and validation set are small, and therefore the results cannot establish clinical diagnostic performance.



---



\## 14.4 Explainable AI



Grad-CAM was successfully integrated into the MRI analysis pipeline.



The generated attention map provides a visual representation of regions that contributed more strongly to the neural network's classification.



This improves the interpretability of the deep learning model and provides users with additional information about the model's decision process.



The Grad-CAM visualization is an explanation of model behavior and should not be interpreted as definitive pathological localization or medical segmentation.



---



\## 14.5 Integrated Application



The trained models were integrated into a Streamlit-based application.



The application provides:



\- Patient information input

\- MRI image upload

\- Clinical NCL subtype prediction

\- MRI Batten/Non-Batten classification

\- MRI model score

\- Grad-CAM visualization

\- AI reasoning

\- Patient summary

\- Supportive nutrition and daily-care information

\- Research disclaimer



The application was functionally tested across the major implemented modules.



---



\## 14.6 Achievement of Objectives



The project objectives were addressed through the implemented system.



The project successfully:



1\. Processed structured NCL clinical information.

2\. Evaluated multiple machine learning algorithms.

3\. Selected Random Forest for the clinical component.

4\. Prepared a Batten/Non-Batten MRI dataset.

5\. Implemented MRI preprocessing.

6\. Developed an EfficientNet-B0 image classifier.

7\. Applied transfer learning.

8\. Performed selective fine-tuning.

9\. Evaluated the MRI model using standard classification metrics.

10\. Integrated Grad-CAM explainability.

11\. Developed a Streamlit-based application.

12\. Integrated clinical and MRI outputs into a unified interface.

13\. Implemented dynamic supportive information.

14\. Performed functional and integration testing.

15\. Documented the limitations and future scope of the research prototype.



---



\## 14.7 Significance of the Project



The project demonstrates how artificial intelligence can be applied to rare-disease research where available datasets are limited and heterogeneous.



The use of transfer learning provides a practical approach for developing an image classifier when large disease-specific MRI datasets are unavailable.



The integration of Grad-CAM demonstrates the importance of explainability when applying deep learning to medical imaging research.



The project also demonstrates that clinical structured data and medical imaging can be incorporated into a single research-oriented application while maintaining separate responsibilities for the different analytical models.



---



\## 14.8 Limitations



The conclusions of this project must be interpreted with the following limitations:



\- The MRI dataset is small.

\- Batten MRI images were obtained from research-literature figures.

\- The non-Batten class consists of normal MRI images and does not represent all neurological disorders.

\- The clinical dataset is not a balanced Batten-versus-healthy dataset.

\- The MRI validation set contains only 25 images.

\- External clinical validation was not performed.

\- The model outputs are not clinically calibrated probabilities.

\- The system does not contain a formally trained multimodal fusion model.

\- The application is an academic research prototype rather than a clinically validated diagnostic system.



---



\## 14.9 Future Scope



Future development can significantly improve the system through:



\- Larger and standardized MRI datasets

\- Multi-center clinical data

\- External validation

\- Patient-level cross-validation

\- Improved MRI preprocessing

\- 3D MRI analysis

\- Multi-class differential diagnosis

\- Multimodal learning

\- Probability calibration

\- Advanced explainability techniques

\- Longitudinal disease monitoring

\- Secure clinical deployment

\- Collaboration with medical professionals



These improvements could provide a stronger foundation for future research into AI-assisted analysis of Batten disease and NCL.



---



\## 14.10 Final Conclusion



The \*\*Batten Disease Prediction Using Artificial Intelligence\*\* project successfully demonstrates an end-to-end research prototype combining machine learning, deep learning, computer vision, explainable AI, and web-based application development.



The Random Forest model provides NCL subtype context, while the EfficientNet-B0 model performs the primary MRI-based Batten/Non-Batten classification. Grad-CAM adds an interpretable visualization layer, and Streamlit provides an accessible interface for interacting with the complete system.



The MRI model achieved an internal validation accuracy of \*\*88.00%\*\*, with a precision of \*\*85.71%\*\*, recall of \*\*75.00%\*\*, and F1-score of \*\*80.00%\*\*.



Overall, the project demonstrates the technical feasibility of applying artificial intelligence techniques to rare-disease research using limited available clinical and imaging data.



The developed system should be regarded strictly as an \*\*academic and research prototype\*\*. Further work involving larger datasets, independent validation, clinical expertise, ethical oversight, and standardized medical imaging data is necessary before any clinical application can be considered.

