\# References



\## 1. Clinical and NCL Data Sources



\### UCL NCL Mutation and Patient Database



University College London.  

NCL Mutation and Patient Database.  

Used as the primary source for CLN1, CLN2 and CLN3 clinical and mutation information used in this project.



---



\## 2. MRI and Batten Disease Literature



\### Reference 1 — MRI Findings in Neuronal Ceroid Lipofuscinosis



"MRI findings in neuronal ceroid lipofuscinosis."



PubMed Central.



https://pmc.ncbi.nlm.nih.gov/articles/PMC7515973/



Used as a research source for MRI findings associated with neuronal ceroid lipofuscinosis.



---



\### Reference 2 — Neuroimaging Phenotype of NCLs



"Expanding the Neuroimaging Phenotype of Neuronal Ceroid Lipofuscinoses."



PubMed Central.



https://pmc.ncbi.nlm.nih.gov/articles/PMC7661073/



Used to understand neuroimaging characteristics across NCL subtypes.



---



\### Reference 3 — Late Infantile NCL and Multiparametric MRI



"Assessment of Disease Severity in Late Infantile Neuronal Ceroid Lipofuscinosis Using Multiparametric MR Imaging."



PubMed Central.



https://pmc.ncbi.nlm.nih.gov/articles/PMC3644851/



Used as a supporting research source for MRI characteristics and disease-related imaging changes.



---



\### Reference 4 — Infantile NCL MRI Brain Volume Measurements



"MRI Brain Volume Measurements in Infantile Neuronal Ceroid Lipofuscinosis."



PubMed Central.



https://pmc.ncbi.nlm.nih.gov/articles/PMC5309144/



Used as supporting literature regarding MRI-based structural brain changes in NCL.



---



\### Reference 5 — CLN3 MRI Brain Volumes



"Natural history of MRI brain volumes in CLN3 disease."



PubMed Central.



https://pmc.ncbi.nlm.nih.gov/articles/PMC9474504/



Used as supporting literature for longitudinal MRI findings associated with CLN3 disease.



---



\### Reference 6 — Clinical and MRI Findings in Batten Disease



"Clinical and MRI findings in Batten disease."



PubMed / National Library of Medicine.



https://pubmed.ncbi.nlm.nih.gov/9392580/



Used as supporting literature concerning clinical and MRI characteristics of Batten Disease.



---



\### Reference 7 — Multiparametric MRI Imaging of CLN2



"Weill Cornell Medicine — Multiparametric MRI Imaging of CLN2 (Form of Batten Disease)."



https://radiology.weill.cornell.edu/research/research-projects/multiparametric-mri-imaging-cln2-form-batten-disease



Used as supporting information regarding MRI imaging research in CLN2 disease.



---



\### Reference 8 — MRI in NCL Subtypes



"MRI in neuronal ceroid lipofuscinosis subtypes."



PubMed / National Library of Medicine.



https://pubmed.ncbi.nlm.nih.gov/24029190/



Used as supporting literature for MRI characteristics across NCL subtypes.



---



\# 3. Deep Learning and Computer Vision



\## EfficientNet



Tan, M., \& Le, Q. V. (2019).



"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."



Proceedings of the 36th International Conference on Machine Learning (ICML).



EfficientNet-B0 is used as the backbone architecture for the MRI classification model.



---



\# 4. Explainable Artificial Intelligence



\## Grad-CAM



Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., \& Batra, D. (2017).



"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."



Proceedings of the IEEE International Conference on Computer Vision (ICCV).



Grad-CAM is used in this project to generate MRI attention visualizations.



---



\# 5. Machine Learning Libraries



\## Scikit-learn



Pedregosa, F., et al. (2011).



"Scikit-learn: Machine Learning in Python."



Journal of Machine Learning Research, 12, 2825–2830.



Used for:



\- Logistic Regression

\- Decision Tree

\- Random Forest

\- Train/validation splitting

\- Performance evaluation

\- Classification metrics

\- Confusion matrix



---



\# 6. Deep Learning Framework



\## TensorFlow



Abadi, M., et al. (2016).



"TensorFlow: A System for Large-Scale Machine Learning."



OSDI.



TensorFlow/Keras is used for:



\- EfficientNet-B0

\- Model training

\- Fine-tuning

\- MRI inference

\- Grad-CAM implementation



---



\# 7. Application Framework



\## Streamlit



Streamlit.



"Streamlit — A faster way to build and share data apps."



https://streamlit.io/



Used to implement the interactive web-based dashboard for the project.



---



\# 8. Image Processing



\## OpenCV



OpenCV.



https://opencv.org/



Used for image-processing operations including MRI image preparation and visualization.



---



\# 9. Data Analysis



\## Pandas



Pandas documentation.



https://pandas.pydata.org/



Used for clinical dataset loading, cleaning, processing, and tabular data manipulation.



---



\## NumPy



NumPy documentation.



https://numpy.org/



Used for numerical processing and image-array operations.



---



\# 10. Model Persistence



\## Joblib



Joblib documentation.



https://joblib.readthedocs.io/



Used to save and load the trained clinical machine learning model.



---



\# 11. Project-Specific Resources



The following project resources were developed as part of this implementation:



```text

models/batten\_disease\_best\_model.pkl

models/batten\_mri\_efficientnet.keras



notebooks/mri\_model\_training.ipynb



clean\_batten\_mri.py

copy\_normal\_mri.py

evaluate\_mri.py



app.py

