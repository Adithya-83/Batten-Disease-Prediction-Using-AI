\# 4. LITERATURE SURVEY



\## 4.1 Introduction



Neuronal Ceroid Lipofuscinoses (NCLs), commonly referred to as Batten Disease, represent a group of inherited neurodegenerative disorders. Because NCL includes multiple genetic subtypes and variable clinical manifestations, research has investigated clinical characteristics, genetic information, and neuroimaging findings to improve understanding of the disease.



Magnetic Resonance Imaging (MRI) has been used in research to investigate structural and disease-related changes associated with NCL. At the same time, advances in Artificial Intelligence and deep learning have enabled automated analysis of complex medical imaging data.



The following literature was reviewed as part of the development of this project.



---



\# 4.2 MRI Findings in Neuronal Ceroid Lipofuscinosis



\### Study



\*\*MRI findings in neuronal ceroid lipofuscinosis\*\*



Source:



https://pmc.ncbi.nlm.nih.gov/articles/PMC7515973/



\### Summary



This work discusses MRI findings associated with neuronal ceroid lipofuscinosis and demonstrates the relevance of neuroimaging in the investigation of NCL.



The study provides supporting background for the use of MRI information in an AI-based NCL research project.



\### Relevance to the Project



The research was used as one of the sources for understanding MRI-related characteristics of NCL and for identifying relevant MRI material during the development of the project's MRI dataset.



---



\# 4.3 Expanding the Neuroimaging Phenotype of NCLs



\### Study



\*\*Expanding the Neuroimaging Phenotype of Neuronal Ceroid Lipofuscinoses\*\*



Source:



https://pmc.ncbi.nlm.nih.gov/articles/PMC7661073/



\### Summary



This research investigates neuroimaging characteristics associated with NCL and expands the understanding of the imaging phenotype across different NCL forms.



\### Relevance to the Project



The study supports the project's focus on MRI-based analysis and provides background regarding variation in neuroimaging findings among NCL subtypes.



---



\# 4.4 Assessment of Disease Severity in Late Infantile NCL Using Multiparametric MR Imaging



\### Study



\*\*Assessment of Disease Severity in Late Infantile Neuronal Ceroid Lipofuscinosis Using Multiparametric MR Imaging\*\*



Source:



https://pmc.ncbi.nlm.nih.gov/articles/PMC3644851/



\### Summary



This research investigates the use of multiparametric MR imaging for assessing disease-related changes in late infantile NCL.



It demonstrates that MRI can provide measurable information related to disease characteristics and severity.



\### Relevance to the Project



The study provides supporting evidence for considering MRI as an important source of information in NCL-related research.



---



\# 4.5 MRI Brain Volume Measurements in Infantile NCL



\### Study



\*\*MRI Brain Volume Measurements in Infantile Neuronal Ceroid Lipofuscinosis\*\*



Source:



https://pmc.ncbi.nlm.nih.gov/articles/PMC5309144/



\### Summary



This study investigates MRI-based brain volume measurements in infantile NCL and provides evidence of structural brain changes associated with the disease.



\### Relevance to the Project



The work demonstrates the usefulness of quantitative and structural MRI information in NCL research and supports investigation of MRI-based computational analysis.



---



\# 4.6 Natural History of MRI Brain Volumes in CLN3 Disease



\### Study



\*\*Natural history of MRI brain volumes in CLN3 disease\*\*



Source:



https://pmc.ncbi.nlm.nih.gov/articles/PMC9474504/



\### Summary



This research examines longitudinal MRI brain volume measurements in CLN3 disease.



The study provides information about changes in brain structure over the course of CLN3 disease.



\### Relevance to the Project



Because CLN3 is one of the NCL subtypes represented in the project's clinical dataset, this research provides useful background for understanding the relationship between CLN3 disease and MRI findings.



---



\# 4.7 Clinical and MRI Findings in Batten Disease



\### Study



\*\*Clinical and MRI findings in Batten disease\*\*



Source:



https://pubmed.ncbi.nlm.nih.gov/9392580/



\### Summary



This research describes clinical and MRI findings associated with Batten Disease.



The work provides historical clinical and neuroimaging evidence supporting the relevance of MRI in Batten Disease research.



\### Relevance to the Project



The study provides background information for the project's MRI-focused analysis of Batten Disease.



---



\# 4.8 Multiparametric MRI Imaging of CLN2



\### Research Project



\*\*Multiparametric MRI Imaging of CLN2 (Form of Batten Disease)\*\*



Source:



https://radiology.weill.cornell.edu/research/research-projects/multiparametric-mri-imaging-cln2-form-batten-disease



\### Summary



This research project investigates multiparametric MRI imaging in CLN2 disease.



\### Relevance to the Project



CLN2 is one of the NCL forms represented in the project's clinical information.



The research provides additional background regarding MRI-based investigation of CLN2.



---



\# 4.9 MRI in NCL Subtypes



\### Study



\*\*MRI in neuronal ceroid lipofuscinosis subtypes\*\*



Source:



https://pubmed.ncbi.nlm.nih.gov/24029190/



\### Summary



This research examines MRI characteristics associated with different NCL subtypes.



\### Relevance to the Project



The study is relevant to the project's use of clinical subtype information and MRI analysis as separate but complementary components.



---



\# 4.10 Artificial Intelligence in the Proposed Approach



The project also incorporates established machine learning and deep learning techniques.



The clinical component evaluates traditional supervised machine learning algorithms including Logistic Regression, Decision Tree, and Random Forest.



The MRI component uses EfficientNet-B0 with transfer learning and fine-tuning.



Grad-CAM is used to provide an interpretable visualization of the MRI model's prediction.



---



\# 4.11 EfficientNet



\### Study



\*\*EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks\*\*



Tan, M., \& Le, Q. V. (2019).



Proceedings of the 36th International Conference on Machine Learning (ICML).



\### Summary



EfficientNet introduces a family of convolutional neural networks designed to achieve strong image classification performance while maintaining computational efficiency.



\### Relevance to the Project



EfficientNet-B0 is used as the backbone of the project's MRI classification model.



The pretrained network provides a starting point for transfer learning, which is particularly useful when the available project-specific dataset is limited.



---



\# 4.12 Grad-CAM



\### Study



\*\*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\*\*



Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., \& Batra, D. (2017).



Proceedings of the IEEE International Conference on Computer Vision (ICCV).



\### Summary



Grad-CAM provides visual explanations for predictions made by convolutional neural networks by identifying image regions associated with the model's classification.



\### Relevance to the Project



Grad-CAM is integrated into the MRI pipeline to provide an attention visualization.



The visualization helps users understand which image regions contributed more strongly to the model's classification.



---



\# 4.13 Research Gap



The reviewed NCL literature demonstrates the importance of clinical information and MRI for understanding Batten Disease and its subtypes.



However, the research sources reviewed for this project primarily focus on clinical characterization, disease progression, and neuroimaging findings rather than the specific interactive AI workflow implemented here.



The project therefore investigates an academic research prototype that combines:



```text

Clinical Information

&nbsp;       +

MRI Image Analysis

&nbsp;       +

Explainable AI

&nbsp;       +

Interactive Application

