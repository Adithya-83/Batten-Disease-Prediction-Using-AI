\# ABSTRACT



Batten Disease, also known as Neuronal Ceroid Lipofuscinosis (NCL), is a group of rare inherited neurodegenerative disorders that can present with progressive neurological manifestations. Early recognition and appropriate clinical evaluation are important, but the rarity and heterogeneity of the disease make automated analysis challenging.



This project presents an Artificial Intelligence-based research prototype for Batten Disease-related analysis using both clinical information and brain Magnetic Resonance Imaging (MRI). The proposed system consists of two specialized machine learning pipelines. The clinical pipeline uses patient and mutation information associated with CLN1, CLN2 and CLN3 to predict the NCL subtype using a Random Forest classifier. The MRI pipeline uses an EfficientNet-B0 deep learning model with transfer learning and fine-tuning to classify MRI images as Batten or Non-Batten.



To improve interpretability of the MRI prediction, Gradient-weighted Class Activation Mapping (Grad-CAM) is incorporated to generate an attention visualization indicating image regions that contributed more strongly to the neural network's classification. The complete system is implemented as an interactive Streamlit application that accepts patient information and MRI images and presents the model outputs, reasoning, visualization, and supportive guidance through a unified dashboard.



The final MRI model achieved 88.00% accuracy, 85.71% precision, 75.00% recall, and 80.00% F1-score on an internal stratified validation split of 25 images. These results demonstrate the feasibility of applying deep learning to the available MRI dataset within an academic research setting.



However, the project has important limitations, including the relatively small MRI dataset, research-derived Batten MRI images, lack of external validation, and the non-clinical nature of the current evaluation. Therefore, the developed system is intended as an academic and research prototype and not as a clinically validated diagnostic system. Future work can focus on larger clinically curated datasets, multi-center validation, independent external testing, probability calibration, and joint multimodal learning using clinical and MRI features.

