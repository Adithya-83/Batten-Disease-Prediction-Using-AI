\# Batten Disease Prediction Using Artificial Intelligence



\## 1. Clinical Prediction Model



\### Dataset

The clinical dataset was constructed from the UCL NCL Mutation and Patient Database using CLN1, CLN2 and CLN3 patient and mutation records.



\### Preprocessing

Clinical records were cleaned and processed before model training. Patient and mutation information was combined where appropriate to construct the processed clinical dataset.



\### Machine Learning Models

The following classification algorithms were evaluated:



\- Logistic Regression

\- Decision Tree

\- Random Forest



The Random Forest classifier achieved the best overall validation performance and was selected as the final clinical model.



\### Saved Model

`models/batten\_disease\_best\_model.pkl`



\### Clinical Model Output

The clinical model predicts the NCL subtype:



\- CLN1

\- CLN2

\- CLN3



The clinical model confidence represents confidence in the predicted subtype. It is NOT interpreted as a probability of having Batten disease.



---



\## 2. MRI Prediction Model



\### Dataset

The MRI dataset contains:



\- Batten/NCL MRI images collected from research literature

\- Normal MRI images from the selected normal/control dataset



The other neurological disease categories in the source dataset were excluded because the project focuses on Batten versus normal classification.



\### Dataset Size



Total MRI images used for model development:



\- Batten: 41

\- Non-Batten: 82

\- Total: 123



\### Preprocessing



Each MRI image is:



1\. Converted to RGB

2\. Resized to 224 × 224 pixels

3\. Converted to a floating-point image array

4\. Expanded to a batch dimension



\### Architecture



The MRI classifier uses transfer learning with EfficientNet-B0.



Pipeline:



MRI Image

↓

RGB Conversion

↓

224 × 224 Resize

↓

Data Augmentation

↓

Pretrained EfficientNet-B0

↓

Global Average Pooling

↓

Dropout

↓

Dense Sigmoid Output



\### Training



EfficientNet-B0 was initially used with its pretrained feature extractor frozen.



The model was then fine-tuned by unfreezing selected upper layers while keeping Batch Normalization layers frozen.



Binary cross-entropy was used as the loss function.



Adam optimization was used during training.



\### Output Interpretation



The training directory labels were:



\- `batten = 0`

\- `non\_batten = 1`



Therefore, the sigmoid model output represents:



`P(Non-Batten)`



The application converts this to:



`P(Batten) = 1 - P(Non-Batten)`



The current application uses a 0.50 decision threshold for the Batten classification.



\### Saved Model



`models/batten\_mri\_efficientnet.keras`



---



\## 3. MRI Model Evaluation



A stratified 20% validation split was used for evaluation.



Validation dataset:



\- Batten: 8 images

\- Non-Batten: 17 images

\- Total: 25 images



\### Evaluation Results



| Metric | Result |

|---|---:|

| Accuracy | 88.00% |

| Precision | 85.71% |

| Recall | 75.00% |

| F1-score | 80.00% |



\### Confusion Matrix



The evaluation produced:



\[\[16, 1],

&nbsp;\[ 2, 6]]



The corresponding classification report indicates:



\- Batten precision: 0.89

\- Batten recall: 0.94

\- Batten F1-score: 0.91

\- Non-Batten precision: 0.86

\- Non-Batten recall: 0.75

\- Non-Batten F1-score: 0.80



These results represent internal validation performance on the available dataset.



---



\## 4. Explainable AI



Grad-CAM is used to generate an attention map for the MRI prediction.



The attention map highlights image regions that contributed more strongly to the neural network's classification.



Grad-CAM is used as an interpretability mechanism and does not establish that a highlighted region is definitively a disease lesion.



---



\## 5. Application Decision Logic



The application combines two sources of information:



\### Clinical Model

Provides the predicted NCL subtype and its classification confidence.



\### MRI Model

Provides the Batten versus Non-Batten classification.



The MRI model is responsible for the direct Batten/Non-Batten image classification.



The clinical model provides subtype/context information.



The clinical subtype confidence is not treated as a calibrated Batten disease probability.



---



\## 6. Application Workflow



1\. User enters patient information.

2\. User uploads an MRI image.

3\. Clinical information is processed by the clinical ML model.

4\. MRI image is processed by the EfficientNet-B0 model.

5\. MRI prediction is generated.

6\. Grad-CAM attention visualization is generated.

7\. The application presents the final assessment.

8\. The application provides model reasoning and supporting context.

9\. Nutrition and daily-care guidance changes according to the assessment.

10\. A medical disclaimer is displayed.



---



\## 7. Limitations



\### Clinical Dataset

The clinical dataset primarily contains NCL/Batten-associated patient and mutation records and should not be interpreted as a balanced healthy-versus-disease clinical dataset.



\### MRI Dataset Size

The MRI dataset is relatively small.



\### MRI Source

The Batten MRI images were collected from research literature and therefore do not represent a standardized clinical imaging database.



\### Validation

The reported MRI metrics are based on an internal stratified validation split rather than an independent external clinical test set.



\### Clinical Use

The system is a research/academic prototype and is not clinically validated or intended to replace diagnosis by qualified healthcare professionals.



\### Generalization

Performance on new hospitals, scanners, acquisition protocols, populations, or clinically different cases may differ from the reported validation results.



---



\## 8. Future Improvements



Potential future improvements include:



\- Larger clinically curated MRI datasets

\- Independent external validation

\- Multi-center MRI data

\- Better class balancing

\- Prospective clinical evaluation

\- Multimodal model training using clinical and MRI features jointly

\- Calibration of prediction probabilities

\- More extensive explainability evaluation

\- Evaluation across additional NCL subtypes



---



\## 9. Final Model Status



The current implementation has completed:



\- Clinical dataset preprocessing

\- Clinical model training

\- MRI dataset preparation

\- MRI model training

\- MRI fine-tuning

\- MRI validation

\- Grad-CAM implementation

\- Dynamic assessment reasoning

\- Dynamic nutrition guidance

\- Application testing



The current trained models are retained without further retraining.

