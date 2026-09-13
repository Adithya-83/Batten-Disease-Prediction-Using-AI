\# Testing and Results



\## 1. Testing Overview



The Batten Disease Prediction Using AI application was tested at multiple levels to verify the functionality of the machine learning models, MRI analysis pipeline, explainability component, user interface, and dynamic application behavior.



The testing process included:



1\. MRI prediction testing

2\. Clinical input testing

3\. Missing MRI testing

4\. Grad-CAM testing

5\. Dynamic nutrition testing

6\. Session-state testing

7\. Application restart testing

8\. MRI model validation



---



\# 2. Functional Testing



\## Test Case 1 — Normal MRI



\### Objective



Verify that the application can process a normal MRI image and produce a Non-Batten assessment when appropriate.



\### Procedure



1\. Open the Streamlit application.

2\. Enter the required patient information.

3\. Upload a normal MRI image.

4\. Click the analysis button.

5\. Observe the MRI prediction and final assessment.



\### Expected Result



The MRI model should process the image successfully and provide a classification without application errors.



\### Result



\*\*PASS\*\*



---



\# 3. Batten MRI Testing



\## Test Case 2 — Batten MRI



\### Objective



Verify that the application can process a Batten-associated MRI image.



\### Procedure



1\. Open the Streamlit application.

2\. Enter the required patient information.

3\. Upload a Batten MRI image.

4\. Run the analysis.

5\. Observe the MRI prediction, assessment, and explanation.



\### Expected Result



The system should process the MRI successfully and display the corresponding model assessment.



\### Result



\*\*PASS\*\*



---



\# 4. Missing MRI Testing



\## Test Case 3 — No MRI Uploaded



\### Objective



Verify that the application handles the absence of an MRI image without crashing.



\### Procedure



1\. Enter patient information.

2\. Do not upload an MRI.

3\. Attempt to perform the analysis.



\### Expected Result



The application should handle the missing input appropriately instead of generating an unhandled error.



\### Result



\*\*PASS\*\*



---



\# 5. Clinical Input Testing



\## Test Case 4 — Different Clinical Inputs



\### Objective



Verify that changes in clinical input are processed by the clinical model.



\### Procedure



Different combinations of available patient information were entered into the application.



The resulting clinical prediction and confidence were observed.



\### Expected Result



The clinical model should process the supplied information and produce an NCL subtype prediction when valid input is provided.



\### Result



\*\*PASS\*\*



---



\# 6. Grad-CAM Testing



\## Test Case 5 — MRI Attention Map



\### Objective



Verify that Grad-CAM can generate an attention visualization for the MRI prediction.



\### Procedure



1\. Upload an MRI image.

2\. Run the analysis.

3\. Observe the generated attention map.



\### Expected Result



The application should generate and display a Grad-CAM visualization without causing an application error.



\### Result



\*\*PASS\*\*



\### Interpretation



The Grad-CAM visualization highlights image regions that contributed more strongly to the neural network's classification.



It is an interpretability aid and does not establish definitive disease localization.



---



\# 7. Dynamic Nutrition Testing



\## Test Case 6 — Nutrition and Daily Care



\### Objective



Verify that the Nutrition \& Daily Care section responds dynamically to the assessment.



\### Procedure



The application was tested with different assessment outcomes.



\### Expected Result



The displayed guidance should change according to the assessment instead of remaining as a fixed static message.



\### Result



\*\*PASS\*\*



The section provides supportive nutrition, hydration, feeding, and daily-care information appropriate to the displayed assessment.



---



\# 8. Session-State Testing



\## Test Case 7 — Assessment State Persistence



\### Objective



Verify that the assessment state is correctly maintained between the analysis section and downstream application sections.



\### Procedure



1\. Perform an MRI analysis.

2\. Generate the assessment.

3\. Navigate through the downstream sections.

4\. Observe the assessment-dependent content.



\### Expected Result



The downstream sections should correctly access the current assessment without producing undefined-variable errors.



\### Result



\*\*PASS\*\*



---



\# 9. Application Restart Testing



\## Test Case 8 — Application Restart



\### Objective



Verify that the application can restart successfully and load the required models.



\### Procedure



1\. Stop the Streamlit application.

2\. Start it again.

3\. Open the dashboard.

4\. Perform an analysis.



\### Expected Result



The application should restart successfully and load the saved clinical and MRI models.



\### Result



\*\*PASS\*\*



---



\# 10. MRI Model Validation



\## 10.1 Dataset



The MRI training/evaluation dataset contains:



| Class | Images |

|---|---:|

| Batten | 41 |

| Non-Batten | 82 |

| Total | 123 |



---



\## 10.2 Validation Strategy



A stratified 20% validation split was used for the final internal MRI evaluation.



Validation set:



| Class | Images |

|---|---:|

| Batten | 8 |

| Non-Batten | 17 |

| Total | 25 |



The stratified split was used to ensure that both classes were represented in the validation subset.



---



\# 11. MRI Model Results



The final internal validation results were:



| Metric | Result |

|---|---:|

| Accuracy | \*\*88.00%\*\* |

| Precision | \*\*85.71%\*\* |

| Recall | \*\*75.00%\*\* |

| F1-score | \*\*80.00%\*\* |



---



\# 12. Confusion Matrix



The evaluation produced the following confusion matrix:



```text

\[\[16, 1],

&nbsp;\[ 2, 6]]

