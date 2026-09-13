\# 12. TESTING AND VALIDATION



\## 12.1 Overview



Testing and validation were performed to verify the functional behavior of the Batten Disease Prediction Using Artificial Intelligence research prototype.



Testing was conducted at multiple levels:



1\. Input validation

2\. Clinical model testing

3\. MRI model testing

4\. Grad-CAM testing

5\. User-interface testing

6\. Dynamic content testing

7\. Session-state testing

8\. Application restart testing

9\. Final system integration testing



The purpose of testing was to identify implementation errors, verify expected model behavior, and ensure that the major application modules operate together correctly.



---



\## 12.2 Testing Strategy



The testing strategy consisted of both model-level and application-level testing.



\### Model-Level Testing



Model-level testing focused on:



\- Clinical prediction

\- MRI prediction

\- Classification threshold behavior

\- Model output interpretation

\- Grad-CAM generation



\### Application-Level Testing



Application-level testing focused on:



\- Patient information input

\- MRI upload

\- Analysis execution

\- Result display

\- AI reasoning

\- Nutrition and daily-care behavior

\- Session-state handling

\- Application restart behavior



---



\## 12.3 Functional Test Cases



The following functional test cases were performed.



| Test ID | Test Case | Expected Result | Status |

|---|---|---|---|

| TC01 | Launch Streamlit application | Application loads successfully | Passed |

| TC02 | Enter valid patient information | Information is accepted | Passed |

| TC03 | Upload valid MRI image | Image is accepted | Passed |

| TC04 | Run analysis with valid inputs | Analysis completes successfully | Passed |

| TC05 | Test known Batten MRI | Batten-associated classification is generated | Passed |

| TC06 | Test known normal MRI | Non-Batten classification is generated | Passed |

| TC07 | Run without MRI input | Application handles missing input | Passed |

| TC08 | Change clinical inputs | Clinical output updates accordingly | Passed |

| TC09 | Generate Grad-CAM | Attention map is generated | Passed |

| TC10 | Test Batten result nutrition section | Batten-supportive information appears | Passed |

| TC11 | Test Non-Batten result nutrition section | General nutrition information appears | Passed |

| TC12 | Test Streamlit rerun/session state | Previous assessment state remains available | Passed |

| TC13 | Restart application | Application starts without state-related failure | Passed |

| TC14 | Complete full analysis workflow | All major modules operate together | Passed |



---



\## 12.4 Clinical Model Validation



The clinical machine learning component was tested using structured patient information.



The model was evaluated during development against the available clinical dataset.



Multiple algorithms were considered:



\- Logistic Regression

\- Decision Tree

\- Random Forest



Random Forest was selected as the final clinical model based on the observed model comparison.



The final model produces an NCL subtype prediction:



```text

CLN1

CLN2

CLN3

