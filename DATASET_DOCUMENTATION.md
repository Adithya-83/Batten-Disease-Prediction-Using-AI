\# Dataset Documentation



\## 1. Overview



The project uses two major categories of data:



1\. Clinical and genetic information

2\. Brain MRI images



These datasets are processed independently because the clinical and MRI models perform different prediction tasks.



---



\# 2. Clinical Dataset



\## 2.1 Data Source



The clinical and mutation information was obtained from the:



\*\*UCL NCL Mutation and Patient Database\*\*



The project uses data associated with three NCL subtypes:



\- CLN1

\- CLN2

\- CLN3



The corresponding raw files are organized under:



```text

data/raw/

├── cln1/

├── cln2/

└── cln3/

