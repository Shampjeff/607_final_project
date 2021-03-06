Live Plagiarism Prediction
========================================================
author: Jeff Shamp
date: 5/1/2020
autosize: true


Live Demo
========================================================

>- Detect Authenticity of Answers
>- http://127.0.0.1:8002


Overview
========================================================
End-to-end data science project culminating in live predictions from a simple web application

- Problem statement
- Features and modeling
- Deploy to cloud
- Launch web application for real-time model feedback

> **Citation**: Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press.

Problem Statement
========================================================
Build a predictor to identify the authenticity of student answer text

<br>
Plagiarism is defined by:
>- Similarity of "answer" text to "source" text.
>- Source texts are derived from Wikipedia pages for each sample question. 


Prediction Features
========================================================
>- Containment of N-grams
  - $\frac{\sum count(ngram_{ans}) \cap count(ngram_{source})}{\sum count(ngram_{ans})}$

>- Longest Common Subsequence
![Longest common sequence calculation](img/matrix_rules.png)

Modeling
========================================================
Binary classification

>- Small dataset
>- AWS SageMaker works well with SKLearn

>![GBM Top Model](img/gbm.png)

Fire Up AWS
========================================================
Lanuch SageMaker Notebook 

>![estimator](img/estimate.png)

>![deploy](img/deploy.png)

Build Web App
========================================================
End-user interface with the model 

>- Easier solution with AWS lambda and API Gateway
  - Hard to use with heavy data pre-processing from text
<br>  
>- Django to manage python scripts for data processing

>- Web developers <3

Invoke the Endpoint
========================================================
Get predictions in real time 

>- Build web application to run a prediction script

>![invoke](img/invoke.png)

Thank You 
========================================================

>- Questions?



