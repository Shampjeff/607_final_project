# 607 Final Project
The final project for Data 607 in CUNY SPS MSDS 

This project spans several `.py` files and `.ipynb` Jupyter notebooks. Using an IDE such as PyCharm or JupyterLab will be helpful in regards to navigating these files with ease. 

I tried to detail in the narrative as well as with comments how and why this code works and seeks to accomplish. This is not a Rpub, but Github does allow for viewing `ipynb` files within the repo. 

## Summary and of Files

### `1_Data_Exploration.ipynb`
Exploration of the project data, counts, file types, and visuals to get a sense of the project. 

### `2_Plagiarism_Feature_Engineering.ipynb` 
Using the ideas in the cited research paper we develop functions to calculate features for plagiarism detection. 

### `utils.py`
A collection of utility functions used in conjunction with the above feature engineering notebook

### `3_Modeling_Trials.ipynb` 
Iterative testing of machine learning models on the training data. Using a custom package developed in another repo. See link in notebook for details. 

### `4_Training_a_Model.ipynb`
This notebook is meant to be used inside a SageMaker notebook instance. The creatation of the model training job, predictor, and deployment endpoint are done in this notebook. Test data is sent to the deployed model in this notebook as well. 

### `prediction_engine.py`
This is the prediction engine for use on the web application. This file handles the pre-processing of strings sent via the web app and sends the results to the SageMaker endpoint. Results from the endpoint are post-processed and sent back to the web app for end-user feedback. 


## Web Application
`Final_webapp` is a directory that houses the necessary files to run a Django web app with python. The web app sends input data to the endpoint and returns the results for the end-user to see. 

Many of the files in `Final_webapp` are generated automatically by Django. The files; `home.html`, `views.py`, and `urls.py` were built for the project and are used for the tranmission of data to the endpoint. 


## `Final_Thoughts.md`
This is a markdown file with a summary of things I learned from this process and some key take aways I had as a result of this completing this final project. 