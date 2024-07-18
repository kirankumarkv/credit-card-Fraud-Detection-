# Credit Card Fraud Detection
## Table of Contents

Project Overview
Dataset
Installation
Usage
Feature Engineering
Model Training
Evaluation
Results
Contributing
License

## Project Overview
This project aims to build a machine learning model to detect fraudulent credit card transactions using a custom dataset. The notebook contains steps for data preprocessing, feature selection, model training, evaluation, and results.

# Dataset
There are no missing values in the dataset

10 variables were dropped from the dataset 
cc_num, first, last, street, lat, long, trans-num, unix_time, merch_lat & merch_long

New variables were derived from the dataset 
Time slot, Week, Weekday & Month from trans_date_trans_time. Age from dob

Feature Chosen for Fraud Detection 
[Category, amt, gender, unix time, previous fraud, number of fraud, session, age, weekday]

# Usage 
# Launch Jupyter Notebook
jupyter notebook

# Open the notebook file
Credit_card_Feature_Selection_Review3.ipynb

# Feature Engineering
# Steps involved in feature engineering, including:
Handling missing values
Encoding categorical features
Scaling numerical features
Data Balancing using SMOTE
Feature Selection 
Classification

# Model Training
Description of algorithms: Random Forest, DNN, 
Hyperparameters tuning: PSO
Cross-validation strategy

# Results
Findings and performance of the model.


Given the dataset, There are no missing values in the dataset
To develop a machine learning model that accurately identifies anomalous (fraudulent) transactions in a dataset of credit card transactions
Sanple Size: Non-Fraud 99479, Fraud: 521
Feature Selection: Autoencoders Classification: Random Forest along With PSO

Model with Autoencoders Feature Selection and  Random Forest classification, With PSO gives the close to ideal results. 

# Best model performance
Confusion matrix





