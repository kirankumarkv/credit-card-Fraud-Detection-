Given the dataset, There are no missing values in the dataset
To develop a machine learning model that accurately identifies anomalous (fraudulent) transactions in a dataset of credit card transactions
Sanple Size: Non-Fraud 99479, Fraud: 521
Feature Selection: Autoencoders Classification: Random Forest along With PSO

Model with Autoencoders Feature Selection and  Random Forest classification, With PSO gives the close to ideal results. 

Classification Report on Original Data:
   precision 		recall     f1-score support
0 	1.00 		1.00 		1.00 	99479 
1 	0.56 		0.99 		0.71 	521
accuracy 				1.00 100000 
Confusion Matrix on Original Data: [[99069 410]
                                      [ 3 518]]



