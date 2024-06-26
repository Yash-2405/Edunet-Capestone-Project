# Edunet-Capestone-Project
# Credit Card Fraud Detection
![image](https://github.com/Yash-2405/Edunet-Capestone-Project/assets/115539756/577f6252-6593-4958-8582-53fc7c76fac6)

# Introduction
Credit card fraud detection is a significant challenge in the financial sector. This project demonstrates the use of machine learning to detect fraudulent transactions from a dataset containing credit card transactions. The dataset used is highly imbalanced, with a very small percentage of transactions being fraudulent.

# Dataset
The dataset used in this project can be found on Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders.

# Dataset Description
The dataset includes the following features:

Time: Number of seconds elapsed between this transaction and the first transaction in the dataset.
V1-V28: Result of a PCA transformation. Due to confidentiality issues, the original features are not provided.
Amount: Transaction amount.
Class: Response variable, and it takes value 1 in case of fraud and 0 otherwise.
The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

# Project Structure
.
├── Capestone_Project_Credit_Card_Fraud_Detection.ipynb  # Jupyter Notebook
├── README.md                                            # README file
├── requirements.txt                                     # Dependencies
└── data                                                 # Dataset folder
    └── creditcard.csv                                   # Dataset file
# Dependencies
To install the required dependencies, run:
pip install -r requirements.txt
# Data Preprocessing
Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the Dataset

# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('data/creditcard.csv')

# Dataset Overview
#first 5 rows of the dataset
credit_card_data.head()

#dataset informations
credit_card_data.info()

# Handling the Imbalanced Dataset
The dataset is highly imbalanced:

0 --> Normal Transaction
1 --> Fraudulent Transaction
To address this, we use under-sampling to create a balanced dataset.

# Model Training
We use Logistic Regression for training the model.

#Splitting the data into Features & Targets
X = credit_card_data.drop(columns='Class', axis=1)
Y = credit_card_data['Class']
#Split the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model Evaluation

#Accuracy Score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)

# Results
The model achieves a certain accuracy on both training and testing datasets. Detailed evaluation metrics can be added here.

# Usage
To use the code in this repository, follow these steps:

# 01. Clone the repository:
git clone https://github.com/yourusername/credit-card-fraud-detection.git

# 02. Navigate to the project directory:
cd credit-card-fraud-detection

# 03. Install the dependencies:
pip install -r requirements.txt

# 04. Run the Jupyter Notebook:
jupyter notebook Capestone_Project_Credit_Card_Fraud_Detection.ipynb

# Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.
