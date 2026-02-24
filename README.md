# Titanic Survival Prediction Using Machine Learning

This project focuses on building and comparing multiple supervised machine learning models to predict passenger survival on the Titanic dataset. The goal is to analyze how different classification algorithms perform on structured tabular data after proper preprocessing and feature engineering.

The dataset used is the Titanic dataset available through Seaborn, where the target variable is `survived`. The project begins with data exploration and cleaning. Irrelevant columns were removed, missing values in the `age` column were handled using mean imputation, and rows with missing `embarked` values were dropped. Categorical features were encoded into numerical format to make them suitable for machine learning models. For distance-based algorithms such as KNN and SVM, feature scaling was applied using StandardScaler.

The following machine learning models were implemented and compared:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Gaussian Naive Bayes  
- Decision Tree Classifier  
- Support Vector Machine (RBF Kernel)  

The dataset was split into training and testing sets to evaluate model performance. Each model was assessed using:

- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, and F1-Score (Classification Report)  

Additionally, 5-fold cross-validation was applied (particularly on SVM) to evaluate the model’s generalization capability and reduce overfitting risk.

This project demonstrates a complete machine learning workflow including data preprocessing, feature engineering, model training, performance evaluation, and comparative analysis of multiple classification algorithms.

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

This repository showcases practical implementation of core machine learning concepts and serves as a strong foundation for classification-based ML projects.
