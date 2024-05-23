# Machine Learning Algorithms Implementation

## Overview
This repository contains implementations of various machine learning algorithms and techniques using Python. The project covers the implementation of gradient descent, logistic regression, Naive Bayes, Support Vector Machines (SVM), decision trees, and techniques for handling imbalanced datasets. Each algorithm is demonstrated with appropriate datasets, showing data preprocessing, model training, evaluation, and comparison.

## Features
- **Gradient Descent Algorithm**
  - Implemented logistic regression using gradient descent.
  - Evaluated the model on the Breast Cancer Wisconsin dataset.
  - Applied L1 and L2 regularization and compared the results.

- **Naive Bayes Algorithm**
  - Implemented Gaussian Naive Bayes for classification.
  - Evaluated the model on the dataset and compared it with scikit-learn's GaussianNB implementation.

- **Support Vector Machines (SVM)**
  - Implemented SVM with different kernel functions: linear, polynomial, and RBF.
  - Evaluated model performance using various parameters and compared results.
  - Used SGDClassifier for SVM implementation.

- **Decision Trees**
  - Implemented decision trees and evaluated the impact of different hyperparameters.
  - Compared the results of models with varying tree depths and split criteria.

- **Handling Imbalanced Datasets**
  - Addressed the problem of imbalanced datasets using techniques like SMOTE.
  - Implemented pipelines with different machine learning algorithms and evaluated their performance on imbalanced data.

## Dataset
The primary dataset used in these implementations is the Breast Cancer Wisconsin dataset, which can be accessed from scikit-learn's dataset module. Additionally, for handling imbalanced datasets, the H1N1 and Seasonal Flu Vaccines dataset from Kaggle is used.

- [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [H1N1 and Seasonal Flu Vaccines dataset](https://www.kaggle.com/datasets/arashnic/flu-data)

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.6 or higher
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

You can install the required packages using the following command:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
