#!bin/usr/python3

""" Getting started with Kaggle: Titanic challenge.
Learning ML and how to compete at Kaggle's challenges by doing the Titanic exercise.
See more:
    - https://www.dataquest.io/mission/74/getting-started-with-kaggle
    - https://www.kaggle.com/c/titanic
Author: matheus_csa
Date: 5th November 2016
Revised: 5th November 2016
"""

from __future__ import division
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold

titanic = pd.read_csv("data/train.csv")
titanic_test = pd.read_csv("data/test.csv")

#
# Step 1: Cleaning and normalizing data
#

print("Step 1: Cleaning and normalizing data")

# Filling missing values
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

# Changing non-numeric values to Enum
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Since the port S is the most common, we fill in all missing values before
# transforming into Enum
titanic.loc[pd.isnull(titanic["Embarked"]), "Embarked"] = "S"
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# Cleaning and normalizing test
# Filling missing values with the median age
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())

# Changing non-numeric values to Enum
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# Since the port S is the most common, we fill in all missing values before
# transforming into Enum
titanic_test.loc[pd.isnull(titanic_test["Embarked"]), "Embarked"] = "S"
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

#
# Step 2: Applying machine learning using modified dataset
# Using Logistic Regression as machine learning model.
#

print("Step 2: Applying machine learning")

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Choosing model
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set (cross validation)
predictions = alg.predict(titanic_test[predictors])

#
# Step 3: Saving submission
# Create a new dataframe with only the columns Kaggle wants from the dataset.
#

submission_name = "submission_{0}.csv".format(time.strftime("%y%m%d_%H%M%S"))

print("Step 3: generating submission {0}".format(submission_name))

submission = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv(
    index=False,
    path_or_buf="data/{0}".format(submission_name)
)
