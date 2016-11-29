
""" Getting started with Kaggle: Titanic challenge.
Learning ML and how to compete at Kaggle's challenges by doing the Titanic exercise.
See more:
    - https://www.dataquest.io/mission/75/improving-your-submission/3/implementing-a-random-forest
    - https://www.kaggle.com/c/titanic
Author: matheus_csa
Date: 20th November 2016
Revised: 20th November 2016
"""

from __future__ import division
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import operator
import re
import time

titanic = pd.read_csv("data/train.csv")
titanic_test = pd.read_csv("data/test.csv")
title_mapping = {
    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5,
    "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8,
    "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10,
    "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 9
}
family_id_mapping = {}

#
# Private methods
#


def title_map(title):
    return title_mapping[title]


def get_title(name):
    """
    Use a regular expression to search for a title.
    Titles always consist of capital and lowercase letters, and end with a period.
    """
    title_search = re.search(" ([A-Za-z]+)\.", name)
    if title_search:
        return title_search.group(1)
    return ""


def get_family_id(row):
    """A function to get the id given a row."""

    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]

    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

#
# Step 1: Cleaning and normalizing train data
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
# Step 2: Generating new features in test and train
#

print("Step 2: Generating new features")

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]
print("    Feature 'family size' added.")

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
print("    Feature 'name lenght' added.")

# Extracting person's title
titles = titanic["Name"].apply(get_title)
titanic["Title"] = map(title_map, titles)

titles_test = titanic_test["Name"].apply(get_title)
titanic_test["Title"] = map(title_map, titles_test)

print("    Feature 'title' added.")

# Compressing family id into feature
family_ids = titanic.apply(get_family_id, axis=1)
family_ids[titanic["FamilySize"] < 3] = -1
titanic["FamilyId"] = family_ids

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids

print("    Feature 'family id' added.")

#
# Step 3: Ensembiling Random Forest and Logistic Regression models
# Calculating ensembling accuracy
#

print("Step 3: Applying machine learning")

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression,
# and everything with the gradient boosting classifier.
algorithms = [
    [
        # GradientBoosting builds an additive model in a forward stage-wise fashion;
        # it allows for the optimization of arbitrary differentiable loss functions.
        GradientBoostingClassifier(
            random_state=1,
            n_estimators=25,
            max_depth=3
        ),
        ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
    ],
    [
        # Logistic Regression (aka logit, MaxEnt) classifier.
        LogisticRegression(
            random_state=1
        ),
        ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]
    ]
]

# Cross validation
kf = cross_validation.KFold(
    titanic.shape[0],
    n_folds=3,
    random_state=1
)

# Training models
predictions = []

for train, test in kf:

    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []

    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:

        # Fit the algorithm on the training data
        alg.fit(titanic[predictors].iloc[train, :], train_target)

        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)

    # Use a simple ensembling scheme -- just average
    # the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2

    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    test_predictions = test_predictions.astype(int)
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("    Accuracy is {0}".format(accuracy))

#
# Step 4: Generating predictions on titanic_test
#

print("Step 4: Generating predictions on titanic_test")

algorithms = [
    [
        GradientBoostingClassifier(
            random_state=1,
            n_estimators=25,
            max_depth=3
        ),
        ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
    ],
    [
        LogisticRegression(random_state=1),
        ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]
    ]
]

full_predictions = []
for alg, predictors in algorithms:

    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])

    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:, 1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)

#
# Step 5: Saving submission
# Create a new dataframe with only the columns Kaggle wants from the dataset.
#

submission_name = "submission_{0}.csv".format(time.strftime("%y%m%d_%H%M%S"))
print("Step 5: Saving submission {0}".format(submission_name))

submission = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv(
    index=False,
    path_or_buf="data/submissions/{0}".format(submission_name)
)
