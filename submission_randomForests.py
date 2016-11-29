
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
import re

titanic = pd.read_csv("data/train.csv")
titanic_test = pd.read_csv("data/test.csv")
title_mapping = {
    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5,
    "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8,
    "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10,
    "Sir": 9, "Capt": 7, "Ms": 2
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
# Step 2: Generating new features
#

print("Step 2: Generating new features")

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
print("    Feature 'family size' added.")

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
print("    Feature 'name lenght' added.")

# Extracting person's title
titles = titanic["Name"].apply(get_title)
titanic["Title"] = map(title_map, titles)
print("    Feature 'title' added.")

# Compressing family id into feature
family_ids = titanic.apply(get_family_id, axis=1)
family_ids[titanic["FamilySize"] < 3] = -1
titanic["FamilyId"] = family_ids
print("    Feature 'family id' added.")

#
# Step 3: Feature selection
#

print("Step 3: Feature selection")

# The columns we'll use to predict the target
predictors = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "FamilySize", "Title", "FamilyId"  # New features
]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation="vertical")
plt.show()

#
# Step 4: Applying Random Forests
# Selecting best features
#

print("Step 4: Applying machine learning")

# Pick only the four best features.
predictors = ["Sex", "Pclass", "Title", "Fare"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
#alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
alg = RandomForestClassifier(
    random_state=1,
    n_estimators=50,
    min_samples_split=4,
    min_samples_leaf=2
)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set (cross validation)
# predictions = alg.predict(titanic_test[predictors])

# Cross validation
kf = cross_validation.KFold(
    titanic.shape[0],
    n_folds=3,
    random_state=1
)

# Calculating score for each fold (3)
scores = cross_validation.cross_val_score(
    alg,
    titanic[predictors],
    titanic["Survived"],
    cv=kf
)

print(scores.mean())
