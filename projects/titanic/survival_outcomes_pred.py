"""
Author: Yuri Silva

Objetive: Predict survival outcomes from the 1912 Titanic disaster
based on each passenger’s features, such as sex and age.

Date: 18/04/19

Original code author: Udacity
"""

import numpy as np
import pandas as pd
import matplotlib
from IPython.display import display
import ML.Titanic.visuals as vs
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """

    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):

        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)

    else:
        return "Number of predictions does not match number of outcomes!"


def predictions_0(data):
    """ Model with no features, always predict that the passengers died"""
    predictions = []
    for _, passenger in data.iterrows():
        predictions.append(0)

    return pd.Series(predictions)



in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# display(full_data.head())


outcomes = full_data['Survived']
data = full_data.drop('Survived', axis=1)

# display(data.head())
# display(outcomes.head())

# predictions = pd.Series(np.ones(5, dtype=int))
# predictions = predictions_0(data)
#
# display(predictions)
#
# print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Sex')

