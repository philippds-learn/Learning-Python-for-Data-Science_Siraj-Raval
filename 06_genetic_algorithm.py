from tpot import TPOT
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

#load the data
telescope = pd.read_csv('MAGIC Gamma Telescope Data.csv')

#clean the data
telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
tele = telescope_shuffle.reset_index(drop = True)

#store 2 classes
tele['Class'] = tele['Class'].map({'g':0, 'h':1})
tele_class = tele['Class'].values

#split training, testing, and calidation data
training_indices, validation_indices = training_indices
testing_indices = train_test_split(tele.index, stratify = tele_class, train_size = 0.75, test_size = 0.25)

#let genetic programming find best ML model and hyperparameters
tpot = TPOT(generation = 5, verbosity = 2)
tpot.fit(tele.drop('Class', axis = 1).loc[training_indices].values,
    tele.loc[validation_indices, 'Class'].values)

#core the accuracy
tpot.score(tele.drop('Class', axis = 1).loc[validation_indices].values,
    tele.loc[validation_indices, 'Class'].values)

#Exporting the generate code
tpot.export('pipeline.py')
