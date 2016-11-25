import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import pylab as P
import seaborn as sns

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.cross_validation import KFold, cross_val_score

dataset = pd.read_csv('data_all.csv')
CV = dataset.health.reshape(len(dataset.health),1)
data = (dataset.ix[:'energy_categorical', 'dance_categorical', 'liveness_categorical', 'valence_categorical', 'tempo_categorical', 'instrumental_categorical', 'acoustic_categorical', 'popularity_categorical'].values).reshape(len(dataset.health),8)

DT = DecisionTreeClassifier(criterion = "entropy")
DT.fit(data, CV)

with open("")