# from sklearn.externals import joblib
import numpy as np
import pandas as pd

# joblib.dump(clf, 'filename.pkl')
from src.utils import data_path

train = data_path('train.csv')['Category']
labels = train['Category']
del train

classifiers_outputs_test = ['train1.pkl']
classifiers_outputs_train = ['test1.pkl']


def create_dataset(paths):
    frames = [pd.read_csv(data_path(path)) for path in paths]
    return pd.concat(frames, axis=1)
