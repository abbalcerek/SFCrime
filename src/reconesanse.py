from src.utils import data_path, setup
import pandas as pd
import numpy as np

setup(pd)


train_path = data_path('train.csv')
train_frame = pd.read_csv(data_path('train.csv'))

columns = list(train_frame.columns)

print(columns)
# print(train_frame['Address'].describe())

# no nulls in data

print(train_frame.ix[0:4])
print(set(train_frame['DayOfWeek']))
#
# print(train_frame[['X', 'Y']].describe())
#
# from sklearn.preprocessing import Normalizer, maxabs_scale, minmax_scale
#
# normalizer = Normalizer()
#
# X = train_frame[['X']]
#
# normalized = normalizer.fit_transform(train_frame[['X']])
#
# print(np.unique(normalized))
#
# scaled = minmax_scale(X).reshape(1, -1)
# print(min(scaled), max(scaled))
#
# print(scaled.shape)
#
# series = pd.Series(scaled[0])
# print(series.describe())
#
#
# np.histogram(series)
#
# import matplotlib.pyplot as plt
# # plt.hist(series, bins=400)
# # plt.show()
#
# for c in train_frame.columns:
#     print("============={}============".format(c))
#     print(train_frame[[c]].describe())
#     print(np.unique(train_frame[[c]]))
#
#
# NONE_resolution_cases = train_frame[train_frame['Resolution'] == 'NONE']
# print("resolution NONE count: {}".format(len(NONE_resolution_cases)))
#
# target = 'Category'
# print("============={}============".format(target))
# print(NONE_resolution_cases[[target]].describe())
# print(np.unique(NONE_resolution_cases[[target]]))

# def plot_categories()

# np.histogram()
