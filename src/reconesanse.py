from src.utils import data_path, setup
import pandas as pd
import numpy as np

setup()


train_path = data_path('train.csv')
train_frame = pd.read_csv(data_path('train.csv'))

columns = list(train_frame.columns)

print(train_frame['Address'].describe())

# no nulls in data
for column in columns:
    print(train_frame[column].isnull().sum())