from sklearn.metrics import accuracy_score, classification_report

from src.utils import data_path, setup
import pandas as pd

setup()

train_path = data_path('train.csv')
train_frame = pd.read_csv(data_path('train.csv'))
print(train_frame)

classes = set(train_frame['Category'])
zipped = zip(classes, enumerate(classes))

print(train_frame.columns)
print(classes)

from collections import Counter
print(Counter(train_frame['Category']).most_common())

expected = None
predicted = None

print(accuracy_score(expected, predicted))
print(classification_report(expected, predicted))