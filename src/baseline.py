from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from src.submission import create_submission
from src.utils import data_path, setup
import pandas as pd
import numpy as np

setup(pd)


def to_singleton(iterable):
    return [[elem] for elem in iterable]


train_path = data_path('train.csv')
train_frame = pd.read_csv(data_path('train.csv'))

submission_size = 884262

classes = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE',
           'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING',
           'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
           'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY',
           'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
           'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

category = train_frame['Category']
mapping = {clazz: num for (num, clazz) in enumerate(classes)}

most_freq_class = Counter(category).most_common()[0][0]

predicted = category.apply(lambda cat: mapping[most_freq_class])
expected = category.apply(lambda cat: mapping[cat])

mlb = MultiLabelBinarizer()

expected_b = mlb.fit_transform(to_singleton(expected))
predicted_b = mlb.transform(to_singleton(predicted))

for (clazz, count) in Counter(category).most_common():
    print("{}\t{}".format(clazz, count))

# todo: use validation.py
print("Accuracy on training: {}".format(accuracy_score(expected_b, predicted_b)))
print("Log los on training: {}".format(log_loss(expected_b, predicted_b)))

test_prediction = np.full((submission_size, len(predicted_b[0])), predicted_b[0])
create_submission(test_prediction, 'baseline_sub.csv')

