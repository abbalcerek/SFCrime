import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.utils import data_path, setup
import pandas as pd

setup()


train_path = data_path('train.csv')
train_frame = pd.read_csv(train_path)
train_frame['Descript'] = train_frame['Descript'].apply(lambda des: re.sub('[\(\),]', '', des))

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ['clf', MultinomialNB()],
])

text_clf = text_clf.fit(train_frame['Descript'], train_frame["Category"])
print(train_frame.ix[0]['Descript'], train_frame.ix[0]['Category'])

prediction = text_clf.predict_proba(train_frame['Descript'])

print(prediction[0])
print(text_clf.classes_)

# TODO: validate

test_path = data_path('test.csv')
test_frame = pd.read_csv(test_path)

# lolz no description in test

# test_frame['Descript'] = test_frame['Descript'].apply(lambda des: re.sub('[\(\),]', '', des))
# test_prediction = text_clf.predict(test_frame['Descript'])
# create_submission(test_prediction, 'description_sub.csv')
