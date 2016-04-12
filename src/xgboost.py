from sklearn.ensemble import GradientBoostingClassifier

from src.utils import data_path, setup
import pandas as pd
from src.transformers import *
from src.OneHotTransformer import OneHotTransformer, categorical
from src.validation import cross_validation


setup(pd)


def transform_set(name, train=True):
    train_path = data_path(name)
    train_frame = pd.read_csv(train_path)

    categories = None
    if train: categories = train_frame['Category']

    if train:
        del train_frame['Descript']
        del train_frame['Resolution']
        del train_frame['Category']
    del train_frame['Address']
    if not train: del train_frame['Id']

    train_frame['X'] = normalize_features(train_frame['X'])
    train_frame['Y'] = normalize_features(train_frame['Y'])
    train_frame['Times'] = train_frame['Dates'].apply(transform_normalized_time)
    # train_frame = transform_address(train_frame)
    # print(train_frame)

    # train_frame['Year'] = train_frame['Dates'].apply(transform_data_to('year'))
    # train_frame['Month'] = train_frame['Dates'].apply(transform_data_to('month'))
    del train_frame['Dates']

    transformer = OneHotTransformer(categorical(train_frame), train_frame.columns)
    transformer.fit(train_frame)
    train_transformed = transformer.transform_frame(train_frame)

    label_transformed = None
    if train:
        values = sorted(list(set(categories)))
        mapping = {value: index for index, value in enumerate(values)}
        label_transformed = [mapping[cat] for cat in categories]

    print(train_transformed.columns)
    return train_transformed, label_transformed


clf = GradientBoostingClassifier()

train_t, labels_t = transform_set('train.csv')


cross = cross_validation(clf, train_t, labels_t)
# xgboost.fit(train_t, labels_t)
#best estimator parameters: {'max_depth': 15, 'n_estimators': 30, 'max_features': 10}
print(cross, cross.mean())

# test, _ = transform_set('test.csv', False)
# test_prediction = xgboost.predict_proba(test)
# create_submission(test_prediction, 'xgboost.csv')

# max 15 35 on my machine
