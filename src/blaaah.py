from xgboost import XGBClassifier
from matplotlib.cbook import mkdirs
from sklearn.linear_model.logistic import LogisticRegression
from src.submission import create_submission
from src.utils import data_path, setup
import pandas as pd
from src.transformers import *
from src.OneHotTransformer import OneHotTransformer, categorical
from src.validation import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


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

    train_frame['Year'] = train_frame['Dates'].apply(transform_data_to('year'))
    print("sdfasdfsadfasdfasdf")
    print(train_frame.columns)
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

    # print(train_transformed.columns)
    return train_transformed, label_transformed


# xgboost = XGBClassifier()
clf = RandomForestClassifier(verbose=1, n_jobs=1, max_depth=15, n_estimators=30, max_features=10)

est = [10, 20, 30, 40, 50, 60]
depth = [5, 7, 10, 15]
max_f = ['auto', 5, 10, 20]


train_t, labels_t = transform_set('train.csv')

# print(train_t.shape, len(labels_t))

# estimator = RandomizedSearchCV(clf,
#                                param_distributions=dict(
#                                     max_depth=depth,
#                                     n_estimators=est,
#                                     max_features=max_f
#                                ),
#                                scoring='log_loss',
#                                verbose=3,
#                                n_iter=15
#                          )

# estimator.fit(train_t, labels_t)
# print('estimator parameters: ' + str(estimator.best_params_))
#
# cross = cross_validation(estimator.best_estimator_, train_t, labels_t)
cross = cross_validation(clf, train_t, labels_t)
# clf.fit(train_t, labels_t)
#best estimator parameters: {'max_depth': 15, 'n_estimators': 30, 'max_features': 10}
print(cross, cross.mean())

# test, _ = transform_set('test.csv', False)
# test_prediction = clf.predict_proba(test)
# create_submission(test_prediction, 'sub_blah.csv')

# max 15 35 on my machine
