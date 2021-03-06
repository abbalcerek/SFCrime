from matplotlib.cbook import mkdirs
from sklearn.linear_model.logistic import LogisticRegression

from src.submission import create_submission
from src.utils import data_path, setup
import pandas as pd
from src.transformers import normalize_features, transform_normalized_time
from src.OneHotTransformer import OneHotTransformer, categorical
from src.validation import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import LinearSVC


setup(pd)
# for now only dates, day of week, pd district, x, y


def transform_set(name, train=True):
    train_path = data_path(name)
    train_frame = pd.read_csv(train_path)
    if train:
        del train_frame['Descript']
        del train_frame['Resolution']
    del train_frame['Address']

    train_frame['X'] = normalize_features(train_frame['X'])
    train_frame['Y'] = normalize_features(train_frame['Y'])
    train_frame['Dates'] = train_frame['Dates'].apply(transform_normalized_time)
    transformer = OneHotTransformer(categorical(train_frame), train_frame.columns)
    transformer.fit(train_frame)
    result = transformer.transform_frame(train_frame)

    not_regex = "^Dates|^PdDistrict|^DayOfWeek|^Resolution|^X|^Y"
    train_transformed = result.filter(regex=not_regex)
    label_transformed = None
    if train: label_transformed = result.filter(regex="^Category")

    return train_transformed, label_transformed

train_transformed, label_transformed = transform_set('train.csv')

print(train_transformed.columns)
print(label_transformed.columns)

# clf = OneVsRestClassifier(LogisticRegression(random_state=0))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=10)
clf.fit(train_transformed, label_transformed)
# train_prediction = clf.predict(train_transformed)

# mkdirs(data_path('serialized/model1/'))
# joblib.dump(clf, data_path('serialized/model1/model1.pkl'))

# print(cross_validation(clf, train_transformed, label_transformed))

test_transformed, _ = transform_set("test.csv", train=False)

print("=======================================")


test_prediction = clf.predict_proba(test_transformed)
print(test_prediction[0])


print(sum([t[0][0] for t in test_prediction]))
print(sum([t[0][1] for t in test_prediction]))

reshaped = [[t[i][1] for t in test_prediction] for i in range(len(test_prediction[0]))]

print(reshaped)


print(len(test_prediction), test_prediction[0].shape)
create_submission(reshaped, "submission12.csv")


# score on kaggle = 2.60553