
from src.utils import data_path, setup
import pandas as pd
from src.longlat import normalize_features, transform_normalized_time


setup(pd)

# for now only dates, day of week, pd district, resolution, x, y


train_path = data_path('train.csv')
train_frame = pd.read_csv(data_path('train.csv'))

del train_frame['Descript']
del train_frame['Address']

train_frame['X'] = normalize_features(train_frame['X'])
train_frame['Y'] = normalize_features(train_frame['Y'])

train_frame['Dates'] = train_frame['Dates'].apply(transform_normalized_time)


from src.OneHotTransformer import OneHotTransformer, categorical
transformer = OneHotTransformer(categorical(train_frame), train_frame.columns)

transformer.fit(train_frame)
result = transformer.transform_frame(train_frame)

print(result.dtypes)
print("========================================")
not_regex = "^Dates|^PdDistrict|^DayOfWeek|^Resolution|^X|^Y"
train_transformed = result.filter(regex=not_regex)
label_transformed = result.filter(regex="^Category")

print(train_transformed.columns)
print(label_transformed.columns)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

clf = OneVsRestClassifier(LinearSVC(random_state=0))
train_prediction = clf.fit(train_transformed, label_transformed).predict(train_transformed)

print(train_prediction.shape)
