from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from src.utils import data_path, setup
import pandas as pd

setup()

# for now only dates, day of week, pd district, resolution, x, y


train_path = data_path('train.csv')
train_frame = pd.read_csv(data_path('train.csv'))


print(train_frame.ix[:3])

del train_frame['Descript']
del train_frame['Address']

print(train_frame.ix[:3])

from src.longlat import normalize_features, transform_normalized_time
train_frame['X'] = normalize_features(train_frame['X'])
train_frame['Y'] = normalize_features(train_frame['Y'])


print(train_frame.ix[:3])

train_frame['Dates'] = train_frame['Dates'].apply(transform_normalized_time)

print(train_frame.ix[:3])


from src.OneHotTransformer import OneHotTransformer, categorical
transformer = OneHotTransformer(categorical(train_frame), train_frame.columns)

transformer.fit(train_frame)
result = transformer.transform_frame(train_frame)
print(result.ix[:3])


print(result.dtypes)
print("========================================")
not_regex = "^Dates|^PdDistrict|^DayOfWeek|^Resolution|^X|^Y"
train_transformed = result.filter(regex=not_regex)
label_transformed = result.filter(regex="^Category")

print(train_transformed.columns)
print(label_transformed.columns)