from matplotlib.cbook import mkdirs
from sklearn.linear_model.logistic import LogisticRegression

from src.submission import create_submission
from src.utils import data_path, setup
import pandas as pd
from src.transformers import normalize_features, transform_normalized_time, transform_data_to, transform_coordinates,\
    transform_address
from src.OneHotTransformer import OneHotTransformer, categorical
from src.validation import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



setup(pd)
# for now only dates, day of week, pd district, x, y


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
    print(train_frame)

    # train_frame['Year'] = train_frame['Dates'].apply(transform_data_to('year'))
    # train_frame['Month'] = train_frame['Dates'].apply(transform_data_to('month'))
    del train_frame['Dates']

    transformer = OneHotTransformer(categorical(train_frame), train_frame.columns)
    transformer.fit(train_frame)
    train_transformed = transformer.transform_frame(train_frame)

    return train_transformed, categories


def run_class(depth, num_est, train_transformed, labels, test_transformed, evaluate=True, file_name=None, jobs=1):
    # train_transformed, labels = transform_set('train.csv')

    categories = sorted(list(set(labels)))
    mapping = {value: index for index, value in enumerate(categories)}
    label_transformed = [mapping[cat] for cat in labels]

    # test_transformed, _ = transform_set("test.csv", train=False)

    clf = RandomForestClassifier(max_depth=depth, verbose=2, n_jobs=jobs, n_estimators=num_est)

    # mkdirs(data_path('serialized/model1/'))
    # joblib.dump(clf, data_path('serialized/model1/model1.pkl'))

    print("=======================================")

    if not evaluate:
        clf.fit(train_transformed, label_transformed)
        del train_transformed
        del label_transformed
        test_prediction = clf.predict_proba(test_transformed)
        del test_transformed
        create_submission(test_prediction, file_name)

    if evaluate:
        cross = cross_validation(clf, train_transformed, label_transformed)
        print(cross, cross.mean())
        return cross, cross.mean(), ', '.join(sorted({feature.split('_')[0] for feature in train_transformed.columns}))


train_transformed, labels = transform_set('train.csv')
test_transformed, _ = transform_set("test.csv", train=False)

result_list = []

# for dep in [15]:
#     for num in [25, 30, 35, 40, 50]:
#         print('starting testing classfier for depth={} and number of trees={}'.format(dep, num))
#         result_list.append((num, dep, run_class(dep, num, train_transformed, labels, test_transformed)))
# #
# #
# with open(data_path("evaluation.txt"), 'a') as f:
#     for i in range(len(result_list)):
#         res = result_list[i]
#         if i == 0: f.write(res[2][2] + '\n')
#         f.write('random forests {} trees, {} depth - {} {}'.format(res[0], res[1], res[2][0], res[2][1]) + '\n')
#     f.write('\n')

# max 15 35 on my machine
# run_class(15, 45, train_transformed, labels, test_transformed, evaluate=False, file_name='submission5.csv')



# for res in result_list:
#     print(res)