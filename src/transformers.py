
from sklearn.preprocessing import StandardScaler


def transform_normalized_time(date):
    hours, minutes, seconds = date.split(' ')[1].split(':')
    in_seconds = int(hours) * 60 * 60 + int(minutes) * 60 + int(seconds)
    return in_seconds / float(24 * 60 * 60)


def normalize_features(features):
    from sklearn.preprocessing import maxabs_scale
    return maxabs_scale(features).reshape(1, -1)[0]


def transform_coordinates(frame):
    xy_scaler = StandardScaler()
    xy_scaler.fit(frame)
    return xy_scaler.transform(frame)


def transform_data(date):
    return date.split(' ').split(':')


def transform_data_to(name):
    from datetime import datetime

    def transform(date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        year, month = str(date_time.year), str(date_time.month)
        if name == 'year': return year
        if name == 'month': return month
        if name == 'day':
            return str(date_time.weekday())

    return transform


def __address_to_abbs(address):
    import re
    no_numbers = re.sub('(^|\s)[0-9]*(\s|$)', '', address).strip()
    return {street.strip().split(' ')[-1] for street in no_numbers.split('/') if street.strip()}


def transform_address(dataframe):
    import pandas as pd
    import numpy as np
    short_abbs = {'I-280', 'WK', 'FERLINGHETTI', 'HWYHY', 'RW', 'BUFANO', 'PARK'}
    abbs = {'RD', 'TER', 'CR', 'ST', 'I-280', 'PL', 'WK', 'MAR', 'LN', 'BL', 'FERLINGHETTI', 'HWYHY', 'WAY', 'RW', 'CT',
            'DR', 'TR', 'PALMS', 'STWY', 'BUFANO', 'AL', 'EX', 'HWY', 'AV', 'HY', 'I-80', 'WY', 'PZ', 'PARK'}
    colls = abbs.difference(short_abbs)
    colls.add('OTH')
    encoded = pd.DataFrame(0, index=np.arange(len(dataframe)), columns=colls)
    dataframe['Address'] = dataframe['Address'].apply(__address_to_abbs)
    for (index, value) in enumerate(dataframe['Address']):
        print(value)
        for abb in value:
            if abb in short_abbs: abbv = 'OTH'
            else: abbv = abb
            encoded.set_value(index, abbv, 1)
    del dataframe['Address']
    return pd.concat([dataframe, encoded], axis=1)


if __name__ == '__main__':
    import time
    from time import mktime
    from datetime import datetime
    from time import gmtime, strftime

    date = '2016-04-17 23:00:00'
    timeInstance = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    print(timeInstance.year)

    def transform_date_test():
        year_transformer = transform_data_to("year")
        date = '2016-04-17 23:00:00'
        result = year_transformer(date)
        assert result == '2016'

        day_transformer = transform_data_to("day")
        assert day_transformer(date) == '6'

        month_transformer = transform_data_to("month")
        assert month_transformer(date) == '4'


    def transform_address_test():
        from src.utils import data_path
        from bokeh.models import pd
        train_path = data_path('train.csv')
        train_frame = pd.read_csv(train_path)

        print(train_frame['Address'].apply(__address_to_abbs))

    transform_address_test()
