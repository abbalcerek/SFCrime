# mb it should be a class coz mb it has to hold position of fist points on grid

# to cluster points


# def long_lat_to_grid(long, lat):
#     x = ...
#     y = ...
#     return x, y


def transform_normalized_time(date):
    hours, minutes, seconds = date.split(' ')[1].split(':')
    in_seconds = int(hours) * 60 * 60 + int(minutes) * 60 + int(seconds)
    return in_seconds / float(24 * 60 * 60)


def normalize_features(features):
    from sklearn.preprocessing import maxabs_scale
    return maxabs_scale(features).reshape(1, -1)[0]


def transform_data(date):
    return date.split(' ').split(':')


def transform_data_to(name):
    import time
    from time import mktime
    from datetime import datetime

    def transform(date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        year, month = str(date_time.year), str(date_time.month)
        if name == 'year': return year
        if name == 'month': return month
        if name == 'day':
            return str(date_time.weekday())
    return transform


if __name__ == '__main__':
    import time
    from time import mktime
    from datetime import datetime
    from time import gmtime, strftime
    date = '2016-04-17 23:00:00'
    timeInstance = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    print(timeInstance.year)
    # dt = datetime.fromtimestamp(mktime(timeInstance))
    # print(dt.weekday())


    def transform_date_test():
        year_transformer = transform_data_to("year")
        date = '2016-04-17 23:00:00'
        result = year_transformer(date)
        assert result == '2016'

        day_transformer = transform_data_to("day")
        assert day_transformer(date) == '6'

        month_transformer = transform_data_to("month")
        assert month_transformer(date) == '4'

    transform_date_test()