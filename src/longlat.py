# mb it should be a class coz mb it has to hold position of fist points on grid

# to cluster points


def long_lat_to_grid(long, lat):
    x = ...
    y = ...
    return x, y


def transform_normalized_time(date):
    hours, minutes, seconds = date.split(' ')[1].split(':')
    in_seconds = int(hours) * 60 * 60 + int(minutes) * 60 + int(seconds)
    return in_seconds / float(24 * 60 * 60)


def normalize_features(features):
    from sklearn.preprocessing import maxabs_scale
    return maxabs_scale(features).reshape(1, -1)[0]


