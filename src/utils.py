from os.path import realpath, dirname, join

curr_path = realpath(__file__)
DATA_PATH = join(dirname(dirname(curr_path)), 'data')


def data_path(path=""):
    return join(DATA_PATH, path)


def setup():
    import pandas as pd
    pd.set_option('expand_frame_repr', False)
