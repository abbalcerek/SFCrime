from os.path import realpath, dirname, join

curr_path = realpath(__file__)
DATA_PATH = join(dirname(dirname(curr_path)), 'data')


def data_path(path=""):
    return join(DATA_PATH, path)


def setup():
    import pandas as pd
    import warnings
    pd.set_option('expand_frame_repr', False)
    pd.options.mode.chained_assignment = None
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings('ignore')
