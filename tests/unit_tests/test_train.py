"""This script is a Unit Test for test.py script"""

import os

from sklearn.svm import SVR

import Housing_Price_Prediction.train as train
from Housing_Price_Prediction.logger import configure_logger

args = train.argparse_func()
logger = configure_logger()


def test_argparse_func():
    """
    Tests argparse_func function.
    """
    assert args.dataset == "data/processed/housing_train.csv"
    assert args.models == "artifacts/"
    assert args.log_level == "DEBUG"
    assert not args.no_console_log
    assert args.log_path == ""


def test_load_data():
    """
    Tests load_data function.
    """
    X, y = train.load_data(args.dataset)
    assert len(X) == len(y)
    assert "median_house_value" not in X.columns
    assert not X.isna().sum().sum()
    assert len(y.shape) == 1


def test_save():
    """
    Tests save_model function.
    """
    svr = SVR()
    train.save_model(svr, args.models)
    name = type(svr).__name__
    assert os.path.isfile(f"{args.models}/{name}.pkl")
    os.remove(f"{args.models}/{name}.pkl")
