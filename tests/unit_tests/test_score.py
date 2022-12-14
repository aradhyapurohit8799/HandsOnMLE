"""Unit test for score.py script"""

import Housing_Price_Prediction.score as score
from Housing_Price_Prediction.logger import configure_logger

args = score.argparse_func()
logger = configure_logger()


def test_argparse_func():
    """
    Tests argparse_func() function.
    """
    assert args.dataset == "data/processed/housing_test.csv"
    assert args.models == "artifacts/"
    assert args.log_level == "DEBUG"
    assert not args.no_console_log
    assert args.log_path == ""


def test_load_data():
    """
    Tests load data function.
    """
    X, y = score.load_data(args.dataset)
    assert len(X) == len(y)
    assert "median_house_value" not in X.columns
    assert not X.isna().sum().sum()
    assert len(y.shape) == 1


def test_load_models():
    """
    Tests load models function.
    """
    models = score.load_models(args.models)
    assert len(models) == 3


def test_score():
    """
    Tests score_model function.
    """
    X, y = score.load_data(args.dataset)
    models = score.load_models(args.models)
    scores = models[0].score(X, y)
    assert score.score_model(models[0], X, y, args)["R2 score"] == scores
