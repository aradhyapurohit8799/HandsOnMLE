""" This script tests and scores the model"""

import os
import pickle
from argparse import ArgumentParser
from argparse import Namespace
from glob import glob
from logging import Logger

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from Housing_Price_Prediction.logger import configure_logger


def argparse_func() -> Namespace:
    """This takes artifacts folder location and path to test set dataset as inputs

        Returns
        -------
        "models": str,
         "dataset": str,
         "rmse": bool,
         "mae": bool,
         "log_level": str,
         "no_console_log": bool,
         "log_path": str]
    ."""
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="artifacts/",
        help="Directory where the models are stored.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="data/processed/housing_test.csv",
        help="Path to test dataset file.",
    )

    parser.add_argument(
        "--rmse",
        action="store_true",
        help="Shows Root Mean Square Error value.",
    )
    parser.add_argument(
        "--mae", action="store_true", help="Shows Mean Absolute Error value."
    )

    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")

    return parser.parse_args()


def load_data(path: str):
    """
    Loads dataset and splits features and labels.
    Takes training dataset path as input.

    Parameters
    ----------
    path : str
        Path to training dataset csv file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Index 0 is the testing features dataframe.
        Index 1 is the testing labels series.
    """
    df = pd.read_csv(path)
    y = df["median_house_value"].copy(deep=True)
    X = df.drop(["median_house_value"], axis=1)
    return (X, y)


def load_models(path: str):
    """
    Loads models from given directory path.
    Takes model pkl file directory path as input parameter.

    Parameters
    ----------
    path : str
        Path to directory with model pkl files.

    Returns
    -------
    list[sklearn.base.BaseEstimator]
        List of models loaded from pkl files in directory.
    """
    paths = glob(f"{path}/*.pkl")
    models = []

    for path in paths:
        if os.path.isfile(path):
            model = pickle.load(open(path, "rb"))
            models.append(model)

    return models


def score_model(
    model: sklearn.base.BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    args: Namespace,
) -> dict:
    """
    Scores the model.
    Takes BaseEstimator as estimator,
    features and labels for scoring and
    argeparse input arguments for defining whether or not to calculate MAE and RMSE.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Estimator to score.
    X : pd.DataFrame
        Input features dataframe.
    y : pd.Series
        Ground truth labels.
    args : Namespace
        Command line arguments. Used to determine which scores to calculate.

    Returns
    -------
    dict
        Contains calculated scores.
    """
    scores = {}
    scores["R2 score"] = model.score(X, y)
    y_hat = model.predict(X)

    if args.rmse:
        rmse = np.sqrt(mean_squared_error(y, y_hat))
        scores["RMSE"] = rmse

    if args.mae:
        mae = mean_absolute_error(y, y_hat)
        scores["MAE"] = mae

    return scores


def validation(args: Namespace, logger: Logger) -> None:
    """Runs and scores entire test set.

    Parameters
    ----------
    args : Namespace
        Commandline arguments from parse_args.
    logger : Logger
        Logs the outputs.
    """

    X, y = load_data(args.dataset)

    models = load_models(args.models)

    for model in models:
        model_name = type(model).__name__
        scores = score_model(model, X, y, args)
        logger.debug(f"Model: {model_name}")
        for k, v in scores.items():
            logger.debug(f"{k}: {v}")


if __name__ == "__main__":
    args = argparse_func()
    logger = configure_logger(
        log_file=args.log_path,
        log_level=args.log_level,
        console=not args.no_console_log,
    )

    validation(args, logger)
