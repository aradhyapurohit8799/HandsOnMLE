""" This script creates a logger and configures it. """

import logging
import logging.config

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s : %(name)s : %(levelname)s : %(funcName)s : %(lineno)d : %(message)s",
            "datefmt": "%Y/%m/%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cnfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """This function configures logger.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    if not cnfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cnfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hndlr in logger.handlers:
            logger.removeHandler(hndlr)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level))
            logger.addHandler(file_handler)

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(getattr(logging, log_level))
            logger.addHandler(stream_handler)

    return logger
