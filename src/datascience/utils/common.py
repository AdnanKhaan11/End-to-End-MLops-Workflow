import sys
import yaml
import json
import os
from ensure import ensure_annoatation
from pathlib import Path
from src.datascience import logger
import pandas as pd
import joblib
from box.exceptions import BoxValueError
from box import ConfigBox as config_box


@ensure_annoatation
def read_yaml(yaml_file_path: Path) -> config_box:
    """
    this functiona will  read the yaml file and return the
    content in the form of config box
    Args:
        output_file (list): list of output file
        yaml_file (str): yaml file path

    return config_boc: confib box object

    """
    try:
        with open(yaml_file_path) as f:
            content = yaml.safe_load(f)
            config_box = config_box(content)
            logger.info(f"yaml file {yaml_file_path} read successfully")

            return config_box
    except BoxValueError as e:
        logger.error(f"Error in yaml file {yaml_file_path}: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error reading yaml file {yaml_file_path}: {e}")
        raise e


@ensure_annoatation
def create_directory(dir_path: list) -> None:
    """
    this function will create the directory if it does not exist
    Args:
        dir_path (Path): directory path
    return None"""

    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"directory {dir_path} created successfully")

    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        raise e


@ensure_annoatation
def save_json(json_file_path: Path, content: dict) -> None:
    """this  function will save the json file
    args:
        json_file_path (Path): json file path
        content (dict): content to be saved in the json file
    """
    try:
        with open(json_file_path, "w") as f:
            json.dump(content, f, indent=4)

            logger.info(f"json file {json_file_path} saved successfully")

    except Exception as e:
        logger.error(f"Error saving json file {json_file_path}: {e}")
        raise e

    # load json file and return the content in the form of config box


@ensure_annoatation
def load_json(json_file_path: Path) -> config_box:
    """this function will load the json file and return  the content in the form of config box"""
    try:
        with open(json_file_path) as f:
            content = json.load(f)
            config_box = config_box(content)
            logger.info(f"json file {json_file_path} loaded successfully")

            return config_box
    except Exception as e:
        logger.error(f"Error loading json file {json_file_path}: {e}")
        raise e


def save_model(model, model_file_path: Path) -> None:
    """this function will save the model using joblib
    args:
        model: model to be saved
        model_file_path (Path): model file path
    """
    try:
        joblib.dump(model, model_file_path)
        logger.info(f"model saved successfully at {model_file_path}")

    except Exception as e:
        logger.error(f"Error saving model at {model_file_path}: {e}")
        raise e
