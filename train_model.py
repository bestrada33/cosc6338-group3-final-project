#train_model.py
#----------------------------------
# Created By : Beto Estrada
#----------------------------------
""" 
This script acts as the main driver for model training. An model configuration csv file must be passed as a command
line argument. The experiment is then parsed in this file. The config can be placed anywhere, but preferably inside
the `model_configs` folder. The tuning/testing results + model weights for each model are placed in the `results` 
directory. Multiple experiment configuration files can be passed.

Example run: `python train_model.py -f model_configs/model_config.json`
""" 
#---------------------------------------------
# 
#
import argparse
import json
import os
import pandas as pd

from src.ml_pipeline import run_ml_pipeline
from src.utils.file_operations import ensure_dir

from dotenv import load_dotenv

load_dotenv()

use_gpu = os.getenv("USE_GPU")

# If use_gpu is false, then just use CPU. Otherwise available GPUs are used
if use_gpu == 'false':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train_model(data_file_path, model_name):
    # All model tuning & test results are saved under directory 
    # named after model name
    results_directory = f'results/{model_name}/'

    # Ensure results directory is created, if not then create it
    ensure_dir(results_directory)

    # Read in the data file from config file
    df_data = pd.read_csv(data_file_path)
    
    # Run ML pipeline. All models training results will be saved
    # in the results directory specified
    run_ml_pipeline(df_data, results_directory)
    

def parse_config_file(config_file_path):
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)

    data_file_path = config_data.get('data_file_path')
    model_name = config_data.get('model_name')

    return data_file_path, model_name


parser = argparse.ArgumentParser(
    description="Run ML models from one or more JSON configuration files"
)

parser.add_argument(
    "-f", "--model_configuration_files",
    nargs="+",  # One flag, multiple files
    required=True,
    help="One or more .json files with model configurations"
)

args = parser.parse_args()

for config_file in args.model_configuration_files:
    data_file_path, model_name = parse_config_file(config_file)

    train_model(data_file_path, model_name)