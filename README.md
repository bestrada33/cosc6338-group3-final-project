# Machine Learning for Synthetic DNA Classification
COSC 6338

Group 3

Matthew Kastl, Beto Estrada, Sainadth Pagadala, and Malaka Sudhakara Reddy

## Requirements
- [Conda package and environment manager](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)


## Setup
1. Create conda environment based on `environment.yml` file
`conda env create -f environment.yml`
2. Activate the environment
`conda activate ml-env`
3. If you need to add or remove any libraries from the environment, run the command: `conda env update --name ml-env --file environment.yml --prune`


## How to train ML model
0. If you haven't already, activate your conda environment using `conda activate ml-env`
1. Run the command `python train_model.py -f {your_model_configuration_json_files}` to train the model in your model configuration json file.
