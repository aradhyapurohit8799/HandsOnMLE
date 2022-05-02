### Median housing value prediction problem ###

# Project Description
In this module, we have ingested, trained and validated the housing dataset with the following ML Models:-
1-> Linear Regression
2-> Decision Tree Regression Model
3-> Random Forest Model :: Fine Tuned Model with GridSearch

# Conda Environment Setup
To setup conda environment, do the following steps:-
1-> Open Anaconda Powershell Prompt
2-> go to the project root directory
3-> type  conda env create --file deploy/conda/conda_env.yml
4-> Now activate the conda environment with conda activate command
# Package INstallation:
Run the following command on Anaconda Powershell Prompt to setup the packages:-

"pip install -e ."
# Data Ingestion:
"python src/Housing_Price_Prediction/ingest_data.py -r data/raw/ -p data/processed/"
# Training Model:
"python src/Housing_Price_Prediction/train.py -d data/processed/housing_train.csv -m artifacts/"
# Model Validation:
"python src/Housing_Price_Prediction/score.py -d data/processed/housing_test.csv -m artifacts/"