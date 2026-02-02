# Student Dropout Prediction

## Description of the problem
This project predicts student dropout outcomes based on institutional, demographic, and academic features. The goal is to understand key drivers of dropout risk and build a model that can support early intervention decisions.

## Project structure
- `notebook.ipynb`: end-to-end analysis (data prep/cleaning, EDA, feature analysis, model selection and tuning).
- `data/dataset.csv`: dataset used for training and evaluation.
- `models/`: saved model artifacts (if generated).

## Data
The dataset is already committed at `data/dataset.csv`.

If you want to replace or refresh the data, put the new file at the same path and keep the same filename, or update the notebook to point to the new location.

## Instructions on how to run the project
1) Create and activate a Python environment (3.10+ recommended).
2) Install dependencies. If a `requirements.txt` is available, use it. Otherwise, install the common stack used in the notebook:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib jupyter
```

3) Launch the notebook:

```bash
jupyter notebook notebook.ipynb
```

## Notebook contents
The notebook includes:
- Data preparation and data cleaning
- EDA and feature analysis
- Model selection and parameter tuning