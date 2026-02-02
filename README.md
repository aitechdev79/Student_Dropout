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

## Railway deployment
1) Push this repo to GitHub.
2) In Railway, create a new project and deploy from GitHub.
3) Railway will detect the `Dockerfile` and build the service automatically.
4) Ensure the model artifacts exist in `models/` (`best_random_forest_model.joblib`, `feature_columns.json`, `label_classes.json`).
5) After deploy, test the service:

```bash
curl -X POST "$RAILWAY_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"Marital status":1,"Application mode":1,"Application order":1,"Course":1,"Daytime/evening attendance":1,"Previous qualification":1,"Nacionality":1,"Mother'"'"'s qualification":1,"Father'"'"'s qualification":1,"Mother'"'"'s occupation":1,"Father'"'"'s occupation":1,"Displaced":0,"Educational special needs":0,"Debtor":0,"Tuition fees up to date":1,"Gender":1,"Scholarship holder":0,"International":0}'
```

## Try it out (Swagger UI)
Open:
`https://studentdropout-production.up.railway.app/docs#/default/predict_predict_post`

Request body:

```json
{
  "Marital status": 1,
  "Application mode": 1,
  "Application order": 1,
  "Course": 1,
  "Daytime/evening attendance": 1,
  "Previous qualification": 1,
  "Nacionality": 1,
  "Mother's qualification": 1,
  "Father's qualification": 1,
  "Mother's occupation": 1,
  "Father's occupation": 1,
  "Displaced": 0,
  "Educational special needs": 0,
  "Debtor": 0,
  "Tuition fees up to date": 1,
  "Gender": 1,
  "Scholarship holder": 0,
  "International": 0
}
```

## Notebook contents
The notebook includes:
- Data preparation and data cleaning
- EDA and feature analysis
- Model selection and parameter tuning
