import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import joblib


def main():
    data_path = Path('data') / 'dataset.csv'
    df = pd.read_csv(data_path)

    le = LabelEncoder()
    df['Target_encoded'] = le.fit_transform(df['Target'])

    categorical_features = [
        'Marital status', 'Application mode', 'Application order', 'Course',
        'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
        "Mother's qualification", "Father's qualification", "Mother's occupation",
        "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
    ]

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('category')

    all_columns_except_target = df.drop(columns=['Target', 'Target_encoded'], errors='ignore').columns
    numerical_features = [col for col in all_columns_except_target if col not in categorical_features]

    X_categorical_processed = pd.get_dummies(df[categorical_features], drop_first=True)
    X_processed = pd.concat([df[numerical_features], X_categorical_processed], axis=1)
    feature_columns = X_processed.columns.tolist()
    y = df['Target_encoded']

    X_train_processed, X_test_processed, y_train_processed, y_test_processed = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    gbc_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    models = {
        'RandomForestClassifier': rf_model,
        'GradientBoostingClassifier': gbc_model,
        'XGBClassifier': xgb_model
    }

    print(f'Performing {k_folds}-Fold Stratified Cross-Validation...')
    for name, model in models.items():
        print(f'\nEvaluating {name}...')
        scores = cross_val_score(model, X_train_processed, y_train_processed, cv=skf, scoring='accuracy', n_jobs=-1)
        print(f'  Mean Accuracy: {scores.mean():.4f}')
        print(f'  Standard Deviation of Accuracy: {scores.std():.4f}')

    print('\nStarting Hyperparameter Tuning with RandomizedSearchCV...')

    print('\n--- Tuning RandomForestClassifier ---')
    param_dist_rf = {
        'n_estimators': randint(50, 200),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    random_search_rf = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist_rf,
        n_iter=50,
        cv=skf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search_rf.fit(X_train_processed, y_train_processed)

    print('Best parameters for RandomForestClassifier:', random_search_rf.best_params_)
    print('Best accuracy for RandomForestClassifier:', random_search_rf.best_score_)

    print('\n--- Tuning GradientBoostingClassifier ---')
    param_dist_gbc = {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4)
    }

    random_search_gbc = RandomizedSearchCV(
        estimator=gbc_model,
        param_distributions=param_dist_gbc,
        n_iter=50,
        cv=skf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search_gbc.fit(X_train_processed, y_train_processed)

    print('Best parameters for GradientBoostingClassifier:', random_search_gbc.best_params_)
    print('Best accuracy for GradientBoostingClassifier:', random_search_gbc.best_score_)

    print('\n--- Tuning XGBoostClassifier ---')
    param_dist_xgb = {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }

    random_search_xgb = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist_xgb,
        n_iter=50,
        cv=skf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search_xgb.fit(X_train_processed, y_train_processed)

    print('Best parameters for XGBoostClassifier:', random_search_xgb.best_params_)
    print('Best accuracy for XGBoostClassifier:', random_search_xgb.best_score_)

    model_path = Path('models') / 'best_random_forest_model.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(random_search_rf.best_estimator_, model_path)
    print(f"Best RandomForestClassifier model saved as '{model_path}'.")

    feature_path = model_path.parent / 'feature_columns.json'
    feature_path.write_text(json.dumps(feature_columns, indent=2), encoding='utf-8')

    label_path = model_path.parent / 'label_classes.json'
    label_path.write_text(json.dumps(le.classes_.tolist(), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
