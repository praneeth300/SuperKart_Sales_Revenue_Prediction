import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Sales-Prediction-Experiment2")

# Hugging Face API authentication
api = HfApi(token=os.getenv("HF_TOKEN"))
Xtrain_path = "hf://datasets/praneeth232/superkart/Xtrain.csv"
Xtest_path = "hf://datasets/praneeth232/superkart/Xtest.csv"
ytrain_path = "hf://datasets/praneeth232/superkart/ytrain.csv"
ytest_path = "hf://datasets/praneeth232/superkart/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()   # flatten for regression
ytest = pd.read_csv(ytest_path).values.ravel()

# ----------------------------
# Numerical features
# ----------------------------
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Age_Years'   
]

# ----------------------------
# Categorical features
# ----------------------------
categorical_features = [
    'Product_Sugar_Content',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Type_Category',
    'Product_Id_char'
]

# Define preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model (regression)
xgb_model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid (regression version)
param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150],
    'xgbregressor__max_depth': [3, 4, 5],
    'xgbregressor__colsample_bytree': [0.6, 0.8],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.5, 1, 1.5]
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="r2")
    grid_search.fit(Xtrain, ytrain)

    # Log hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluation
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    train_mae = mean_absolute_error(ytrain, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(ytrain, y_pred_train))
    train_r2 = r2_score(ytrain, y_pred_train)

    test_mae = mean_absolute_error(ytest, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(ytest, y_pred_test))
    test_r2 = r2_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    })

    # Save the model locally
    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "praneeth232/superkart-sales-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_superkart_sales_model_v1.joblib",
        path_in_repo="best_superkart_sales_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
