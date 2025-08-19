# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/praneeth232/superkart/SuperKart.csv"
superkart_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable
# ----------------------------
target = 'Product_Store_Sales_Total'   # Continuous variable → sales revenue

# ----------------------------
# List of numerical features
# ----------------------------
numeric_features = [
    'Product_Weight',           # Weight of the product
    'Product_Allocated_Area',   # Shelf/area allocated to the product
    'Product_MRP',              # Maximum Retail Price
    'Store_Establishment_Year'  # Year the store was established
]

# ----------------------------
# List of categorical features
# ----------------------------
categorical_features = [
    'Product_Sugar_Content',      # e.g., Low Sugar, Regular
    'Product_Type',               # e.g., Fruits and Vegetables, Snack Foods
    'Store_Id',                   # Store code (OUT001, OUT004, etc.)
    'Store_Size',                 # e.g., Small, Medium, High
    'Store_Location_City_Type',   # e.g., Tier 1, Tier 2, Tier 3
    'Store_Type'                  # e.g., Supermarket Type1, Grocery Store
]

# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------
X = superkart_df[numeric_features + categorical_features]

# ----------------------------
# Define target vector y
# ----------------------------
y = superkart_df[target]

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="praneeth232/superkart",
        repo_type="dataset",
    )
