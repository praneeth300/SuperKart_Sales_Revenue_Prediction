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

# Fix inconsistent sugar content labels
superkart_df.Product_Sugar_Content.replace(to_replace=["reg"], value=["Regular"], inplace=True)

# Extract first 2 chars of Product_Id
superkart_df["Product_Id_char"] = superkart_df["Product_Id"].str[:2]

# Store age feature
superkart_df["Store_Age_Years"] = 2025 - superkart_df.Store_Establishment_Year

# Create perishables vs non-perishables category
perishables = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]
def change(x):
    return "Perishables" if x in perishables else "Non Perishables"

superkart_df['Product_Type_Category'] = superkart_df['Product_Type'].apply(change)

# Drop unnecessary columns
superkart_df = superkart_df.drop(["Product_Id","Product_Type","Store_Id","Store_Establishment_Year"], axis=1)

# ----------------------------
# Define the target variable
# ----------------------------
target = 'Product_Store_Sales_Total'

# ----------------------------
# Numerical features
# ----------------------------
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Age_Years'   # moved here
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

# ----------------------------
# Feature matrix X and target y
# ----------------------------
X = superkart_df[numeric_features + categorical_features]
y = superkart_df[target]

# ----------------------------
# Train-test split
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save to CSV
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False, header=True)
ytest.to_csv("ytest.csv", index=False, header=True)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="praneeth232/superkart",
        repo_type="dataset",
    )
