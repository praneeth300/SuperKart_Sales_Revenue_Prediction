import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="praneeth232/tourism-package-model", filename="best_tourism_package_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# ----------------------------
# Streamlit UI for Sales Prediction
# ----------------------------
st.title("SuperKart Sales Prediction")
st.write("Fill in the product and store details below to predict sales revenue.")

# Collect user input
Product_Weight = st.slider("Product Weight (kg)", 1.0, 30.0, 12.0)
Product_Allocated_Area = st.number_input("Allocated Shelf Area", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
Product_MRP = st.number_input("Product MRP (₹)", min_value=1.0, max_value=500.0, value=150.0, step=1.0)
Store_Establishment_Year = st.selectbox("Store Establishment Year", list(range(1987, 2010)))

Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "High Sugar", "No Sugar"])
Product_Type = st.selectbox("Product Type", [
    "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", 
    "Baking Goods", "Household", "Canned", "Health and Hygiene", 
    "Meat", "Others", "Soft Drinks", "Hard Drinks", "Breads", 
    "Starchy Foods", "Breakfast", "Seafood"
])
Store_Id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
])

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Type': Product_Type,
    'Store_Id': Store_Id,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Sales"):
    predicted_sales = model.predict(input_data)[0]
    st.success(f"Predicted Sales Revenue: ₹{predicted_sales:,.2f}")
