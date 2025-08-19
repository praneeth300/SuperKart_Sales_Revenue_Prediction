import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import datetime

# Download the model from the Model Hub
model_path = hf_hub_download(
    repo_id="praneeth232/superkart-sales-model",
    filename="best_superkart_sales_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# ----------------------------
# Streamlit UI for Sales Prediction
# ----------------------------
st.title("SuperKart Sales Prediction")
st.write("Fill in the product and store details below to predict sales revenue.")

# Collect user input
Product_Weight = st.slider("Product Weight (kg)", 0.5, 50.0, 10.0, step=0.5)
Product_Allocated_Area = st.number_input("Allocated Shelf Area (0–1 scale)", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
Product_MRP = st.number_input("Product MRP (₹)", min_value=1.0, max_value=500.0, value=150.0, step=1.0)

# Store establishment year → derive Store_Age_Years
Store_Establishment_Year = st.selectbox("Store Establishment Year", list(range(1985, 2011)))
current_year = datetime.datetime.now().year
Store_Age_Years = current_year - Store_Establishment_Year

Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "High Sugar", "No Sugar"])
Product_Type_Category = st.selectbox("Product Category", ["Perishables", "Non Perishables"])
Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])  # adjust based on dataset

Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
])

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Age_Years': Store_Age_Years,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type,
    'Product_Type_Category': Product_Type_Category,
    'Product_Id_char': Product_Id_char
}])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Sales"):
    predicted_sales = model.predict(input_data)[0]
    st.success(f"Predicted Sales Revenue: ₹{predicted_sales:,.2f}")
