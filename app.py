import streamlit as st
import pickle
import pandas as pd

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Energy Efficiency Prediction",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------------------------------
# Load Model and Scaler
# ----------------------------------------------------
with open("lasso_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.title("🏠 Energy Efficiency Prediction System")

st.markdown("""
Predict the **Cooling Load** (or Heating Load if you trained on it)
using a Machine Learning model built with **Lasso Regression**.
""")

st.divider()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.header("Project Information")

st.sidebar.info("""
**Model:** Lasso Regression

**Algorithm Type:** Regression

**Dataset:** Energy Efficiency Dataset

**Target:** Cooling Load
""")

# ----------------------------------------------------
# Input Section
# ----------------------------------------------------
st.subheader("Enter Building Parameters")

col1, col2 = st.columns(2)

with col1:

    relative_compactness = st.number_input(
        "Relative Compactness",
        min_value=0.50,
        max_value=1.00,
        value=0.98,
        step=0.01
    )

    surface_area = st.number_input(
        "Surface Area",
        value=514.5
    )

    wall_area = st.number_input(
        "Wall Area",
        value=294.0
    )

    roof_area = st.number_input(
        "Roof Area",
        value=110.25
    )

with col2:

    overall_height = st.selectbox(
        "Overall Height",
        [3.5, 7.0]
    )

    orientation = st.selectbox(
        "Orientation",
        [2,3,4,5]
    )

    glazing_area = st.selectbox(
        "Glazing Area",
        [0.0,0.10,0.25,0.40]
    )

    glazing_distribution = st.selectbox(
        "Glazing Area Distribution",
        [0,1,2,3,4,5]
    )

# ----------------------------------------------------
# Prediction Button
# ----------------------------------------------------
if st.button("🔍 Predict Energy Load", use_container_width=True):

    sample = pd.DataFrame({
        "Relative_Compactness":[relative_compactness],
        "Surface_Area":[surface_area],
        "Wall_Area":[wall_area],
        "Roof_Area":[roof_area],
        "Overall_Height":[overall_height],
        "Orientation":[orientation],
        "Glazing_Area":[glazing_area],
        "Glazing_Area_Distribution":[glazing_distribution]
    })

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    st.success(f"### Predicted Energy Load : {prediction[0]:.2f}")

# ----------------------------------------------------
# Sample Input
# ----------------------------------------------------
st.divider()

st.subheader("📋 Sample Input")

sample_df = pd.DataFrame({
    "Feature":[
        "Relative Compactness",
        "Surface Area",
        "Wall Area",
        "Roof Area",
        "Overall Height",
        "Orientation",
        "Glazing Area",
        "Glazing Distribution"
    ],
    "Value":[
        0.98,
        514.5,
        294,
        110.25,
        7,
        2,
        0.25,
        3
    ]
})

st.table(sample_df)

# ----------------------------------------------------
# About
# ----------------------------------------------------
st.divider()

st.subheader("ℹ️ About the Project")

st.write("""
This application predicts the energy efficiency of residential buildings
using a **Lasso Regression** model trained on the **Energy Efficiency Dataset**.

### Input Features
- Relative Compactness
- Surface Area
- Wall Area
- Roof Area
- Overall Height
- Orientation
- Glazing Area
- Glazing Area Distribution

### Output
Predicted Cooling Load (or Heating Load depending on the trained model).
""")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.caption("Developed using Streamlit | Machine Learning Project")
