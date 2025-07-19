
import streamlit as st
import numpy as np
from src.predict import predict

st.title("Tree Species Classification")

sepal_len = st.slider("Sepal Length", 0.0, 10.0, 5.0)
sepal_wid = st.slider("Sepal Width", 0.0, 10.0, 3.0)
petal_len = st.slider("Petal Length", 0.0, 10.0, 1.5)
petal_wid = st.slider("Petal Width", 0.0, 10.0, 0.2)

if st.button("Predict Species"):
    species = predict([sepal_len, sepal_wid, petal_len, petal_wid])
    st.success(f"Predicted Species: {species}")
