import joblib
import pandas as pd

def predict(input_features):
    model = joblib.load("models/tree_species_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    pred = model.predict([input_features])
    return le.inverse_transform(pred)[0]
