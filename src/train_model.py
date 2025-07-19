from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_and_preprocess_data

def train():
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data("data/processed/tree_data.csv")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    joblib.dump(model, "models/tree_species_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

if __name__ == "__main__":
    train()
