
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le
