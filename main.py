import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_housing_data(csv_path):
    df = pd.read_csv(csv_path)
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[categorical_columns] = df[categorical_columns].replace({'yes': 1, 'no': 0})
    
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    np.random.seed(0)
    train, test = train_test_split(df, test_size=0.2)

    y_train = train.pop('price').values
    x_train = train.values

    y_test = test.pop('price').values
    x_test = test.values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler

class SimpleLinearRegression:
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        return X.dot(self.weights)

def train_model(x_train, y_train):
    model = SimpleLinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = np.mean((y_test - y_pred)**2)
    print(f"Mean Squared Error: {mse}")
    return mse

def save_model_and_scaler(model, scaler, model_filename, scaler_filename):
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

if __name__ == "__main__":
    csv_path = 'dataset/Housing.csv'
    x_train, y_train, x_test, y_test, scaler = preprocess_housing_data(csv_path)

    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)

    save_model_and_scaler(model, scaler, 'best_model.pkl', 'scaler.pkl')