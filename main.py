import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data Preprocessing
def preprocess_housing_data(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # # 1. Data Exploration
    # print(df.info())
    # print(df.describe())
    # print(df.isnull().sum())

    # 2. Data Cleaning
    # Drop rows with missing values
    df.dropna(inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert categorical columns into numeric
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[categorical_columns] = df[categorical_columns].replace({'yes': 1, 'no': 0})
    
    # One-hot encoding for 'furnishingstatus',  (this prevents multicollinearity apparently) 
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    # 3. Train-Test Split
    np.random.seed(0)
    train, test = train_test_split(df, test_size=0.2)

    # 4. Splitting data into X (features) and y (target)
    y_train = train.pop('price')
    x_train = train

    y_test = test.pop('price')
    x_test = test

    #5. Apply standard scaling (normalization)
    scaler = StandardScaler()

    # Fit on train and transform
    x_train_scaled = scaler.fit_transform(x_train)

    # Transform the test set
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test

# Model Training
def train_linear_regression_model(x_train, y_train):
    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(x_train, y_train)

    return model

# Model Evaluation
def evaluate_model(model, x_test, y_test):
    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate the model with Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return mse

if __name__ == "__main__":
    # Preprocess the data
    csv_path = 'dataset/Housing.csv'
    x_train, y_train, x_test, y_test = preprocess_housing_data(csv_path)

    # Train the model
    model = train_linear_regression_model(x_train, y_train)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)
    print(f"Variance of Test Set Prices: {np.var(y_test)}")





