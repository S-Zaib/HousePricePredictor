import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Data Preprocessing
def preprocess_housing_data(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # 1. Data Cleaning
    # Drop rows with missing values
    df.dropna(inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert categorical columns into numeric
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[categorical_columns] = df[categorical_columns].replace({'yes': 1, 'no': 0})
    
    # One-hot encoding for 'furnishingstatus'
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    # Train-Test Split
    np.random.seed(0)
    train, test = train_test_split(df, test_size=0.2)

    # Splitting data into X (features) and y (target)
    y_train = train.pop('price')
    x_train = train

    y_test = test.pop('price')
    x_test = test

    # Apply standard scaling (normalization)
    scaler = StandardScaler()

    # Fit on train and transform
    x_train_scaled = scaler.fit_transform(x_train)

    # Transform the test set
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test

# Hyperparameter Tuning with GridSearchCV
def train_with_grid_search(x_train, y_train):
    # Initialize the Ridge regression model
    ridge = Ridge()

    # Define hyperparameters for tuning
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],   # Regularization strength
        'fit_intercept': [True, False],     # Whether to fit the intercept
        # Removed 'normalize' as it's not a valid parameter for Ridge
    }

    # Initialize GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)

    # Fit GridSearchCV
    grid_search.fit(x_train, y_train)

    # Best parameters from the grid search
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # The best model
    best_model = grid_search.best_estimator_

    return best_model

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

    # Train the model with hyperparameter tuning
    best_model = train_with_grid_search(x_train, y_train)

    # Evaluate the model
    evaluate_model(best_model, x_test, y_test)
    print(f"Variance of Test Set Prices: {np.var(y_test)}")
