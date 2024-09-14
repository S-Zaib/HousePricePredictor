import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform
import joblib

# Data Preprocessing
def preprocess_housing_data(csv_path):
    df = pd.read_csv(csv_path)
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[categorical_columns] = df[categorical_columns].replace({'yes': 1, 'no': 0})
    
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    np.random.seed(0)
    train, test = train_test_split(df, test_size=0.2)

    y_train = train.pop('price')
    x_train = train

    y_test = test.pop('price')
    x_test = test

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler

# Hyperparameter Tuning with GridSearchCV
def train_with_grid_search(x_train, y_train):
    ridge = Ridge()
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
    }
    grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    print("Grid Search Best Hyperparameters:", best_params)
    return grid_search.best_estimator_

# Hyperparameter Tuning with RandomizedSearchCV
def train_with_random_search(x_train, y_train):
    ridge = Ridge()
    param_distributions = {
        'alpha': uniform(0.01, 100),
        'fit_intercept': [True, False],
    }
    random_search = RandomizedSearchCV(ridge, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=0)
    random_search.fit(x_train, y_train)
    best_params = random_search.best_params_
    print("Random Search Best Hyperparameters:", best_params)
    return random_search.best_estimator_


# Model Evaluation
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return mse

# Save the best model and the scaler
def save_model_and_scaler(model, scaler, model_filename, scaler_filename):
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

if __name__ == "__main__":
    csv_path = 'dataset/Housing.csv'
    x_train, y_train, x_test, y_test, scaler = preprocess_housing_data(csv_path)

    # Train models with different hyperparameter optimization methods
    print("Grid Search:")
    grid_search_model = train_with_grid_search(x_train, y_train)
    grid_search_mse = evaluate_model(grid_search_model, x_test, y_test)
    
    print("\nRandom Search:")
    random_search_model = train_with_random_search(x_train, y_train)
    random_search_mse = evaluate_model(random_search_model, x_test, y_test)

    # Save the best model and scaler
    best_model = grid_search_model if grid_search_mse < random_search_mse else random_search_model
    save_model_and_scaler(best_model, scaler, 'best_model.pkl', 'scaler.pkl')