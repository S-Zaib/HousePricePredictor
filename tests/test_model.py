import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from main import preprocess_housing_data, train_with_grid_search, evaluate_model

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'price': [100000, 200000, 150000, 250000, 180000],  # Increased to 5 samples
        'area': [1000, 1500, 1200, 1800, 1300],
        'bedrooms': [2, 3, 2, 4, 3],
        'bathrooms': [1, 2, 2, 3, 2],
        'stories': [1, 2, 1, 2, 1],
        'mainroad': [1, 0, 1, 1, 0],  # Changed to 1/0 instead of 'yes'/'no'
        'guestroom': [0, 1, 0, 1, 1],
        'basement': [0, 1, 0, 1, 0],
        'hotwaterheating': [0, 1, 0, 0, 1],
        'airconditioning': [1, 0, 1, 1, 1],
        'parking': [1, 2, 1, 2, 1],
        'prefarea': [0, 1, 0, 1, 1],
        'furnishingstatus': [0, 1, 2, 0, 1]  # 0: unfurnished, 1: semi-furnished, 2: furnished
    })

def test_preprocess_housing_data(sample_data, tmp_path):
    # Save sample data to tmp file
    csv_path = tmp_path / "test_housing.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Run preprocessing
    x_train, y_train, x_test, y_test, scaler = preprocess_housing_data(csv_path)
    
    # Check  outputs
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, pd.Series)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, pd.Series)
    assert isinstance(scaler, StandardScaler)

def test_train_with_grid_search(sample_data):
    # Prepare data
    X = sample_data.drop('price', axis=1)
    y = sample_data['price']
    
    # Train model
    model = train_with_grid_search(X, y)
    
    # Check output
    assert isinstance(model, Ridge)

def test_evaluate_model(sample_data):
    # Prepare data
    X = sample_data.drop('price', axis=1)
    y = sample_data['price']
    
    # Train model
    model = Ridge()
    model.fit(X, y)
    mse = evaluate_model(model, X, y)
    
    # Check output
    assert isinstance(mse, float)
    assert mse >= 0