import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from main import preprocess_housing_data, SimpleLinearRegression, train_model, evaluate_model

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'price': [100000, 200000, 150000, 250000, 180000],
        'area': [1000, 1500, 1200, 1800, 1300],
        'bedrooms': [2, 3, 2, 4, 3],
        'bathrooms': [1, 2, 2, 3, 2],
        'stories': [1, 2, 1, 2, 1],
        'mainroad': ['yes', 'no', 'yes', 'yes', 'no'],
        'guestroom': ['no', 'yes', 'no', 'yes', 'yes'],
        'basement': ['no', 'yes', 'no', 'yes', 'no'],
        'hotwaterheating': ['no', 'yes', 'no', 'no', 'yes'],
        'airconditioning': ['yes', 'no', 'yes', 'yes', 'yes'],
        'parking': [1, 2, 1, 2, 1],
        'prefarea': ['no', 'yes', 'no', 'yes', 'yes'],
        'furnishingstatus': ['unfurnished', 'semi-furnished', 'furnished', 'unfurnished', 'semi-furnished']
    })

def test_preprocess_housing_data(sample_data, tmp_path):
    # Save sample data to tmp file
    csv_path = tmp_path / "test_housing.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Run preprocessing
    x_train, y_train, x_test, y_test, scaler = preprocess_housing_data(csv_path)
    
    # Check outputs
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(scaler, StandardScaler)

def test_train_model(sample_data):
    # Prepare data
    X = sample_data.drop('price', axis=1)
    y = sample_data['price']
    
    # Train model
    model = train_model(X.values, y.values)
    
    # Check output
    assert isinstance(model, SimpleLinearRegression)
    assert hasattr(model, 'weights')
    assert isinstance(model.weights, np.ndarray)

def test_evaluate_model(sample_data):
    # Prepare data
    X = sample_data.drop('price', axis=1)
    y = sample_data['price']
    
    # Train model
    model = SimpleLinearRegression()
    model.fit(X.values, y.values)
    
    # Evaluate model
    mse = evaluate_model(model, X.values, y.values)
    
    # Check output
    assert isinstance(mse, float)
    assert mse >= 0

def test_simple_linear_regression():
    # Create a simple dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    # Create and train the model
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    # Check if the model can make predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    
    # Check if the model can be saved and loaded
    model.save('test_model.json')
    loaded_model = SimpleLinearRegression.load('test_model.json')
    
    # Check if the loaded model makes the same predictions
    loaded_predictions = loaded_model.predict(X)
    np.testing.assert_array_almost_equal(predictions, loaded_predictions)