import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    # Prepare test data
    test_data = {
        "area": "1000",
        "bedrooms": "2",
        "bathrooms": "1",
        "stories": "1",
        "mainroad": "on",
        "guestroom": "off",
        "basement": "off",
        "hotwaterheating": "off",
        "airconditioning": "on",
        "parking": "1",
        "prefarea": "off",
        "furnishingstatus": "furnished"
    }

    # Make a POST request
    response = client.post("/predict", data=test_data)

    # Check response
    assert response.status_code == 200
    assert "prediction" in response.json
    assert isinstance(response.json["prediction"], float)

def test_predict_endpoint_invalid_input(client):
    invalid_data = {
        "area": "1000",
        # 'bedrooms' is missing
        "bathrooms": "1",
        "stories": "1",
        "parking": "1",
        "furnishingstatus": "furnished"
    }

    # Make a POST request
    response = client.post("/predict", data=invalid_data)

    # Check response
    assert response.status_code == 400
    assert "error" in response.json
    assert "Missing required field: bedrooms" in response.json["error"]

def test_predict_endpoint_invalid_furnishingstatus(client):
    invalid_data = {
        "area": "1000",
        "bedrooms": "2",
        "bathrooms": "1",
        "stories": "1",
        "parking": "1",
        "furnishingstatus": "invalid"
    }

    # Make a POST request
    response = client.post("/predict", data=invalid_data)

    # Check response
    assert response.status_code == 400
    assert "error" in response.json
    assert "Invalid value for furnishingstatus" in response.json["error"]