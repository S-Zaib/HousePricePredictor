import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_housing_data(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # 1. Data Exploration
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

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

    return x_train, y_train, x_test, y_test

# # Test the function
# x_train, y_train, x_test, y_test = preprocess_housing_data('dataset/housing.csv')
# print(x_train.head())
# print(y_train.head())
# print(x_test.head())
# print(y_test.head())







