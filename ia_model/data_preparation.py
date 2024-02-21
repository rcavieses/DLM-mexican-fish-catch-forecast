#Functions for preprossesing the data 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib

def load_data(file_path):
    """
    Load the data and transfor to datetime type the date column

    """
    df = pd.read_csv(file_path, low_memory=False, index_col=0)
    df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y-%m')
    df = df.sort_values(by=['NOMBRE PRINCIPAL', 'NOMBRE ESTADO', 'NOMBRE OFICINA', 'YearMonth'])
    return df

def prepare_data(df, target_col, time_steps, categorical_columns, numeric_columns):
    """
    Prepares data for LSTM model by encoding categorical data and scaling numerical data.

    :param df: DataFrame containing the dataset.
    :param feature_cols: List of column names to be used as features (excluding categorical and target columns).
    :param target_col: Name of the column to be used as the target variable.
    :param time_steps: Number of time steps to be used in each input sequence.
    :param categorical_columns: List of column names to be one-hot encoded.
    :param numeric_columns: List of numerical column names to be scaled (excluding the target column).
    
    :return: Tuple of (data_X, data_Y, scaler_x, scaler_y, encoder, n_features) where
        data_X is the feature data shaped into sequences of (samples, time_steps, features),
        data_Y is the target data,
        scaler_x is the scaler used for the feature data,
        scaler_y is the scaler used for the target data,
        encoder is the encoder used for categorical data,
        n_features is the number of features after encoding and scaling.
    """
    # One-hot encoding for categorical columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    # Min-Max scaling for numeric feature columns
    scaler_x = MinMaxScaler()
    scaled_data = scaler_x.fit_transform(df[numeric_columns])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)

    # Min-Max scaling for the target column
    scaler_y = MinMaxScaler()
    scaled_target = scaler_y.fit_transform(df[[target_col]])  # Ensure target is 2D for scaling
    df[target_col] = scaled_target  # Replace original with scaled

    # Combine encoded and scaled data
    prepared_data = pd.concat([encoded_df, scaled_df], axis=1)  # Don't include target here
    data_X, data_Y = [], []
    for i in range(time_steps, len(df)):
        data_X.append(prepared_data.iloc[i-time_steps:i].to_numpy())
        data_Y.append(scaled_target[i, 0])  # Directly use scaled target data
    
    n_features = prepared_data.shape[1]  # Number of features excluding target
    return np.array(data_X), np.array(data_Y), scaler_x, scaler_y, encoder, n_features


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits feature and target data into training and testing sets.

    :param X: Array-like, shape [n_samples, n_features]. Feature data.
    :param y: Array-like, shape [n_samples]. Target data.
    :param test_size: Float or int, default 0.2. If float, represents the proportion of the dataset to include in the test split.
    :param random_state: Int or RandomState instance, default 42. Controls the shuffling applied to the data before applying the split.

    :return: Tuple of (X_train, X_test, y_train, y_test)
        WHERE
        X_train is the training feature data,
        X_test is the testing feature data,
        y_train is the training target data,
        y_test is the testing target data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

