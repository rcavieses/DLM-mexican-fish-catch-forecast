import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False, index_col=0)
    df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y-%m')
    df = df.sort_values(by=['NOMBRE PRINCIPAL', 'NOMBRE ESTADO', 'NOMBRE OFICINA', 'YearMonth'])
    return df

def prepare_data(df, feature_col, target_col, n_past):
    data_X, data_Y = [], []
    for name, group in df.groupby(['NOMBRE PRINCIPAL', 'NOMBRE ESTADO', 'NOMBRE OFICINA']):
        group_features = group[[feature_col]].values
        group_target = group[[target_col]].values
        for i in range(n_past, len(group_features)):
            data_X.append(group_features[i-n_past:i, 0])
            data_Y.append(group_target[i, 0])
    return np.array(data_X), np.array(data_Y)

def scale_data(X, y, save_scalers=False):
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_Y.fit_transform(y.reshape(-1,1))
    if save_scalers:
        joblib.dump(scaler_X, 'scaler_X.pkl')
        joblib.dump(scaler_Y, 'scaler_Y.pkl')
    return X, y, scaler_X, scaler_Y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
