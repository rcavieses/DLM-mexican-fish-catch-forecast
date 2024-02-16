from matplotlib import pyplot as plt
import joblib

def ia_plot_history(history):
    plt.figure(figsize=(20,10))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def load_scalers(scaler_X_path, scaler_Y_path):
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)
    return scaler_X, scaler_Y
