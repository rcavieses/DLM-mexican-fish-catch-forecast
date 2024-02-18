# Import necessary modules from the ia_model package and other required libraries
from ia_model.data_preparation import load_data, prepare_data, split_data
from ia_model.model import  train_model, evaluate_model, save_model, build_dynamic_model
from ia_model.utils import ia_plot_history
import config as cfg
import joblib

""" Help:
 En este codigo estamos entrenando un modelo de deep learning con multiples capas,
 la información de las capas y las variables que se están usando se pueden ver en el
 archivo config. 

 In this code we are trainning a deep learning model who has multiples layers, the information
 about the layers and the variables are described in config file
"""

def main():
    # Load and prepare data
    df = load_data(str(cfg.FILE_PATH))
    X, y, n_features, scaler_x,scaler_y, encoder = prepare_data(df, cfg.FEATURE_COL, cfg.TARGET_COL, cfg.time_steps,cfg.CATEGORICAL_COL,cfg.NUMERIC_COL)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print('done')

    # Build model

    model = build_dynamic_model(cfg.time_steps,n_features,cfg.layer_type,cfg.neurons_per_layer)
    print('done')
    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test, cfg.EPOCHS, cfg.BATCH_SIZE)
    ia_plot_history(history)

    # Evaluate model
    mse, mae, r2 = evaluate_model(model, X_test, y_test)

    print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')

    save= input('save the model? y/n: ')
    if save == 'y':
        save_model(model,'mi_modelo.h5')
        joblib.dump(scaler_x, 'scaler_x.pkl')  # Saves the scaler
        joblib.dump(scaler_y, 'scaler_x.pkl')  # Saves the scaler
        joblib.dump(encoder, 'encoder_filename.pkl')  # Saves the encoder


if __name__ == "__main__":
    main()
