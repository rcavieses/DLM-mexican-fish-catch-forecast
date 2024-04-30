from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
import numpy as np
import pandas as pd

# Same activation function for all layers except the last one( Dense - linear)
def evaluate_combinations(X_train, y_train, X_test, y_test, time_steps, n_features, neurons_list, layers_list, activations_list, epochs=50, batch_size=32):

    best_mse = np.inf
    best_r2 = -np.inf
    best_configuration = None

    # Iterate through each combination of layer counts, neurons, and single activation functions
    for layer_count in layers_list:
        for neurons in neurons_list:
            for activation in activations_list:  # Now iterating directly over activation functions
                model = Sequential()
                for i in range(layer_count):
                    # Decide if the current layer is the first or a subsequent layer and whether it returns sequences
                    return_sequences = i < layer_count - 1
                    if i == 0:  # First layer needs input shape
                        model.add(LSTM(neurons, activation=activation, input_shape=(time_steps, n_features), return_sequences=return_sequences))
                    else:  # Subsequent layers
                        model.add(LSTM(neurons, activation=activation, return_sequences=return_sequences))
                    model.add(Dropout(0.2))
                model.add(Dense(1, activation='linear'))
                model.compile(optimizer='adamax', loss='mean_squared_error')

                # Early stopping to avoid overfitting
                early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=0, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop], verbose=0)

                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                # Update best metrics and configuration if current model is better
                if mse < best_mse or r2 > best_r2:
                    best_mse = min(best_mse, mse)
                    best_r2 = max(best_r2, r2)
                    best_configuration = {'layers': layer_count, 'neurons': neurons, 'activation': activation, 'MSE': mse, 'R2': r2}

    return best_configuration, best_mse, best_r2


# Build model
def build_best_model(time_steps, n_features, best_configuration, epochs=50, batch_size=32):
    
    layer_count = best_configuration['layers']
    neurons = best_configuration['neurons']
    activation = best_configuration.get('activation', 'linear')  # Default to 'linear' if not specified
    
    model = Sequential()
    for i in range(layer_count):
        if i == 0:  # First layer needs input shape
            model.add(LSTM(neurons, activation=activation, input_shape=(time_steps, n_features), return_sequences=(layer_count > 1)))
        else:  # Subsequent layers
            model.add(LSTM(neurons, activation=activation, return_sequences=(i != layer_count - 1)))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adamax', loss='mean_squared_error')

    # EarlyStopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop], verbose=1)

    # Evaluate model on test data
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Test MSE: {mse}")
    print(f"Test R^2: {r2}")

    return model, mse, r2


# Can evaluate different activation function for each layer, but will take long time to run
# def evaluate_combinations(X_train, y_train, X_test, y_test,time_steps, n_features, neurons_list, layers_list, activations_list, epochs=50, batch_size=32):
#     from itertools import product
#     best_mse = np.inf
#     best_r2 = -np.inf
#     best_configuration = None

#     # Generate all combinations of layer counts, neurons, and activation functions
#     for layer_count in layers_list:
#         for neurons in neurons_list:
#             # Generate all possible combinations of activation functions for the given layer count
#             for activations in product(*activations_list[:layer_count]):
#                 model = Sequential()
#                 for i in range(layer_count):
#                     if i == 0:  # First layer needs input shape
#                         model.add(LSTM(neurons, activation=activations[i], input_shape=(time_steps, n_features), return_sequences=(layer_count > 1)))
#                     else:  # Subsequent layers
#                         model.add(LSTM(neurons, activation=activations[i], return_sequences=(i != layer_count - 1)))
#                     model.add(Dropout(0.2))
#                 model.add(Dense(1, activation = 'linear'))
#                 model.compile(optimizer='adamax', loss='mean_squared_error')

#                 early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=0, restore_best_weights=True)
#                 model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop], verbose=0)

#                 predictions = model.predict(X_test)
#                 mse = mean_squared_error(y_test, predictions)
#                 r2 = r2_score(y_test, predictions)

#                 if mse < best_mse:
#                     best_mse = mse
#                     best_r2 = r2
#                     best_configuration = {'layers': layer_count, 'neurons': neurons, 'activations': activations, 'MSE': mse}
#                 if r2 > best_r2:
#                     best_r2 = r2
#                     best_configuration = {'layers': layer_count, 'neurons': neurons, 'activations': activations, 'R2': r2}


#     return best_configuration, best_mse, best_r2