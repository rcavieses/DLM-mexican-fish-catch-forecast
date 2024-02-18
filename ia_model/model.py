from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

def build_lstm_model(time_steps,n1,n2,n3,n_features,dropout=0.2):
    """
    Used only for test
    """
    model = Sequential([
        LSTM(n1, activation='relu', input_shape=(time_steps, n_features), return_sequences=True),
        LSTM(n2, activation='relu', return_sequences=True),
        LSTM(n3, activation='relu', return_sequences=True),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adamax', loss='mean_squared_error')
    return model


def build_dynamic_model(time_steps, n_features, layer_types, neurons_per_layer, dropout=0.2):
    """
    Builds a dynamic neural network model based on specified parameters.

    :param time_steps: Integer. Number of time steps to consider in each input sequence.
    :param n_features: Integer. Number of input features per time step.
    :param layer_types: List of strings. Specifies the types of layers to include in the model (e.g., 'LSTM', 'Dense').
    :param neurons_per_layer: List of integers. Number of neurons in each layer specified in 'layer_types'.
    :param dropout: Float, default 0.2. Dropout rate to apply after each layer except the last one.

    :return: A Sequential model with the specified layers and configuration.

    Raises:
        ValueError: If 'layer_types' and 'neurons_per_layer' have different lengths.
    """

    if len(layer_types) != len(neurons_per_layer):
        raise ValueError("layer_types and neurons_per_layer must have the same length")
    model = Sequential()
    for i, (layer_type, neurons) in enumerate(zip(layer_types, neurons_per_layer)):
        if i == 0:  # Primera capa, necesita definir input_shape
            if layer_type == 'LSTM':
                model.add(LSTM(neurons, activation='relu', input_shape=(time_steps, n_features), return_sequences=True if i < len(layer_types) - 1 else False))
            elif layer_type == 'Dense':
                model.add(Dense(neurons, activation='relu', input_shape=(time_steps, n_features)))
        else:  # Capas subsiguientes
            if layer_type == 'LSTM':
                model.add(LSTM(neurons, activation='relu', return_sequences=True if i < len(layer_types) - 1 else False))
            elif layer_type == 'Dense':
                model.add(Dense(neurons, activation='relu'))

        # Añadir Dropout después de cada capa si no es la última capa
        if i < len(layer_types) - 1:
            model.add(Dropout(dropout))

    # Asegurarse de que la última capa siempre sea una Dense para la salida
    if layer_types[-1] != 'Dense':
        model.add(Dense(1))  # Asumiendo un problema de regresión. Cambiar según sea necesario.

    model.compile(optimizer='adamax', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=30):
    """
    Trains the neural network model with the provided data.

    :param model: The Sequential model to be trained.
    :param X_train: Array-like, shape (n_samples, time_steps, n_features). Training feature data.
    :param y_train: Array-like, shape (n_samples,). Training target data.
    :param X_test: Array-like, shape (n_test_samples, time_steps, n_features). Testing feature data.
    :param y_test: Array-like, shape (n_test_samples,). Testing target data.
    :param epochs: Integer, default 50. Number of epochs to train the model.
    :param batch_size: Integer, default 30. Size of batches of data to use for each training epoch.

    :return: Tuple (model, history) where 'model' is the trained model and 'history' contains information about the training process.
    """
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='min', restore_best_weights=True)
    #change patience to train as you want the model be care of overfitting
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stop])
    return model, history

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    predictions = model.predict(X_test)[:,0,:].reshape(-1, 1)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2

def save_model(model, file_name):
    model.save(file_name)

def load_trained_model(file_name):
    return load_model(file_name)

