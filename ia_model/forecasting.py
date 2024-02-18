import pandas as pd
import numpy as np
from ia_model.utils import load_scalers
from ia_model.model import load_trained_model
import plotly.graph_objects as go

def filtrar_datos(df, producto, estado, oficina):
    return df[(df['NOMBRE PRINCIPAL'] == producto) &
              (df['NOMBRE ESTADO'] == estado) &
              (df['NOMBRE OFICINA'] == oficina)].sort_values(by='YearMonth')


# Function to prepare last 12 timesteps data
def prepare_last_data(df, categorical_columns, numeric_columns, time_steps, encoder, scaler_x):
    # Apply one-hot encoding and scaling to the last 'time_steps' records
    encoded_data = encoder.transform(df[categorical_columns][-time_steps:])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

    scaled_data = scaler_x.transform(df[numeric_columns][-time_steps:])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)

    prepared_data = pd.concat([encoded_df, scaled_df], axis=1).to_numpy()
    prepared_data = prepared_data.reshape(1, time_steps, -1)  # Reshape for LSTM input
    return prepared_data

# Function to make forecast using the model
def realizar_pronostico(model, prepared_data, scaler_y, n_forecast):
    forecasts_scaled = []
    current_data = prepared_data
    for _ in range(n_forecast):
        predicted_value_scaled = model.predict(current_data)
        forecasts_scaled.append(predicted_value_scaled.flatten())
        
        # Update current_data with the new prediction for the next step
        current_data = np.roll(current_data, -1, axis=1)
        current_data[:, -1, :] = predicted_value_scaled
        
    # Inverse transform to get original scale
    pronosticos = scaler_y.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1))
    return pronosticos.flatten()

def graficar_pronostico(serie_historica, pronosticos, producto, oficina, estado, inicio_fechas_futuras, n_forecast):
    """
    Plots historical series and forecasts for a specified product and location.

    :param serie_historica: Series, historical data to be plotted.
    :param pronosticos: Array-like, forecasted values to be plotted.
    :param producto: String, name of the product being forecasted.
    :param oficina: String, name of the office related to the forecast.
    :param estado: String, name of the state related to the forecast.
    :param inicio_fechas_futuras: Datetime, the start date for the forecasted data.
    :param n_forecast: Integer, number of steps (months) to forecast.

    :return: Plotly graph object showing the historical data and the forecasts.
    """
 
    # Asegurarse de que las fechas históricas estén en formato correcto
    fechas_historicas = pd.date_range(start=serie_historica.index.min(), periods=len(serie_historica), freq='M')
    
    # Corregir: Asegura que las fechas de pronóstico inician correctamente después del último mes de datos históricos
    # Generar fechas futuras comenzando después del último registro de la serie histórica
    fechas_futuras = pd.date_range(start=inicio_fechas_futuras, periods=n_forecast, freq='M')

    fig = go.Figure()
    # Añade los datos históricos
    fig.add_trace(go.Scatter(x=fechas_historicas, y=serie_historica, mode='lines+markers', name='Datos históricos'))
    # Añade los datos de pronóstico
    fig.add_trace(go.Scatter(x=fechas_futuras, y=pronosticos, mode='lines+markers', name='Pronóstico'))
    
    # Configura el resto del gráfico
    fig.update_layout(title=f'Pronóstico de {n_forecast} meses para {producto} en {oficina}, {estado}',
                      xaxis_title='Fecha', yaxis_title='Peso Desembarcado (Kilogramos)',
                      xaxis_rangeslider_visible=True)
    fig.show()



