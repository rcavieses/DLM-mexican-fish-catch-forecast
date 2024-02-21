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
        # Tomamos el primer paso de tiempo de la secuencia pronosticada
        forecasts_scaled.append(predicted_value_scaled[:, 0, :].flatten())

        # Actualizamos current_data con la nueva predicción para el siguiente paso
        current_data = np.roll(current_data, -1, axis=1)
        current_data[:, -1, :] = predicted_value_scaled[:, 0, :].flatten()  # Asegúrate de adaptar la forma correctamente

    # Transformación inversa para obtener la escala original
    pronosticos = scaler_y.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1))
    return pronosticos.flatten()

import pandas as pd
import plotly.graph_objects as go

def graficar_pronostico(serie_historica, pronosticos, producto, oficina, estado, n_forecast):
    # Extraer el último valor real de la serie histórica
    ultimo_dato_real = serie_historica['PESO DESEMBARCADO_KILOGRAMOS'].iloc[-1]
    
    # Añadir el último dato real al inicio de la serie de pronósticos
    pronosticos_con_real = np.insert(pronosticos, 0, ultimo_dato_real)

    # Fechas para la serie histórica y las predicciones
    fechas_historicas = pd.to_datetime(serie_historica['YearMonth'], format='%Y%m')
    inicio_fechas_futuras = fechas_historicas.iloc[-1] + pd.DateOffset(months=-1)
    fechas_futuras = pd.date_range(start=inicio_fechas_futuras, periods=n_forecast + 1, freq='M')  # +1 para incluir el último dato real

    # Crear la figura y añadir las series de datos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas_historicas, y=serie_historica['PESO DESEMBARCADO_KILOGRAMOS'], mode='lines+markers', name='Datos históricos'))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=pronosticos_con_real, mode='lines+markers', name='Pronóstico'))

    # Configurar y mostrar el gráfico
    fig.update_layout(title=f'Pronóstico de {n_forecast} meses para {producto} en {oficina}, {estado}',
                      xaxis_title='Fecha', yaxis_title='Peso Desembarcado (Kilogramos)',
                      xaxis_rangeslider_visible=True)
    fig.show()




