import pandas as pd
import numpy as np
from ia_model.utils import load_scalers
from ia_model.model import load_trained_model
import plotly.graph_objects as go

def filtrar_datos(df, producto, estado, oficina):
    return df[(df['NOMBRE PRINCIPAL'] == producto) &
              (df['NOMBRE ESTADO'] == estado) &
              (df['NOMBRE OFICINA'] == oficina)].sort_values(by='YearMonth')

def realizar_pronostico(model, data, scaler_X, scaler_Y, n_pasos):
    pronosticos = []
    current_data = data.copy()
    for _ in range(n_pasos):
        # Escalar los datos para el modelo
        data_scaled = scaler_X.transform(current_data.reshape(1, -1)).reshape((1, -1, 1))
        predicted_value_scaled = model.predict(data_scaled)
        predicted_value = scaler_Y.inverse_transform(predicted_value_scaled).flatten()[0]
        
        # Añadir el valor pronosticado a los datos actuales para futuras predicciones
        current_data = np.roll(current_data, -1)
        current_data[-1] = predicted_value
        
        # Guardar el pronóstico
        pronosticos.append(predicted_value)
    return np.array(pronosticos)

def graficar_pronostico(serie_historica, pronosticos, producto, oficina, estado, inicio_fechas_futuras, n_pasos):
    # Asegurarse de que las fechas históricas estén en formato correcto
    fechas_historicas = pd.date_range(start=serie_historica.index.min(), periods=len(serie_historica), freq='M')
    
    # Corregir: Asegura que las fechas de pronóstico inician correctamente después del último mes de datos históricos
    # Generar fechas futuras comenzando después del último registro de la serie histórica
    fechas_futuras = pd.date_range(start=inicio_fechas_futuras, periods=n_pasos, freq='M')

    fig = go.Figure()
    # Añade los datos históricos
    fig.add_trace(go.Scatter(x=fechas_historicas, y=serie_historica, mode='lines+markers', name='Datos históricos'))
    # Añade los datos de pronóstico
    fig.add_trace(go.Scatter(x=fechas_futuras, y=pronosticos, mode='lines+markers', name='Pronóstico'))
    
    # Configura el resto del gráfico
    fig.update_layout(title=f'Pronóstico de {n_pasos} meses para {producto} en {oficina}, {estado}',
                      xaxis_title='Fecha', yaxis_title='Peso Desembarcado (Kilogramos)',
                      xaxis_rangeslider_visible=True)
    fig.show()


def pronosticar_y_graficar(df, producto, estado, oficina, n_pasos, model_path, scaler_X_path, scaler_Y_path):
    # Filtrar datos
    filtered_df = filtrar_datos(df, producto, estado, oficina)
    last_n_months = filtered_df[-12:]['PESO DESEMBARCADO_KILOGRAMOS'].values  # Asumiendo que n_past=12
    
    # Configurar la serie histórica con 'YearMonth' como índice
    # Asegurarse de que 'YearMonth' esté en el formato correcto y establecido como índice.
    serie_historica = filtered_df.set_index('YearMonth')['PESO DESEMBARCADO_KILOGRAMOS']

    # Cargar el modelo y los escaladores
    model = load_trained_model(model_path)
    scaler_X, scaler_Y = load_scalers(scaler_X_path, scaler_Y_path)
    
    # Realizar pronóstico
    pronosticos = realizar_pronostico(model, last_n_months, scaler_X, scaler_Y, n_pasos)
    
    # Ajustar la fecha de inicio para las predicciones para asegurar la continuidad
    ultimo_mes = filtered_df['YearMonth'].max()
    # Asegurar que la fecha de inicio de las predicciones sea el mes siguiente al último mes de la serie histórica.
    inicio_fechas_futuras = ultimo_mes + pd.DateOffset(months=-1)
    
    # Graficar resultados
    graficar_pronostico(serie_historica, pronosticos, producto, oficina, estado, inicio_fechas_futuras, n_pasos)

