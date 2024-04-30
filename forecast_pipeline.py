from ia_model.data_preparation import load_data
from ia_model.forecasting import realizar_pronostico, graficar_pronostico, filtrar_datos, prepare_last_data
from ia_model.utils import load_scalers
import config as cfg
from tensorflow.keras.models import load_model
import numpy as np

def ejecutar_pronostico(ruta_datos, ruta_modelo, nombre_producto, nombre_estado, nombre_oficina, n_pasos):
    # Carga de los datos y configuraciones
    df = load_data(ruta_datos)
    scaler_x, scaler_y, encoder = load_scalers('scaler_X.pkl', 'scaler_Y.pkl', 'encoder_filename.pkl')
    model = load_model(ruta_modelo)

    # Preparación y filtrado de datos
    data = filtrar_datos(df, nombre_producto, nombre_estado, nombre_oficina)
    prepared_data = prepare_last_data(data, cfg.CATEGORICAL_COL, cfg.NUMERIC_COL, cfg.time_steps, encoder, scaler_x)

    # Realización del pronóstico
    pronosticos = realizar_pronostico(model, prepared_data, scaler_y, n_pasos)

    # Generación y visualización del gráfico
    graficar_pronostico(data, pronosticos, nombre_producto, nombre_oficina, nombre_estado, n_pasos)

# Esto permite que la función sea importada y ejecutada desde otros módulos