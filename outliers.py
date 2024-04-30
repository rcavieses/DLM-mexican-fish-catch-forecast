import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar los datos
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return pd.DataFrame()

# Función para graficar la serie de tiempo de una oficina
def plot_time_series(data, office_name, column='PESO DESEMBARCADO_KILOGRAMOS'):
    plt.figure(figsize=(10, 5))
    plt.plot(data['YearMonth'], data[column], marker='o')
    plt.title(f"Serie de Tiempo para {office_name}")
    plt.xlabel('Fecha')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# UI
st.title('Revisión de Outliers en Series de Tiempo por Oficina')

# Campo para ingresar el nombre del archivo
filename = st.text_input('Ingrese el nombre del archivo CSV:')
if st.button('Cargar Datos'):
    data = load_data(filename)
    if not data.empty:
        # Crear una lista única de oficinas
        offices = data['NOMBRE OFICINA'].unique()
        st.session_state['data'] = data
        st.session_state['offices'] = offices
        st.session_state['current_index'] = 0
        st.success('Datos cargados correctamente!')

# Funcionalidad para seleccionar y cambiar outliers
if 'data' in st.session_state and 'offices' in st.session_state:
    # Obtener la oficina actual
    current_office = st.session_state['offices'][st.session_state['current_index']]
    office_data = st.session_state['data'][st.session_state['data']['NOMBRE OFICINA'] == current_office]

    # Mostrar gráfico
    fig = plot_time_series(office_data, current_office)
    st.pyplot(fig)

    # Seleccionar y modificar un dato
    selected_index = st.selectbox('Seleccione el índice del dato a modificar:', office_data.index)
    new_value = st.number_input('Ingrese el nuevo valor:', value=float(office_data.loc[selected_index, 'PESO DESEMBARCADO_KILOGRAMOS']))
    if st.button('Actualizar Valor'):
        st.session_state['data'].loc[selected_index, 'PESO DESEMBARCADO_KILOGRAMOS'] = new_value
        st.success('Valor actualizado correctamente!')

    # Navegación entre oficinas
    if st.button('Siguiente Oficina'):
        if st.session_state['current_index'] < len(st.session_state['offices']) - 1:
            st.session_state['current_index'] += 1
        else:
            st.session_state['current_index'] = 0  # Volver al inicio si es el final de la lista
        st.experimental_rerun()

    if st.button('Guardar Archivo'):
        st.session_state['data'].to_csv(filename, index=False)
        st.success('Archivo guardado exitosamente!')
