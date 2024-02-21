# Configuration parameters for the model and data preparation
FILE_PATH = 'aggregated_data3.csv' #This data can be download from https://rcavieses.pythonanywhere.com/
FEATURE_COL = ['NOMBRE PRINCIPAL','NOMBRE ESTADO','NOMBRE OFICINA','SST','PESO DESEMBARCADO_KILOGRAMOS']#Change this if you want to add input variables
TARGET_COL = 'PESO DESEMBARCADO_KILOGRAMOS'
CATEGORICAL_COL = ['NOMBRE PRINCIPAL','NOMBRE ESTADO','NOMBRE OFICINA']
NUMERIC_COL = ['PESO DESEMBARCADO_KILOGRAMOS','SST']
steps_forecast = 12
time_steps= 12 # number of steps that lstm forecast, change this will be errors on scoore metrics ;)
layer_type =  ['LSTM','LSTM','Dense'] # add as many layer you need options: Dense or LSTM
neurons_per_layer = [10,15,1] #match the number of layers and length of neurons number list
EPOCHS = 50
BATCH_SIZE = 30
MODEL_PATH = 'mi_modelo.h5'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_Y.pkl'


