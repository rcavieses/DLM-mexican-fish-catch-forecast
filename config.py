# Configuration parameters for the model and data preparation
FILE_PATH = '/aggregated_data2.csv'
FEATURE_COL = 'PESO DESEMBARCADO_KILOGRAMOS'#Change this if you want to add input variables
TARGET_COL = 'PESO DESEMBARCADO_KILOGRAMOS'
N_PAST = 12
EPOCHS = 50
BATCH_SIZE = 30
MODEL_PATH = 'mi_modelo.h5'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_Y.pkl'
