
# Script principal del proyecto: Ejecuta el pipeline ETL y ML

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Proceso ETL: Cargar y limpiar los datos
def load_and_process_data(filepath):
    # Cargar los datos
    data = pd.read_csv(filepath)
    
    # Seleccionar columnas relevantes
    data = data[['age', 'overall']]
    
    # Limpiar datos nulos y duplicados
    data = data.dropna()
    data = data.drop_duplicates()
    
    return data

# Proceso ML: Entrenar y evaluar el modelo
def split_features_target(data):
    X = data[['age']]
    y = data['overall']
    return X, y

def train_model(X, y):
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    # Visualización de la relación edad-media predicha
    sns.scatterplot(x=X_test['age'], y=y_test, label='Actual')
    sns.scatterplot(x=X_test['age'], y=y_pred, label='Predicted')
    plt.title('Actual vs Predicted: Relación Edad-Media')
    plt.legend()
    plt.show()

# Ruta al archivo de datos
data_filepath = 'data/raw/raw_data.csv'

# Ejecución del pipeline ETL y ML
if __name__ == "__main__":
    data = load_and_process_data(data_filepath)
    X, y = split_features_target(data)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
