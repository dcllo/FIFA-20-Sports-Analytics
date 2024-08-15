# Crear los módulos de apoyo para ETL y ML con funciones separadas

# Módulo ETL (extraer, transformar y cargar)
etl_content = """
import pandas as pd

def load_and_process_data(filepath):
    # Cargar los datos
    data = pd.read_csv(filepath)
    
    # Seleccionar columnas relevantes
    data = data[['age', 'overall']]
    
    # Limpiar datos nulos y duplicados
    data = data.dropna()
    data = data.drop_duplicates()
    
    return data
"""

# Módulo ML (split, entrenar, evaluar)
ml_content = """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

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
"""

# Guardar los archivos etl.py y ml.py
etl_filepath = "/mnt/data/etl.py"
ml_filepath = "/mnt/data/ml.py"

with open(etl_filepath, "w") as etl_file:
    etl_file.write(etl_content)

with open(ml_filepath, "w") as ml_file:
    ml_file.write(ml_content)

(etl_filepath, ml_filepath)
