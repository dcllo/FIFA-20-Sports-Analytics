# Análisis de la Influencia de la Edad en la Media de los Jugadores de FIFA

## Resumen Ejecutivo

Este proyecto tiene como objetivo analizar la relación entre la edad de los jugadores de FIFA y su calificación general (overall rating). A través del uso de técnicas de ciencia de datos y machine learning, hemos implementado un pipeline que permite limpiar y procesar los datos, entrenar un modelo de regresión lineal, y evaluar el rendimiento del modelo al predecir la media de un jugador basado en su edad.

### Objetivos del Proyecto

1. **Exploración de Datos**: Cargar, limpiar y explorar un conjunto de datos de jugadores de FIFA.
2. **Análisis Estadístico**: Calcular la correlación entre la edad y la media de los jugadores.
3. **Modelado Predictivo**: Implementar un modelo de regresión lineal para predecir la media de un jugador basado en su edad.
4. **Evaluación del Modelo**: Medir la precisión del modelo usando métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R²).

### Estructura del Proyecto

El proyecto está organizado en las siguientes carpetas y archivos:

- **`/notebooks`**: Contiene el notebook `exploration_notebook.ipynb`, que sirve como borrador para explorar el conjunto de datos y probar el plan antes de la implementación formal final.
- **`/scripts`**:
  - `etl.py`: Script encargado del proceso ETL (Extracción, Transformación y Carga).
  - `ml_pipeline.py`: Script que contiene las funciones para entrenar y evaluar el modelo de regresión lineal.
- **`/data`**: Carpeta que contiene los archivos de datos utilizados en el proyecto, incluyendo los datos brutos y transformados.
- **`/docs`**: Incluye el presente archivo `README.md`.
  
### Uso del Proyecto

#### 1. Subir los Datos

- En el entorno de Google Colab, sube el archivo ZIP que contiene los datos de jugadores de FIFA.
  
#### 2. Ejecución del Código

- Cargar y procesar los datos mediante el script `etl.py`.
- Ejecutar el script `ml_pipeline.py` para entrenar y evaluar el modelo.

#### 3. Interpretación de Resultados

- El proyecto calcula la correlación entre la edad y la media de los jugadores, generando una visualización de esta relación.
- El modelo de regresión lineal entrenado permite predecir la media de un jugador en función de su edad y se evalúa su rendimiento a través de métricas clave.

### Requisitos del Proyecto

Para ejecutar este proyecto, se requiere:

- **Python 3.7+**
- Bibliotecas:
  - `pandas`
  - `scikit-learn`
  - `seaborn`
  - `matplotlib`

### Resultados

1. **Correlación**: Se encontró una correlación entre la edad de los jugadores y su media. Este valor proporciona una indicación de cuán fuerte es la relación entre estas variables.
2. **Modelo Predictivo**: El modelo de regresión lineal fue capaz de predecir la media de un jugador en función de su edad con un cierto nivel de precisión, que se mide a través de métricas como el MSE y R².

### Contribución

Este proyecto está destinado a aquellos interesados en explorar datos de FIFA y aplicar técnicas de machine learning para análisis predictivo.

