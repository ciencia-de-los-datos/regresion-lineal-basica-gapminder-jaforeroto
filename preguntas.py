"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
from tkinter import Y
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv", sep= ",", thousands=None, decimal= ".") 

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = df['life'].array
    X = df['fertility'].array

    # Imprima las dimensiones de `y`
    print(y.shape)

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.reshape(-1,1)

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.reshape(-1,1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresión de algunas estadísticas básicas
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv", sep= ",", thousands=None, decimal= ".")

    # Imprima las dimensiones del DataFrame
    print(df.shape)

    # Imprima la correlación entre las columnas `life` y `fertility` con 4 decimales.
    print("{:4f}".format(df['life'].corr(df['fertility'])))

    # Imprima la media de la columna `life` con 4 decimales.
    print("{:4f}".format(df['life'].mean()))

    # Imprima el tipo de dato de la columna `fertility`.
    print(type(df['fertility']))

    # Imprima la correlación entre las columnas `GDP` y `life` con 4 decimales.
    print("{:4f}".format(df['GDP'].corr(df['life'])))


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv", sep= ",", thousands=None, decimal= ".") 

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df['fertility']

    # Asigne a la variable los valores de la columna `life`
    y_life = df['life']

    # Importe LinearRegression
    from ____ import ____

    # Cree una instancia del modelo de regresión lineal
    reg = ____

    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = ____(
        ____,
        ____,
    ).reshape(____, _____)

    # Entrene el modelo usando X_fertility y y_life
    reg.fit(____, ____)

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    print(____.score(____, ____).round(____))


def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from ____ import ____

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = ____

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = ____

    # Asigne a la variable los valores de la columna `life`
    y_life = ____

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = ____(
        ____,
        ____,
        test_size=____,
        random_state=____,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = ____

    # Entrene el clasificador usando X_train y y_train
    ____.fit(____, ____)

    # Pronostique y_test usando X_test
    y_pred = ____

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(____(____, ____))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
