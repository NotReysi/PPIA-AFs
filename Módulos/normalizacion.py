import pandas as pd

from sklearn.preprocessing import MinMaxScaler

#   Conversión de columnas categóricas a numéricas
""" Debido a que las columnas "sex", "smoker" y "region" son categóricas, se necesita utilzar el método de One-hot-encoding para generar columnas numericas que se puedan normalizar.
Esto se repite por cada una de las 3 columnas categóricas   """

def one_hot_encoding(datos):
    #   Utilizamos la función "get_dummies" de la librería "pandas" para generar las columnas de manera fácil.
    s_dummies = pd.get_dummies(datos['sex']).astype(int)
    sm_dummies = pd.get_dummies(datos['smoker']).astype(int)
    r_dummies = pd.get_dummies(datos['region']).astype(int)

    print("\n... Generando Dummies ...\n")

    #   Solo se tienen que eliminar las columnas pasadas e insertar las nuevas.
    ndatos = datos.drop(["sex"], axis=1)
    ndatos = ndatos.join(s_dummies)

    ndatos = ndatos.drop(["smoker"], axis=1)
    ndatos = ndatos.join(sm_dummies)

    ndatos = ndatos.drop(["region"], axis=1)
    ndatos = ndatos.join(r_dummies)

    print("... Insertando Dummies ...")

    return ndatos

#   Normalización de la Base de Datos

def normalizar(ndatos):
    #   Se utiliza la función MinMaxScaler, que se encarga de hacer todas las funciones y procesos que se necesitan para normalizar los datos y dejarlas en un rango entre 0 y 1
    scaler = MinMaxScaler()
    df_normalizado = pd.DataFrame(scaler.fit_transform(ndatos), columns=ndatos.columns)

    print("\n... Normalizando Base de Datos ...")

    return df_normalizado
