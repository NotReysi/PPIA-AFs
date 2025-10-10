import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def normalizar(ndf):
    print("\n****************NORMALIZANDO BASE DE DATOS*****************\n")
    print("...Generando Dummies...\n")
    #   Se utilizó la técnica de One-Hot Encoding para las columnas categóricas "city" y "area" y que se puedan normalizar sin problema
    #   Esto crea nuevas columnas binarias (0 y 1) para cada categoría en las columnas originales donde si es 1 significa que pertenece a esa categoría y 0 que no
    dumisc = pd.get_dummies(ndf['city']).astype(int)
    dumisa = pd.get_dummies(ndf['area']).astype(int)

    print("...Integrando Dummies...\n")
    #   Al utilizar One-Hot Encoding, se crean muchas columnas nuevas, por lo que es necesario agregarlas al data set original y eliminar las columnas originales
    #   Se agregan los Dummies en el data set
    newndf = ndf.join([dumisc, dumisa])

    #   Se eliminan las columnas originales de "city" y "area" porque ya no nos sirven
    newndf = newndf.drop(columns=['city','area'])

    print("...Normalizando datos numéricos...\n")
    #   Se normalizan todos los datos numéricos para que estén en un rango de 0 a 1
    scaler = MinMaxScaler()
    df_normalizado = pd.DataFrame(scaler.fit_transform(newndf), columns=newndf.columns)
    print("----------------BASE DE DATOS NORMALIZADA----------------")

    return df_normalizado










