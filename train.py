import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def entrenar(df_normalizado):
    print("\n****************ENTRENANDO MODELO*****************\n")
    columnas = ['reply time','guest favourite','host since','host Certification','room_type','host total listings count','consumer','total reviewers number','accommodates','bathrooms','bedrooms','beds','listing number','host response rate','sales','NewYork','Toronto','sydney','North America']
    
    #   Se definen X y Y
    x = df_normalizado[columnas].values
    y = df_normalizado['price'].values

    #   Se dividen los datos en Training Set (60%), Validation Set (20%) y Testing Set (20%)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    print("...Dividiendo datos...\n")
    print("...Entrenando modelo...\n")
    
    print(f"Tamaño de Training Set: {len(x_train)}")
    print(f"Tamaño de Validation Set: {len(x_val)}")
    print(f"Tamaño de Testing Set: {len(x_test)}")

    print("\nX_train (primeras filas):\n", x_train[:3])
    print("y_train:", y_train)

    print("---------------- Modelo entrenado exitosamente ----------------")




