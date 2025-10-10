import pandas as pd
import numpy as np

#   Función que sirve para cargar y limpiar los datos de la base de datos
def load_clean_data(file_path = "airbnb.csv"):

    print("Cargando la base de datos...\n")

    datos = pd.read_csv("airbnb.csv", low_memory=False)   #   Leer el archivo .csv
    
    print("...Limpiando la base de datos...\n")
    #   Toda la tabla utiliza "," en lugar de "."
    #   Además de que muchas columnas que deberían ser numéricas están como objetos (strings)

    #   La columna "price" Se tiene que cambiar primero a string para poder eliminar los caracteres "$" y "," y finalmente a float
    datos['price'] = datos['price'].astype(str)
    datos['price'] = datos['price'].str.replace('$', '')
    datos['price'] = datos['price'].str.replace(',', '')
    datos['price'] = datos['price'].astype(float)
    
    #   Las columnas "consumer", "bathrooms", "host response rate" y "host acceptance rate" solo necesitan cambiar "," por "." y luego a float

    datos['consumer'] = datos['consumer'].str.replace(',', '.')
    datos['consumer'] = datos['consumer'].astype(float)

    datos['bathrooms'] = datos['bathrooms'].str.replace(',', '.')
    datos['bathrooms'] = datos['bathrooms'].astype(float)

    datos['host response rate'] = datos['host response rate'].str.replace(',', '.')
    datos['host response rate'] = datos['host response rate'].astype(float)

    datos['host acceptance rate'] = datos['host acceptance rate'].str.replace(',', '.')
    datos['host acceptance rate'] = datos['host acceptance rate'].astype(float)

    ndf = datos.dropna()    #   Esta función Quita las filas que tienen valores nulos

    ndf = ndf.drop(columns=['id', 'name', 'host_id', 'host_name'])  #   Se eliminan las columnas que no son necesarias

    print("---------------- BASE DE DATOS LIMPIA ----------------")
    print(f"Número de filas originales: {len(datos)}")
    print(f"Número de filas DESPUÉS de eliminar nulos: {len(ndf)}")

    return ndf


def exportar(df_normalizado):
    print("\n**************** EXPORTANDO BASE DE DATOS NORMALIZADA *****************\n")
    df_normalizado.to_csv('airbnb_normalizado.csv', index=False)
    print("DataFrame guardado exitosamente como 'airbnb_normalizado.csv'")
