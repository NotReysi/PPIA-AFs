import pandas as pd
import numpy as np

import gestion as gt
import normalizacion as nz
import train as tr

print("\n################ INICIANDO PROGRAMA ################\n")

ndf = gt.load_clean_data()    #   Función para cargar y limpiar la base de datos

df_normalizado = nz.normalizar(ndf)     #   Función para normalizar la base de datos

tr.entrenar(df_normalizado)        #   Función para entrenar el modelo

gt.exportar(df_normalizado)  #   Función para exportar el DataFrame normalizado a un archivo CSV

print("\n################ PROGRAMA TERMINADO ################\n")

