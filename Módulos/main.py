import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import normalizacion as nz
import train as tr
import regresionlineal as rl
import evaluacion as ev

#   Se lee el archivo .csv
datos = pd.read_csv('dataset.csv')

print("... Cargando Base de Datos ...")

ndatos = nz.one_hot_encoding(datos)
df_normalizado = nz.normalizar(ndatos)

#   Tabla de Correlación
"""Se utiliza la función "heatmap" de "seaborn" para esto, donde se puede ver de manera gráfica con colores, las variables que más influyen en los costos médicos.

En este caso nos podemos dar cuenta de que las variables que están directamente relacionadas con los costos son la edad, indice de masa corporal y que fumen.

Se puede entender que estos valores sean los más relacionados porque si estos aumentan, eres más propenso a tener complicaciones en la salud."""

sb.heatmap(df_normalizado.corr(), annot=True, cmap='YlGnBu')

X_train, X_test, Y_train, Y_test = tr.train(df_normalizado)

modelo = rl.linearregresion(X_train,Y_train)

#   Gráfica de Regresión Lineal
"""Se utiliza la librería "matplotlib" para generar una gráfica que compara el costo predicho con el real. Con esta gráfica podemos visualizar la regresión lineal."""

Y_pred = modelo.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Valor Predicho')
plt.title('Regresión Lineal')
plt.grid(True)
plt.show()

ev.evaluar(Y_test, Y_pred)