import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


#   Regresión Linear
def linearregresion(X_train,Y_train):
    #   Se llama a la función "LinearRegression" de "scikit learn" para crear una instancia de regresión lineal
    modelo = LinearRegression()

    modelo.fit(X_train,Y_train) #   Se especifica al modelo que tiene que utilizar las variables de entrenamiento
    print("\nModelo entrenado exitosamente :D\n")

    return modelo