from sklearn.model_selection import train_test_split

#   Entrenamiento del Modelo
""" Para este dataset se dividieron los datos en entrenamiento y prueba 70% - 30%. En este caso, lo que se busca predecir es el costo médico, por lo que la columna "charges" es la variable Y, así mismo, a la variable X le corresponde el resto del dataset. """

def train(df_normalizado):
    X = df_normalizado.drop(["charges"], axis = 1)
    Y = df_normalizado["charges"]

    #   División 70% - 30% para entrenar y prueba
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    print("\n... Entrenando Modelo ...")

    print(f"Tamaño de Training Set: {len(X_train)}")
    print(f"Tamaño de Validation Set: {len(X_val)}")
    print(f"Tamaño de Testing Set: {len(X_test)}")

    return X_train, X_test, Y_train, Y_test