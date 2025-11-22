import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

#   Evaluación del Modelo**
""" Después de entrenar el modelo, necesitamos evaluar su rendimiento. Para los modelos de regresión lineal, utilizamos métricas como el Error Cuadrático Medio (MSE), la Raíz del Error Cuadrático Medio (RMSE) y el Coeficiente de Determinación (R2 Score).  """

def evaluar(Y_test, Y_pred):
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)

    print("--- Evaluación del Modelo ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")