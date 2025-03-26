from imblearn.combine import SMOTETomek
import pandas as pd
import logging
import mlflow
from utils.utils import ExtractionData 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


# print(os.getcwd())  # Visualizar nuestro posicionamiento en el directorio

# Definir la ruta de almacenamiento de los experimentos
tracking_uri = os.path.abspath("ml_experiments")  # Guardará los experimentos en una carpeta "ml_experiments" en el directorio actual
mlflow.set_tracking_uri(f"file:///{tracking_uri}")  # Asegura el formato correcto en Windows

# Configurar el experimento
mlflow.set_experiment("churn_ml")

util = ExtractionData()

data = util.read_csv('data','data_prep.csv')        # Importamos la data
x_smot_train, x_smot_test, y_smot_train, y_smot_test = util.split_SMOTEENN(data) # Resampleo de data

# Iniciar y registrar un experimento
with mlflow.start_run(run_name="propotipe_1"):

    rf = RandomForestClassifier(bootstrap=True, random_state=69)

    param_grid = {
        'min_samples_leaf': [2,3,4]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, scoring='accuracy')
    grid_search.fit(x_smot_train, y_smot_train)

    print("Mejores parámetros:", grid_search.best_params_)
    mlflow.log_params(grid_search.best_params_)
    
    print("Mejor puntuación de validación cruzada:", grid_search.best_score_)
    mlflow.log_metric("acuracy_train",grid_search.best_score_)

    best_rf2 = grid_search.best_estimator_
    mlflow.sklearn.log_model(best_rf2, "model_1")

    # Predicciones en el conjunto de prueba
    y_pred = best_rf2.predict(x_smot_test)

    # Calcular precisión
    accuracy = accuracy_score(y_smot_test, y_pred)
    print("Precisión en el conjunto de prueba:", accuracy)
    mlflow.log_metric("accuracy_test", accuracy)

    print(f"default artifact location: '{mlflow.get_artifact_uri()}'")