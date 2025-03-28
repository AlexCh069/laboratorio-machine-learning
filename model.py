from imblearn.combine import SMOTETomek
import pandas as pd
import logging
import mlflow
import numpy as np
from utils.utils import ExtractionData 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import xgboost as xgb

# print(os.getcwd())  # Visualizar nuestro posicionamiento en el directorio

# Definir la ruta de almacenamiento de los experimentos
tracking_uri = os.path.abspath("ml_experiments")  # Guardará los experimentos en una carpeta "ml_experiments" en el directorio actual
mlflow.set_tracking_uri(f"file:///{tracking_uri}")  # Asegura el formato correcto en Windows

# Configurar el experimento
mlflow.set_experiment("churn_ml")

util = ExtractionData()

data = util.read_csv('data','data_prep.csv')        # Importamos la data

data_resample = util.smoteenn_resample(data,'Exited')
x_smot_train, x_smot_test, y_smot_train, y_smot_test = util.split_data(data_resample) # Resampleo de data

# Iniciar y registrar un experimento
with mlflow.start_run(run_name="xgb_smoteenn"):

    # Model Training --------------------------------------------------------------------------------
    rf = xgb.XGBClassifier(use_label_encoder = False, eval_metric = 'logloss')

    param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],       # Número de árboles
    'learning_rate': np.arange(0.01, 0.051, 0.03), # Tasa de aprendizaje
    'max_depth': [3, 5, 7],            # Profundidad máxima de los árboles
    'subsample': [0.05, 0.8, 1.0],           # Fracción de datos usada por árbol
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0]     # Fracción de features usadas por árbol
    }



    grid_search = GridSearchCV(estimator=rf, 
                               param_grid=param_grid, 
                               cv=2, 
                               verbose=2, 
                               n_jobs=-1,
                               scoring='accuracy')
    
    grid_search.fit(x_smot_train, y_smot_train)
    # ----------------------------------------------------------------------------------------------
    
    # Resutls --------------------------------------------------------------------------------------
    print("Mejores parámetros:", grid_search.best_params_)
    mlflow.log_params(grid_search.best_params_)
    
    print("Mejor puntuación de validación cruzada:", grid_search.best_score_)
    mlflow.log_metric("acuracy_train",grid_search.best_score_)
    # ----------------------------------------------------------------------------------------------

    # Model tracking -------------------------------------------------------------------------------
    best_rf2 = grid_search.best_estimator_
    mlflow.sklearn.log_model(best_rf2, "model_1")
    # ----------------------------------------------------------------------------------------------

    # Predicciones en el conjunto de prueba
    y_pred = best_rf2.predict(x_smot_test)

    # Generar la matriz de confusión ---------------------------------------------------------------
    cm = confusion_matrix(y_smot_test, y_pred)
    # Convertir a un diccionario para almacenarla como un parámetro
    cm_notation = ['TN','FP','FN','TP']
    cm_dict = {
        cm_notation[0]: cm[0,0],
        cm_notation[1]: cm[0,1],
        cm_notation[2]: cm[1,0],
        cm_notation[3]: cm[1,1]
        }
    mlflow.log_params(cm_dict)
    # ----------------------------------------------------------------------------------------------

    # Exactitud, Precision, Sensibilidad y Exhaustividad -------------------------------------------
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_smot_test, y_pred)
    precision = precision_score(y_smot_test, y_pred)
    recall = recall_score(y_smot_test, y_pred)
    specificity = tn/(tn + fp)
    auc_roc = roc_auc_score(y_smot_test, best_rf2.predict_proba(x_smot_test)[:, 1])
    auc_pr = average_precision_score(y_smot_test, best_rf2.predict_proba(x_smot_test)[:,1])
    
    mlflow.log_metric("accuracy_test", round(accuracy,5))
    mlflow.log_metric('precision_test', round(precision,5))
    mlflow.log_metric('recall_test', round(recall,5))
    mlflow.log_metric('specificity_test', round(specificity,5))
    mlflow.log_metric('auc_roc_test', round(auc_roc,5))     # Que tan bien separa ambas clases
    mlflow.log_metric('auc_pr_test', round(auc_pr,5))       # Mejor adaptacion a la clase positiva

    # ---------------------------------------------------------------------------------------------

    print(f"default artifact location: '{mlflow.get_artifact_uri()}'")


    # correr la ui: mlflow ui --backend-store-uri "file://C:/Users/Ares/Documents/LIBROS/PLATZI CURSOS/CIENCIA DE DATOS/MACHINE LEARNING/laboratorio-machine-learning/ml_experiments"

