from fastapi import FastAPI 
from predict_model import Prediction, DataCreate
from db import SessionDep, create_all_tables
import pandas as pd 
import os
import pickle

app = FastAPI(lifespan = create_all_tables)
# -------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "churn", "models", "column_equivalence.pk")

with open(model_path, "rb") as f:
    column_equivalence = pickle.load(f)

model = pickle.load(open(r"C:\Users\Ares\Documents\LIBROS\PLATZI CURSOS\CIENCIA DE DATOS\MACHINE LEARNING\laboratorio-machine-learning\ml_experiments\406633100773434584\c526123d4919405bb861e7e1dd3f48b3\artifacts\model_1\model.pkl", "rb"))
# column_equivalence = pickle.load(open("churn/models/column_equivalence.pk", "rb"))

def convert_numerical(features):
    output = []
    for i, feat in enumerate(features):
        if i in column_equivalence:
            output.append(column_equivalence[i][feat])
        else:
            try:
                output.append(pd.to_numeric(feat))
            except:
                output.append(0)
    return output

@app.post('/predict') #, response_model = Prediction)
async def predic_prob(predict_data: DataCreate, session: SessionDep):

    """
    Este endpoint nos permite realizar la prediccion ingresando los datos necesarios en el request body.
    Los datos necesarios son los siguientes:
  
    - CreditScore[int]: Puntuación crediticia
    - Geography[int]: País de residencia {'France': 0, 'Germany': 1, 'Spain': 2}
    - Gender[int]: Género del cliente {'Female': 0, 'Male': 1}}
    - Age[int]: Edad en años.
    - Tenure[int]: Años que el cliente ha sido cliente
    - Balance[float]: Saldo bancario 
    - NumOfProducts[int]: Número de productos contratados 
    - HasCrCard[int]: Indica si tiene tarjeta de crédito
    - IsActiveMember[int]: Cliente activo 
    - EstimatedSalary[float]: Salario estimado 
    """

    X = list(predict_data.model_dump().values())

    # Crear una instancia del modelo ORM Prediction
    db_data = Prediction(**predict_data.dict())  # dict() es mejor aquí que model_dump()
    db_data.predict = int(model.predict([X]))
    db_data.predict_proba = model.predict_proba([X])[:,1]

    session.add(db_data)
    session.commit()
    session.refresh(db_data)

    response = {
        'probability': db_data.predict_proba,  # Convertir a lista para JSON
        'response': db_data.predict
    }
    return response


