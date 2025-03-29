import pickle
import json
from flask import Flask, request
import pandas as pd

FEATURES = pickle.load(open("churn/models/features.pk", "rb"))

#model = pickle.load(open("churn/models/model.pk", "rb"))
model = pickle.load(open(r"C:\Users\Ares\Documents\LIBROS\PLATZI CURSOS\CIENCIA DE DATOS\MACHINE LEARNING\laboratorio-machine-learning\ml_experiments\406633100773434584\c526123d4919405bb861e7e1dd3f48b3\artifacts\model_1\model.pkl", "rb"))
column_equivalence = pickle.load(open("churn/models/column_equivalence.pk", "rb"))


# create the Flask app
app = Flask(__name__)

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

@app.route('/query')
def query_example():
    features = convert_numerical(request.args.get('feats').split(','))
    probabilities = model.predict_proba([features])[:, 1]  # Probabilidad de la clase positiva
    
    response = {
        'probability': probabilities.tolist(),  # Convertir a lista para JSON
        'response': [int(x) for x in model.predict([features])] #
    }
    return json.dumps(response)

if __name__ == '__main__':
    # run app in debug mode on port 3001
    app.run(debug=True, port=3001)
