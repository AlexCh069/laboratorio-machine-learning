import os 
import pandas as pd 
import logging
from sklearn.model_selection import train_test_split 
from imblearn.combine import SMOTEENN


class ExtractionData:
    def __init__(self) -> None:
        # self.path = 'data/data_prep.csv'
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)   # Nombre del logger de acuerdo al script

    def read_csv(self,  path: str, file_name: str) -> pd.DataFrame:
        file_path = os.path.join(path, file_name)
        try:
            self.logger.info(f'Reading data from {file_path}')
            return pd.read_csv(file_path)
        except FileNotFoundError:
            self.logger.error(f'File {file_name} not found in {path}')
            return None
        
    def smoteenn_resample(self, data: pd.DataFrame,target:str = None) -> pd.DataFrame:

        """SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors) 
        es una técnica de preprocesamiento que combina sobremuestreo con SMOTE y submuestreo 
        con ENN para mejorar el balance de clases en conjuntos de datos desbalanceados.
        
        Parametros:
            - data[DataFrame]: Datos en formato dataframe para realizar el resampleo 
            - target[str]: Nombre de la columna que contiene la variable objetivo (opcional)
        
        Returns:
            - data_smoteenn[DataFrame]: Datos resampleados en formato dataframe

        Exceptions:
            - Exception: Error durante el resampleo con SMOTEENN

        """

        try:
            if target is None:
                X = data.drop(data.columns[-1], axis = 1)
                y = data[data.columns[-1]]

                smotee_nn = SMOTEENN(random_state=69)
                x_resampled, y_resampled = smotee_nn.fit_resample(X,y)

                data_smoteenn = pd.DataFrame(x_resampled, columns = X.columns)
                data_smoteenn['Exited'] = pd.Series(y_resampled)
                self.logger.info("SMOTEEN RESAMPLING")

                return data_smoteenn
            
            else:
                X = data.drop(target, axis = 1)
                y = data[target] 
                smotee_nn = SMOTEENN(random_state=69)
                x_resampled, y_resampled = smotee_nn.fit_resample(X,y)

                data_smoteenn = pd.DataFrame(x_resampled, columns = X.columns)
                data_smoteenn['Exited'] = pd.Series(y_resampled)
                self.logger.info("SMOTEEN RESAMPLING")
                return data_smoteenn
            
        except Exception as e:
            self.logger.error("Don't apply SMOTEENN")
            return None



    # def split_SMOTEENN(self, data:pd.DataFrame):

    #     try:
    #         self.logger.info('Apply SMOTEENN')
    #         X = data.drop(data.columns[-1], axis = 1)
    #         y = data[data.columns[-1]]

    #         smotee_nn = SMOTEENN(random_state=69)
    #         x_resampled, y_resampled = smotee_nn.fit_resample(X,y)

    #         x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, train_size = 0.3, random_state=69)
    #         return x_train, x_test, y_train, y_test

    #     except Exception as e:
    #         self.logger.error("Don't apply SMOTEENN")
    #         return None


    def split_data(self, data: pd.DataFrame):
        # Implementación para separar datos de entrenamiento y prueba
        try:
            self.logger.info('Data successfully split')
            X = data.drop(data.columns[-1], axis=1)
            y = data[data.columns[-1]]
            x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, random_state=69)
            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            self.logger.error(f'Error occurred while splitting data: {str(e)}')
            return None
        
"""
util = ExtractionData()

os.chdir('..') # Nos posicionamos en el directorio padre
data = util.read_csv('data','data_prep.csv')    # lectura del csv

x_train, x_test, y_train, y_test = util.split_SMOTEENN(data)    # separación de datos de entrenamiento y prueba

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)"""