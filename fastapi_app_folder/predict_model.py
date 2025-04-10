from pydantic import BaseModel
from sqlmodel import SQLModel, Field

class DataPredict(SQLModel):
    creditscore: int
    geograpy: int 
    gender: int
    age: int
    tenure: int
    balance: float
    num_of_products: int
    hascrcard: int
    isActiveMember: int 
    estimatedSalary: float

class Prediction(DataPredict, table = True):
    id: int | None = Field(default = None, primary_key = True)
    predict: int 
    predict_proba: float

class DataCreate(DataPredict):
    pass 