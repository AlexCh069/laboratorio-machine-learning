from sqlmodel import Session, create_engine, SQLModel
from typing import Annotated
from fastapi import Depends, FastAPI 

sqlite = "db.sqlite3"
sqlite_url = f"sqlite:///{sqlite}"

# 1. Configuro la conexión con SQLite
engine = create_engine(sqlite_url)

# 2. Defino una función que me da una sesión de base de datos (abrir y cerrar automático)
def get_session():
    with Session(engine) as session:
        yield session

def create_all_tables(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield

# Creo un tipo de parámetro que puedo usar en endpoints para obtener la sesión automáticamente
SessionDep = Annotated[Session, Depends(get_session)]
