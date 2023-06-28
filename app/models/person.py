from pydantic import BaseModel

class Person(BaseModel):
    nom: str
    prenom: str
    age: int
