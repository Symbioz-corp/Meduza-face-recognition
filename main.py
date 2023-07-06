from fastapi import FastAPI, UploadFile, File
from typing import List
import os
from pathlib import Path
import json
import face_recognition
import uuid
import np from numpy
app = FastAPI()

# Endpoint pour ajouter une personne
@app.post("/add")
async def add_person(images: List[UploadFile] = File(...), data: UploadFile = File(...)):
    # Créer un dossier unique pour chaque personne
    person_folder = str(uuid.uuid4())
    person_path = Path("persons") / person_folder
    person_path.mkdir(parents=True, exist_ok=True)

    # Enregistrer les images dans le dossier de la personne
    image_paths = []
    for image in images:
        image_path = person_path / image.filename
        image_paths.append(str(image_path))
        with open(image_path, "wb") as f:
            f.write(image.file.read())

    # Enregistrer le fichier JSON de données dans le dossier de la personne
    data_path = person_path / data.filename
    with open(data_path, "wb") as f:
        f.write(data.file.read())

    # Calculer les fichiers de reconnaissance faciale pour chaque image
    face_encodings = []
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        face_encodings.append(face_encoding)

    # Sauvegarder les fichiers de reconnaissance faciale dans le dossier de la personne
    for i, face_encoding in enumerate(face_encodings):
        face_encoding_path = person_path / f"face_{i}.npy"
        np.save(str(face_encoding_path), face_encoding)

    return {"message": "Person added successfully"}
