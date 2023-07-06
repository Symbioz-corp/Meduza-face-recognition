from fastapi import FastAPI, UploadFile, File
from typing import List
import os
from pathlib import Path
import json
import face_recognition
import uuid
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.post("/search")
async def search_person(image: UploadFile = File(...)):
    # Charger l'image et détecter les visages
    image_data = image.file.read()
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)  # Convertir l'image en tableau NumPy
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return {"message": "No face detected"}

    # Trouver le visage le plus grand
    largest_face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[3] - loc[1]))

    # Rechercher une correspondance de visage parmi les fichiers existants
    persons_path = Path("persons")
    for person_folder in persons_path.iterdir():
        face_encodings = []
        for face_encoding_path in person_folder.glob("*.npy"):
            face_encodings.append(np.load(str(face_encoding_path)))

        face_encodings = np.array(face_encodings)  # Convertir la liste en tableau NumPy

        face_distances = face_recognition.face_distance(face_encodings, face_encodings)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.6:
            # Correspondance trouvée, renvoyer le fichier de données JSON de la personne
            data_path = person_folder / "data.json"
            with open(data_path) as f:
                data = json.load(f)
            return {"person": data}

    return {"message": "No match found"}


@app.post("/search")
async def search_person(image: UploadFile = File(...)):
    # Charger l'image et détecter les visages
    image_data = image.file.read()
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)  # Convertir l'image en tableau NumPy
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return {"message": "No face detected"}

    # Trouver le visage le plus grand
    largest_face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[3] - loc[1]))

    # Rechercher une correspondance de visage parmi les fichiers existants
    persons_path = Path("persons")
    for person_folder in persons_path.iterdir():
        face_encodings = []
        for face_encoding_path in person_folder.glob("*.npy"):
            face_encodings.append(np.load(str(face_encoding_path)))

        face_distances = face_recognition.face_distance(face_encodings, face_encodings)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.6:
            # Correspondance trouvée, renvoyer le fichier de données JSON de la personne
            data_path = person_folder / "data.json"
            with open(data_path) as f:
                data = json.load(f)
            return {"person": data}

    return {"message": "No match found"}


@app.get("/")
async def ping():
    print('Soeone pinged')
    return {"message": "Server working"}
