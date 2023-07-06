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
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return {"message": "No face detected"}

    # Trouver le visage le plus grand
    largest_face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[3] - loc[1]))

    # Calculer l'encodage de visage de l'image de recherche
    image_encoding = face_recognition.face_encodings(image, [largest_face_location])[0]

    # Rechercher une correspondance de visage parmi les fichiers existants
    persons_path = Path("persons")
    best_match_distance = float('inf')
    best_match_data = None

    for person_folder in persons_path.iterdir():
        face_encodings = []
        for face_encoding_path in person_folder.glob("*.npy"):
            face_encodings.append(np.load(str(face_encoding_path)))

        face_distances = face_recognition.face_distance(face_encodings, image_encoding)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < best_match_distance:
            best_match_distance = face_distances[best_match_index]
            # Correspondance trouvée, récupérer les données de la personne
            data_path = person_folder / "data.json"
            with open(data_path) as f:
                best_match_data = json.load(f)
                
    print("best_match_distance: ", best_match_distance)
    
    if best_match_data is not None and best_match_distance < 0.6:
        return {"person": best_match_data}

    return {"message": "No match found"}

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

@app.get("/")
async def ping():
    print('Soeone pinged')
    return {"message": "Server working"}
