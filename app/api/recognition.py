from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.models.person import Person
from app.utils.face import calculate_face_features, calculate_distance

router = APIRouter()

@router.post("/reconnaissance-faciale/")
async def recognize_face(file: UploadFile = File(...)):
    try:
        # Sauvegarder temporairement le fichier reçu
        temp_image_path = f"temp_{file.filename}"
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())

        # Effectuer la reconnaissance faciale
        recognized_person = recognize_person(temp_image_path)

        # Supprimer le fichier temporaire
        os.remove(temp_image_path)

        if recognized_person:
            return JSONResponse(content=recognized_person, status_code=200)
        else:
            return JSONResponse(content={"message": "Personne non reconnue."}, status_code=404)

    except Exception as e:
        print(f"Erreur lors de la reconnaissance faciale : {e}")
        return JSONResponse(content={"message": "Erreur lors de la reconnaissance faciale."}, status_code=500)


@router.post("/reconnaissance-faciale-multiple/")
async def recognize_multiple_faces(files: List[UploadFile] = File(...), data_file: UploadFile = File(...)):
    try:
        # Sauvegarder les fichiers reçus
        temp_image_paths = []
        for file in files:
            temp_image_path = f"temp_{file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(await file.read())
            temp_image_paths.append(temp_image_path)

        # Calculer les caractéristiques des images de référence
        reference_features = []
        for i, temp_image_path in enumerate(temp_image_paths):
            reference_features.append(calculate_face_features(temp_image_path))

        # Charger les informations de chaque personne à partir du fichier JSON
        with open(data_file.filename) as json_file:
            data = json.load(json_file)

        results = []
        for person in data:
            # Charger les images de référence de chaque personne
            known_features = []
            for i in range(1, 4):
                image_name = f"{person['nom']}_visage{i}.jpg"
                image_path = os.path.join(base_image_path, person['nom'], image_name)
                known_features.append(calculate_face_features(image_path))

            # Calculer la distance entre les caractéristiques des visages
            distances = []
            for i, reference_feature in enumerate(reference_features):
                distance = calculate_distance(known_features[i], reference_feature)
                distances.append(distance)
            average_distance = np.mean(distances)

            if average_distance < 0.5:  # Ajuster cette valeur seuil selon vos besoins
                results.append(person)

        # Supprimer les fichiers temporaires
        for temp_image_path in temp_image_paths:
            os.remove(temp_image_path)

        if results:
            return JSONResponse(content=results, status_code=200)
        else:
            return JSONResponse(content={"message": "Aucune correspondance trouvée."}, status_code=404)

    except Exception as e:
        print(f"Erreur lors de la reconnaissance faciale multiple : {e}")
        return JSONResponse(content={"message": "Erreur lors de la reconnaissance faciale multiple."}, status_code=500)

