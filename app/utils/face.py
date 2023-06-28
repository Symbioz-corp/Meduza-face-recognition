import face_recognition
import numpy as np

def calculate_face_features(image_path):
    try:
        # Charger l'image et effectuer les calculs sur les caractéristiques du visage
        image = face_recognition.load_image_file(image_path)
        face_landmarks = face_recognition.face_landmarks(image)[0]
        face_features = {
            'left_eye': np.mean(face_landmarks['left_eye'], axis=0),
            'right_eye': np.mean(face_landmarks['right_eye'], axis=0),
            'nose_bridge': np.mean(face_landmarks['nose_bridge'], axis=0),
            'mouth': np.mean(face_landmarks['top_lip'] + face_landmarks['bottom_lip'], axis=0)
        }
        return face_features

    except Exception as e:
        print(f"Erreur lors du calcul des caractéristiques du visage : {e}")
        return None

def calculate_distance(features1, features2):
    # Calculer la distance entre les caractéristiques des visages
    distances = []
    for key in features1:
        distance = np.linalg.norm(features1[key] - features2[key])
        distances.append(distance)
    return np.mean(distances)
