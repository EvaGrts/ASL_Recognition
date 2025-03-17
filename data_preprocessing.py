import os
import cv2
import mediapipe as mp
import numpy as np

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dossier contenant les images
DATA_DIR = "archive/asl_alphabet_train"

# Préparation des listes pour les données
X, y = [], []
labels_dict = {}  # Pour mapper les labels en index

# Parcourir les dossiers (chaque dossier = une classe)
for idx, label in enumerate(os.listdir(DATA_DIR)):
    labels_dict[label] = idx
    folder_path = os.path.join(DATA_DIR, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Détection des landmarks
        with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7) as hands:
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extraire les 21 landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    X.append(landmarks)
                    y.append(idx)

# Convertir en tableau numpy
X = np.array(X)
y = np.array(y)

# Sauvegarder les données
np.save("X_landmarks.npy", X)
np.save("y_labels.npy", y)
print("Recuperation des landmarks terminée")



