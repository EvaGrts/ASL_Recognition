import cv2
import os
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Initialisation des outils MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dossier contenant les images
DATA_DIR = "archive/asl_alphabet_train"

labels_dict = {}  # Pour mapper les labels en index

# Parcourir les dossiers (chaque dossier = une classe)
for idx, label in enumerate(os.listdir(DATA_DIR)):
    labels_dict[label] = idx

# Charger le modèle entraîné
model = tf.keras.models.load_model("landmarks_model.h5")

# Initialisation de la webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des mains
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Convertir en numpy array et faire une prédiction
                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks)
                class_id = np.argmax(prediction)
                label = list(labels_dict.keys())[class_id]

                # Afficher le label prédit
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Dessiner les landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            # Afficher 'nothing' si aucune main n'est détectée
            cv2.putText(frame, "nothing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Afficher l'image dans la fenêtre
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
