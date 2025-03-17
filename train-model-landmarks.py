import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Charger les données
X = np.load("X_landmarks.npy")  
y = np.load("y_labels.npy")

# Afficher les labels uniques présents dans y pour comprendre d'où vient l'erreur
print(f"Valeurs uniques dans y avant filtration : {np.unique(y)}")

# Exclure les indices correspondants à 'nothing', 'space', et 'del' (indices 4, 15, 21)
invalid_indices = np.isin(y, [4, 15, 21])
valid_indices = ~invalid_indices  # Garder les indices qui ne sont pas dans invalid_indices

# Appliquer le filtre
X = X[valid_indices]  # Filtrer X pour ne garder que les exemples valides
y = y[valid_indices]  # Filtrer y pour ne garder que les labels valides

# Afficher les labels uniques après filtration
print(f"Valeurs uniques dans y après filtration : {np.unique(y)}")

# Encoder les labels en entiers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Sauvegarder l'encoder pour l'utiliser lors de la reconnaissance
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Normaliser les données
X = X / np.max(X)

# Diviser en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Construction du modèle
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(26, activation="softmax")  # Nombre de classes = 26 (A-Z)
])

# Compiler et entraîner le modèle
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Sauvegarder le modèle
model.save("landmarks_model.h5")
print("Modèle entraîné et sauvegardé")
