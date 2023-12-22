import cv2
import streamlit as st
import numpy as np

# Charger le classificateur en cascade pour la détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def hex_to_rgb(hex_color):
    #  convertir chaque paire hexadécimale en un entier
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def detect_faces(image, min_neighbors, scale_factor, rectangle_color):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les visages en utilisant le classificateur en cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Dessiner des rectangles autour des visages détectés avec la couleur choisie
    for (x, y, w, h) in faces:
        # Convertir la couleur hexadécimale en un triplet RGB
        color_rgb = hex_to_rgb(rectangle_color)
        cv2.rectangle(image, (x, y), (x + w, y + h), color_rgb, 2)

    # Afficher l'image avec les rectangles
    st.image(image, channels="BGR", use_column_width=True)

def app():
    st.title("Face Detection using Viola-Jones Algorithm")

    # Ajouter des instructions à l'interface
    st.write("Upload an image from your phone and adjust the parameters below to detect faces.")

    # Ajouter un uploader pour télécharger une image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convertir le fichier uploader en une image OpenCV
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)

        # Ajouter des sliders pour ajuster les paramètres
        min_neighbors = st.slider("Adjust minNeighbors:", min_value=1, max_value=10, value=5)
        scale_factor = st.slider("Adjust scaleFactor:", min_value=1.01, max_value=1.5, value=1.3)

        # Ajouter un color picker pour choisir la couleur des rectangles
        rectangle_color = st.color_picker("Choose Rectangle Color")

        # Ajouter un bouton pour détecter les visages dans l'image
        if st.button("Detect Faces in Photo"):
            detect_faces(image, min_neighbors, scale_factor, rectangle_color)

if __name__ == "__main__":
    app()
