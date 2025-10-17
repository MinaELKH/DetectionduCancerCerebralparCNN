import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

labels_name = np.unique(['glioma', 'meningioma', 'notumor', 'pituitary'])
model = load_model('../models/best_model.h5', compile=False)

st.set_page_config(page_title="Prédiction Cérébrale", page_icon="🧠", layout="centered")

st.title("🧠 Détection de tumeur cérébrale")
st.write("Téléchargez une image IRM et laissez le modèle prédire la classe correspondante.")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.image(uploaded_file, caption=f"Classe prédite : {labels_name[predicted_class]} ({confidence*100:.2f}%)", use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(labels_name, prediction[0])
    ax.set_ylabel("Probabilité")
    ax.set_title("Distribution des prédictions")
    st.pyplot(fig)
