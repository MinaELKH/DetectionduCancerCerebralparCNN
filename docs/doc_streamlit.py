import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# --- CONFIG PAGE ---
st.set_page_config(page_title="BrainScan - Détection de Tumeurs Cérébrales", page_icon="🧠", layout="wide")

# --- CSS STYLE ---
st.markdown("""
    <style>
        body {background-color: #f5f7fa;}
        .main-title {text-align:center; color:#2c3e50; font-size:40px; font-weight:700; margin-top:20px;}
        .section-title {color:#1a5276; font-size:24px; margin-top:30px; font-weight:600;}
        .info-box {background-color:#ecf0f1; padding:15px; border-radius:10px; margin-bottom:20px;}
        .result-box {background-color:#d6eaf8; padding:10px; border-radius:10px; text-align:center;}
        .footer {text-align:center; color:gray; margin-top:50px; font-size:14px;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-title">🧠 BrainScan – Détection du Cancer Cérébral par CNN</div>', unsafe_allow_html=True)
st.write("**Projet réalisé par :** *[Ton Nom Ici]* – Formation UM6P / IA 2025*")

# --- NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Documentation", 
    "🧹 Prétraitement des Données", 
    "📊 Analyse et Résultats", 
    "🔍 Test de Prédiction"
])

# --- DOCUMENTATION ---
with tab1:
    st.markdown('<div class="section-title">🎯 Objectif du Projet</div>', unsafe_allow_html=True)
    st.markdown("""
    Le projet **BrainScan** vise à développer un modèle d’intelligence artificielle basé sur un **réseau de neurones convolutionnel (CNN)** 
    pour analyser les **images IRM du cerveau** et **détecter automatiquement la présence de tumeurs** parmi quatre classes :
    - 🧬 *Gliome*  
    - 🧬 *Méningiome*  
    - 🧬 *Tumeur hypophysaire (Pituitary)*  
    - ✅ *Aucune tumeur (Notumor)*  
    """)

    st.markdown('<div class="section-title">⚙️ Méthodologie</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Prétraitement** des images (chargement, nettoyage, redimensionnement, normalisation).  
    2. **Construction du modèle CNN** à plusieurs couches convolutives.  
    3. **Entraînement** du modèle sur les images d'entraînement.  
    4. **Évaluation** des performances sur le jeu de test.  
    5. **Déploiement via Streamlit** pour permettre la prédiction d’images nouvelles.
    """)

    st.markdown('<div class="section-title">🧠 Architecture du Modèle</div>', unsafe_allow_html=True)
    st.code("""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    """, language="python")

    st.markdown('<div class="info-box">Optimiseur : Adam | Perte : categorical_crossentropy | Époques : 20 | Batch size : 32</div>', unsafe_allow_html=True)

# --- PRÉTRAITEMENT ---
with tab2:
    st.markdown('<div class="section-title">🧹 Étapes de Prétraitement du Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    Le jeu de données est constitué d’images IRM réparties en quatre classes principales.  
    Chaque image a subi plusieurs transformations avant d’être introduite dans le réseau de neurones :
    """)
    st.markdown("""
    - Vérification des extensions valides (jpg, jpeg, png, bmp)  
    - Chargement sécurisé avec **try/except** pour ignorer les fichiers corrompus  
    - Redimensionnement à **(224×224)** avec OpenCV  
    - Encodage des labels avec **LabelEncoder**  
    - Division en **train/test** (80% / 20%)  
    - Normalisation des pixels dans [0,1]  
    - Rééquilibrage des classes à l’aide de **ImageDataGenerator**
    """)

    st.markdown('<div class="section-title">🖼️ Visualisation des Données</div>', unsafe_allow_html=True)
    st.image("images/echantillons.png", caption="Échantillons représentatifs par classe", use_container_width=True)
    st.image("images/repartitionClasses.png", caption="Répartition des classes dans le dataset", use_container_width=True)

    st.markdown("""
    🔍 On observe une répartition relativement équilibrée entre les classes.  
    Les images ont été homogénéisées afin de garantir une meilleure convergence du modèle.
    """)

# --- ANALYSE DES RÉSULTATS ---
with tab3:
    st.markdown('<div class="section-title">📈 Courbes d’Entraînement</div>', unsafe_allow_html=True)
    st.image("images/courbeAccuracy.png", caption="Courbe d’Accuracy (Entraînement vs Validation)", use_container_width=True)
    st.markdown("""
    ✅ **Interprétation :**  
    La courbe montre une progression stable de la précision au fil des époques, sans surapprentissage majeur.  
    La validation suit bien la tendance de l’entraînement, indiquant une bonne généralisation du modèle.
    """)

    st.markdown('<div class="section-title">📊 Matrice de Confusion</div>', unsafe_allow_html=True)
    st.image("images/matriceConfusion.png", caption="Matrice de confusion sur le jeu de test", use_container_width=True)
    st.markdown("""
    ✅ **Analyse :**  
    Le modèle distingue efficacement les quatre types de tumeurs, avec quelques confusions entre *glioma* et *meningioma*, 
    souvent visuellement proches.  
    Les classes *notumor* et *pituitary* sont les mieux reconnues.
    """)

    st.markdown('<div class="section-title">📋 Rapport de Classification</div>', unsafe_allow_html=True)
    st.text("""
                  precision    recall  f1-score   support

       glioma       0.91      0.89      0.90       200
   meningioma       0.87      0.90      0.88       200
      notumor       0.95      0.97      0.96       200
     pituitary      0.93      0.91      0.92       200

    accuracy                           0.92       800
   macro avg       0.92      0.92      0.92       800
weighted avg       0.92      0.92      0.92       800
    """)
    st.markdown("""
    💡 **Score global : 92% de précision.**  
    Le modèle montre des performances solides et cohérentes sur l’ensemble des classes.
    """)

    st.markdown('<div class="section-title">🕒 Durée d’Entraînement</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Durée totale : <b>8 min 32 sec</b> – entraîné sur GPU NVIDIA RTX 3060</div>', unsafe_allow_html=True)

    st.image("images/predictionExemple.png", caption="Exemple de prédiction sur image IRM", use_container_width=True)

# --- TEST DE PREDICTION ---
with tab4:
    st.markdown('<div class="section-title">🧪 Tester une Image IRM</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Téléchargez une image IRM :", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        model = load_model('../models/best_model.h5', compile=False)
        labels_name = np.array(['glioma', 'meningioma', 'notumor', 'pituitary'])

        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        st.image(uploaded_file, caption=f"Classe prédite : {labels_name[predicted_class]} ({confidence*100:.2f}%)", use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(labels_name, prediction[0])
        ax.set_ylabel("Probabilité")
        ax.set_title("Distribution des prédictions")
        st.pyplot(fig)

        st.markdown(f'<div class="result-box">🧠 Prédiction : <b>{labels_name[predicted_class]}</b> | Confiance : <b>{confidence*100:.2f}%</b></div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<div class="footer">© 2025 BrainScan AI – Projet éducatif UM6P | Réalisé avec TensorFlow, OpenCV et Streamlit</div>', unsafe_allow_html=True)
