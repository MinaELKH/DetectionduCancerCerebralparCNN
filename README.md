# 🧠 BrainScan – Détection du Cancer Cérébral par CNN

## 📌 Description du projet
**BrainScan** est une application basée sur le **Deep Learning** permettant de détecter automatiquement les **tumeurs cérébrales** à partir d’images IRM.  
Le modèle utilise un **réseau de neurones convolutionnel (CNN)** entraîné sur un dataset médical contenant quatre classes :

- 🧬 **Gliome**  
- 🧬 **Méningiome**  
- 🧬 **Tumeur hypophysaire (Pituitary)**  
- ✅ **Aucune tumeur (Notumor)**  

L’objectif est d’assister les médecins dans le diagnostic précoce et d’améliorer la précision des analyses radiologiques.

---

## ⚙️ Technologies utilisées
- **Python 3.10+**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **Streamlit**
- **Jupyter Notebook**

---

## 🧩 Architecture du projet

📦 breif3_DetectionduCancerCerebralparCNN
│
├── 📁 data/ # les images 
├── 📁 models/ # Modèles CNN sauvegardés (.h5)
├── 📁 docs/
│ ├── 📁 images/ # Graphiques & illustrations
│ ├── 📁 doc_streamlit.py # pages web contient les resultats et les analyses
│ 
├── 📁 notebooks/
│ ├── 1_preprocessing.ipynb
│ ├── 2_model_training.ipynb
│ ├── 3_evaluation.ipynb
│
├── app # Application Streamlit principale (page web qui permet de parcourir l image et voir la perdiction)
│ ├── app.py
├── requirements.txt # Liste des dépendances
└── README.md # Fichier de documentation

## 🔬 Étapes de réalisation

### 1️⃣ Prétraitement du dataset
- Chargement et nettoyage des images (OpenCV)
- Redimensionnement (224×224)
- Normalisation des pixels [0, 1]
- Encodage des labels
- Division train/test
- Rééquilibrage des classes via `ImageDataGenerator`

### 2️⃣ Entraînement du modèle CNN
Architecture du réseau :
```python
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
Optimiseur : Adam
Fonction de perte : categorical_crossentropy
Époques : 20
Batch size : 32

📈 Évaluation du modèle
Classe	Précision	Rappel	F1-score
Glioma	0.91	0.89	0.90
Meningioma	0.87	0.90	0.88
Notumor	0.95	0.97	0.96
Pituitary	0.93	0.91	0.92

✅ Accuracy globale : 92%

📊 Résultats visuels
Image	Description
Courbe d’accuracy (train/validation)
Matrice de confusion du modèle
Échantillons d’images du dataset
Répartition des classes
Exemple de prédiction avec le modèle

🧠 Interprétation
Le modèle montre une forte capacité de généralisation, avec des erreurs limitées entre glioma et meningioma (due à leur similarité visuelle).
L’accuracy stable entre train et validation prouve une absence de surapprentissage.
Les courbes montrent une convergence rapide et une bonne séparation des classes dans la matrice de confusion.

🚀 Déploiement avec Streamlit
L’application permet à l’utilisateur de :

Télécharger une image IRM.

Visualiser la classe prédite et la probabilité associée.

Afficher les performances du modèle.




▶️ Exécution locale

*Installer les dépendances :
pip install -r requirements.txt


*Lancer l’application Streamlit :
cd app
streamlit run app.py

Ouvrir dans le navigateur :
👉 http://localhost:8501

🧾 Exemple de prédiction
charger l image 
Classe prédite : meningioma
Confiance : 94.6 %