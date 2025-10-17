# Traitement et gestion des données
import numpy as np
import pandas as pd
import os
import cv2
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Deep Learning (Keras / TensorFlow)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


from tensorflow.keras.preprocessing import image




# Mesure du temps d’entraînement
import time

# Interface (pour le déploiement avec Streamlit)
import streamlit as st






## variables utlisées
images = []
labels = []
dict_labels_list_images_index = {}
dict_labels_images_count={}


## parcourir data set 
dataset_path = "../Data"
valid_extension = ('.jpeg' , '.jpg' , '.bmp' , '.png')

# parcourir chaque sous-dossier
img_size = (224 , 224)
for class_name in os.listdir(dataset_path) :
    class_folder = os.path.join(dataset_path , class_name)
    if os.path.isdir(class_folder) :
        for file_name in os.listdir(class_folder) :
            if file_name.lower().endswith(valid_extension) : 
                img_path = os.path.join(class_folder , file_name)
                try : 
                    img = cv2.imread(img_path)
                    if img is not None :
                        img = cv2.resize(img, img_size)  ## redimenssionner 
                        images.append(img)
                        labels.append(class_name)
                    else :
                        print(f"Image illisible : {img_path}")
                except Exception as e :
                     print(f"Erreur lors du chargement de {img_path} : {e}")
            else:
                print(f"Extension invalide : {file_name}")
# Convertir les listes en tableaux NumPy
images = np.array(images)
labels = np.array(labels)

print(f"Nombre d'images valides chargées : {len(images)}")
print(f"les noms des classes : {np.unique(labels)}")
# melange avec methode shuffle
images, labels = shuffle(images, labels, random_state=42)

## affichage des graphes (nombre d image par classe)
df = pd.DataFrame({'label' : labels})
count_by_class = df['label'].value_counts()

sns.countplot(x=labels)  ## le x peut etre une dataframe ou liste , il complte le nombre d occurences de chaq categorie
plt.title("Répartition des classes dans le dataset")
plt.xlabel("Classe")
plt.ylabel("Nombre d’images")
plt.show()


## echantillon d image 

# Définir le nombre d'images à afficher par classe
n_samples_per_class = 5

# Obtenir les classes uniques
classes = np.unique(labels)   

# Créer la figure
plt.figure(figsize=(15, len(classes)*3))

# Parcourir chaque classe
for i, cls in enumerate(classes):
    # Trouver les indices des images appartenant à cette classe
    idxs = np.where(labels == cls)[0]
    print(f"{cls} : {idxs}")
    # Sélectionner aléatoirement n_samples_per_class images
    selected_idxs = np.random.choice(idxs, size=n_samples_per_class, replace=False)
    dict_labels_list_images_index[cls] = idxs
    for j, idx in enumerate(selected_idxs):
        plt_idx = i * n_samples_per_class + j + 1
        plt.subplot(len(classes), n_samples_per_class, plt_idx)
        plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))  # Convertir BGR -> RGB
        plt.axis('off')
        if j == 0:
            plt.ylabel(cls, fontsize=12)
            
plt.suptitle("Échantillons d'images par classe", fontsize=16)
plt.tight_layout()
plt.show()


## eqluipre data 
print(dict_labels_list_images_index)
for label in dict_labels_list_images_index:
    num_images = len(dict_labels_list_images_index[label])
    dict_labels_images_count[label] = num_images
    print(f"{label} : {num_images} images")

# max
max_nb_img = max(dict_labels_images_count.values())
print(f"\n nombre max des images {max_nb_img}")


def fun_dict_labels_list_images_index(dict_labels_list_images_index) :
    for label in dict_labels_list_images_index:
        num_images = len(dict_labels_list_images_index[label])
        dict_labels_images_count[label] = num_images
        return  dict_labels_images_count 
print(fun_dict_labels_list_images_index(dict_labels_list_images_index))






# Création du générateur d'images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

blanced_images = []
blanced_labels = []

for label, count_img in dict_labels_images_count.items():
    # Récupérer les indices des images de cette classe
    idxs = dict_labels_list_images_index[label]
    while count_img < max_nb_img:
        # Choisir une image existante aléatoire
        idx = np.random.choice(idxs)
        img = images[idx]

        # Ajouter une dimension pour le batch
        img_batch = np.expand_dims(img, 0)

        # Générer une image augmentée
        batch = next(datagen.flow(img_batch, batch_size=1))
        new_gen_img = batch[0].astype(np.uint8)

        # Ajouter l'image et son label
        blanced_images.append(new_gen_img)
        blanced_labels.append(label)
        count_img += 1

    

# Convertir en arrays
blanced_images = np.array(blanced_images)
blanced_labels = np.array(blanced_labels)


# Concaténer les arrays
all_images = np.concatenate((images, blanced_images), axis=0)
all_labels = np.concatenate((labels, blanced_labels), axis=0)

print("Nombre total de labels :", len(all_labels))
print("Nombre d'images générées :", len(blanced_images))


## transformer les etiquette textuelles en format numerique 
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
labels_encoded = le.fit_transform(labels) 
labels_onehot = to_categorical(labels_encoded)
print("Shape du one-hot :", labels_onehot.shape)
print(labels_onehot[:5])  # afficher les 5 premiers vecteurs


## normaliser 
images = np.array(images, dtype='float32') / 255.0


## diviser les donnees 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_onehot,       
    test_size=0.2,                
    random_state=42,            
    stratify=labels_encoded       # pour garder le même ratio de classes
)


############    Conception du modele CNN ##################

# definir l architecture 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

num_classes = len(np.unique(labels))  # à définir selon ton dataset

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])


# Compilation du modèle
model.compile(
    optimizer='adam',                 
    loss='categorical_crossentropy',  
    metrics=['accuracy']              
)

# summary 

# Afficher un résumé du modèle
model.summary()

# Hyperparamètres principaux
learning_rate = 0.001   # Taux d'apprentissage pour Adam
epochs = 20             # Nombre d'époques pour l'entraînement
batch_size = 32         # Nombre d'images traitées à chaque itératio
# chekpoint 
checkpoint = ModelCheckpoint(
    filepath='../models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

## time 

start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint]
)

end_time = time.time()
training_time = end_time - start_time
print(f"Durée totale de l'entraînement : {training_time:.2f} secondes")
print(f"Durée totale en minute : {training_time/60:.2f} minutes")

# evalue module 

best_model = load_model('../models/best_model.h5')
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Précision sur l'ensemble de test : {test_accuracy:.4f}")
print(f"Perte sur l'ensemble de test : {test_loss:.4f}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Courbe Accuracy')
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Courbe Loss')
plt.show()

## matrice de confusion
y_pred = best_model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion')
plt.xlabel('Prédiction')
plt.ylabel('Vérité')
plt.show()




## rapport classifier 

print(classification_report(y_true, y_pred_classes))

# afficher prediction 
correct_idx = np.where(y_pred_classes == y_true)[0]
incorrect_idx = np.where(y_pred_classes != y_true)[0]

plt.figure(figsize=(10,5))
for i, idx in enumerate(correct_idx[:5]):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[idx])
    plt.title(f"Vrai:{y_true[idx]} / Préd:{y_pred_classes[idx]}")
    plt.axis('off')

for i, idx in enumerate(incorrect_idx[:5]):
    plt.subplot(2,5,i+6)
    plt.imshow(X_test[idx])
    plt.title(f"Vrai:{y_true[idx]} / Préd:{y_pred_classes[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


############  deploiment et utlisation _ prediction ##########


# Exemple de classes (adapter selon votre dataset)
labels_name = np.unique(labels)   #  les noms des classes : ['glioma' 'meningioma' 'notumor' 'pituitary']

# Charger le modèle entraîné
model = load_model('../models/best_model.h5', compile=False)


def predict_image(path):
    img = image.load_img(path, target_size=(224, 224))  # 1. Charger et redimensionner
    img_array = image.img_to_array(img)                 # 2. Convertir en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)       # 3. Ajouter la dimension batch
    img_array = img_array / 255.0                       # 4. Normaliser entre 0 et 1

    prediction = model.predict(img_array)               # 5. Faire la prédiction
    predicted_class = np.argmax(prediction, axis=1)[0]  # 6. Classe prédite (indice)
    confidence = np.max(prediction)                     # 7. Probabilité associée

    print(f"Classe prédite : {labels_name[predicted_class]}")
    print(f"Confiance : {confidence*100:.2f}%")
    return labels_name[predicted_class], confidence


## test image existante

predict_image("../test_images/image_Te-gl_0027.jpg") ## image de glioma image_Te-gl_0027


#########  l app stream lit est sur le dossier app 

# Dans ton terminal, place-toi dans le dossier du fichier et lance :

# streamlit run app.py