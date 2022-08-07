#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 12:35:09 2022

@author: abdarrahmane
"""

# Architecture réseau : CNN réseau LeNet-5
# Nb itérations       : 10
# Fct activation      : Relu, softmax (sortie)
# Mesure erreurs      : categorical_crossentropy
# Algo apprentissage  : adam
# Utilisation dropout : oui
# Couches de convolution : 5
# Couches de pooling  : 3
# Couches de Full connection  : 1
#---------------------------------------------------------
#    Bruitage en mode sel               : 
# Taux apprentissage  : 99,45 %
# taux validation     : 98,32 %
#---------------------------------------------------------
#    Bruitage en mode sel+inversion de couleur               : 
# Taux apprentissage  : 99,18 %
# taux validation     : 14,55 %
#---------------------------------------------------------

#############################################################################
#                FONCTIONNEMENT DU RESEAU LeNet-5                           #
#############################################################################

# Importation des modules TensorFlow & Keras
#  => construction et exploitation de réseaux de neurones
import tensorflow as tf

# Importation du module numpy
#  => manipulation de tableaux multidimensionnels
import numpy as np

# Importation du module graphique
#  => tracé de courbes et diagrammes
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Chargement des données d'apprentissage et de tests
#----------------------------------------------------------------------------
# Chargement en mémoire de la base de données des caractères MNIST
#  => tableaux de type ndarray (Numpy) avec des valeur entières
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Bruitage des images en grains de sel
for i in range(len(x_test)):
  sel =25
  noise = np.random.randint(sel+1, size=(24, 24))
  indexe = np.where(noise == sel)
  A = indexe[0]
  B = indexe[1]
  x_test[i][A,B] = 255

#inversion de couleur
for i in range(len(x_test)):
  rows , cols =x_test[i].shape
  for ligne in range(rows):
    for col in range(cols):
          x_test[i][ligne,col] = 255-x_test[i][ligne,col]

#----------------------------------------------------------------------------
# Changements de format pour exploitation
#----------------------------------------------------------------------------
# les valeurs associées aux pixels sont des entiers entre 0 et 255
#  => transformation en valeurs réelles entre 0.0 et 1.0
x_train, x_test = x_train / 255.0, x_test / 255.0
# Les données en entrée sont des matrices de pixels 28x28
#  => transformation en matrices 28x28 sur 1 plan en profondeur
#     (format en 4D nécessaire pour pouvoir réaliser des convolutions (conv2D))
print(x_test.shape)
x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)
# Les données de sortie sont des entiers associés aux chiffres à identifier
#  => transformation en vecteurs booléens pour une classification en 10 valeurs
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

#----------------------------------------------------------------------------
# DESCRIPTION du modèle LeNet-5 (CNN)
#  => 2 étapes Convolution + Pooling
#  => 3 couches FC (Full Connected)
#----------------------------------------------------------------------------
# Création d'un réseau multicouches
MonReseau = tf.keras.Sequential()

# C1: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
        filters=32,             # 32 noyaux de convolutions (32 feature maps)
        kernel_size=(3,3),     # noyau de convolution 3x3
        strides=(1,1),         # décalages horizontal=1 / vertical=1
        activation='relu',     # fct d'activation=Tanh
        input_shape=(28,28,1), # taille des entrées (car c'est la 1ère couche)
        padding='same'))       # ajout d'un bord à l'image pour éviter la 
                               # réduction de taille (nb de pixels calculé
                               # à partir de la taille du noyau)
#C2
MonReseau.add(tf.keras.layers.Conv2D(
        filters=32,            # 32 noyaux de convolutions (32 feature maps)
        kernel_size=(3,3),     # noyau de convolution 3x3
        strides=(1,1),         # décalages horizontal=1 / vertical=1
        activation='relu',     # fct d'activation=Tanh
        padding='same'))      # pas d'ajout de bord à l'image

# S1: description de la couche de pooling (average)
MonReseau.add(tf.keras.layers.AveragePooling2D(
        pool_size=(2,2),       # noyau de pooling 2x2
        strides=(2,2),         # décalages horizontal=2 / vertical=2
        padding='valid'))      # pas d'ajout de bord

MonReseau.add(tf.keras.layers.Dropout(0.25))
# C3: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
        filters=64,            # 64 noyaux de convolutions (64 feature maps)
        kernel_size=(3,3),     # noyau de convolution 3x3
        strides=(1,1),         # décalages horizontal=1 / vertical=1
        activation='relu',     # fct d'activation=Tanh
        padding='same'))      # pas d'ajout de bord à l'image

# C4: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
        filters=64,            # 64 noyaux de convolutions (64 feature maps)
        kernel_size=(3,3),     # noyau de convolution 3x3
        strides=(1,1),         # décalages horizontal=1 / vertical=1
        activation='relu',     # fct d'activation=Tanh
        padding='same'))      # pas d'ajout de bord à l'image

# S2: description de la couche de pooling (average)
MonReseau.add(tf.keras.layers.AveragePooling2D(
        pool_size=(2,2),       # noyau de pooling 2x2
        strides=(2,2),         # décalages horizontal=2 / vertical=2
        padding='valid'))      # pas d'ajout de bord à l'image
MonReseau.add(tf.keras.layers.Dropout(0.25))
# C5: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
        filters=128,            # 128 noyaux de convolutions (128 feature maps)
        kernel_size=(3,3),     # noyau de convolution 3x3
        strides=(1,1),         # décalages horizontal=1 / vertical=1
        activation='relu',     # fct d'activation=Tanh
        padding='same'))      # pas d'ajout de bord à l'image    

# S3: description de la couche de pooling (average)
MonReseau.add(tf.keras.layers.AveragePooling2D(
        pool_size=(3,3),       # noyau de pooling 3x3
        strides=(2,2),         # décalages horizontal=2 / vertical=2
        padding='valid'))      # pas d'ajout de bord à l'image
MonReseau.add(tf.keras.layers.Dropout(0.25))
# C5: connexion totale entre les pixels et la 1ère couche de 120 neurones
# Mise à plat des 16x(5x5)=400 pixels des images de convolution
MonReseau.add(tf.keras.layers.Flatten())
# Création d'un couche de 200 neurones avec fonction d'activation Tanh
MonReseau.add(tf.keras.layers.Dense(200, activation='relu'))

# FC6: connexion totale avec couche de 84 neurones avec fct d'activation Tanh
#MonReseau.add(tf.keras.layers.Dense(84, activation='tanh'))

MonReseau.add(tf.keras.layers.Dropout(0.25))

# Sortie: 10 neurones avec fct d'activation Softmax
MonReseau.add(tf.keras.layers.Dense(10, activation='softmax'))

# Affichage du descriptif du réseau
MonReseau.summary()


#----------------------------------------------------------------------------
# COMPILATION du réseau 
#  => configuration de la procédure pour l'apprentissage
#----------------------------------------------------------------------------
MonReseau.compile(optimizer='adam',                # algo d'apprentissage
                  loss='categorical_crossentropy', # mesure de l'erreur
                  metrics=['accuracy'])            # mesure du taux de succès

#----------------------------------------------------------------------------
# APPRENTISSAGE du réseau
#  => calcul des paramètres du réseau à partir des exemples
#----------------------------------------------------------------------------
hist=MonReseau.fit(x=x_train, # données d'entrée pour l'apprentissage
                   y=y_train, # sorties désirées associées aux données d'entrée
                   epochs=10, # nombre de cycles d'apprentissage 
                   batch_size=128, # taille des lots pour l'apprentissage
                   validation_data=(x_test,y_test)) # données de test

#------------------------------------------------------------
# Affichage des graphiques d'évolutions de l'apprentissage
#------------------------------------------------------------
# création de la figure ('figsize' pour indiquer la taille)
plt.figure(figsize=(8,8))
# evolution du pourcentage des bonnes classifications
plt.subplot(2,1,1)
plt.plot(hist.history['accuracy'],'o-')
plt.plot(hist.history['val_accuracy'],'x-')
plt.title("Taux d'exactitude des prévisions",fontsize=15)
plt.ylabel('Taux exactitude',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15)
plt.legend(['apprentissage', 'validation'], loc='lower right',fontsize=12)
# Evolution des valeurs de l'erreur résiduelle moyenne
plt.subplot(2,1,2)
plt.plot(hist.history['loss'],'o-')
plt.plot(hist.history['val_loss'],'x-')
plt.title('Ereur résiduelle moyenne',fontsize=15)
plt.ylabel('Erreur',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15)
plt.legend(['apprentissage', 'validation'], loc='upper right',fontsize=12)
# espacement entre les 2 figures
plt.tight_layout(h_pad=2.5)
plt.show()

# performances du réseau sur les données de tests
perf=MonReseau.evaluate(x=x_test, y=y_test)
print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100))

#----------------------------------------------------------------------------
# EXPLOITATION du réseau
#  => affichage des exemples de caractères bien et mal reconnus
#----------------------------------------------------------------------------
print('FIABILITE DU RESEAU:')
print('====================')
# Résultat du réseau avec des données de tests
perf=MonReseau.evaluate(x=x_test, # données d'entrée pour le test
                        y=y_test) # sorties désirées pour le test
print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100))
NbErreurs=int(10000*(1-perf[1]))
print("==>",NbErreurs," erreurs de classification !")
print("==>",10000-NbErreurs," bonnes classifications !")
# Calcul des prédictions du réseaux pour l'ensemble des données de test
Predictions=MonReseau.predict(x_test)
# Affichage des caractères bien/mal reconnus avec une matrice d'images
i=-1
Couleur='Red' # à remplacer par 'Green' pour les bonnes reconnaissances
plt.figure(figsize=(12,8), dpi=200)
for NoImage in range(12*8):
    i=i+1
    # '!=' pour les bonnes reconnaissances, '==' pour les erreurs
    while y_test[i].argmax() == Predictions[i].argmax(): i=i+1
    plt.subplot(8,12,NoImage+1)
    # affichage d'une image de digit, en format niveau de gris
    plt.imshow(x_test[i].reshape(28,28), cmap='Greys', interpolation='none')
    # affichage du titre (utilisatin de la méthode format du type str)
    plt.title("Prédit:{} - Correct:{}".format(MonReseau.predict(
                        x_test[i:i+1])[0].argmax(),y_test[i].argmax()),
                        pad=2,size=5, color=Couleur)
    # suppression des graduations sur les axes X et Y
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
# Affichage de la figure
plt.show()