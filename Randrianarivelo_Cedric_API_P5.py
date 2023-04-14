# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:17:27 2023

@author: cedri
"""


import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import pickle
from Randrianarivelo_Cédric_P5_Prepross import main
import streamlit as st
import pandas as pd
import numpy as np
import joblib


#https://pypi.org/project/streamlit/
#streamlit run streamlit_app.py

st.set_page_config(page_title=' Prédiction de tags sur des questions ',
                   page_icon=':smiley:',
                   layout='wide',
                   initial_sidebar_state='expanded',
                   )
# Titre de la page
st.title("Prédiction de tags sur des questions")

# Sous-titre de la page
st.subheader("Modèle de NLP")

st.sidebar.header("Les parametres d'entrée")
st.sidebar.write("- Modèle entrainé sur 42 110 questions.")
st.sidebar.write("- Avec une précision de 50%")



##st.button
####========================================= 

# Charger le modèle préalablement entraîné
with open("Modele SGD Tf IDF.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("MLB.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("Model_Tf_Id.pkl", "rb") as f:
    TF_ID = pickle.load(f)

# Créer une fonction de prédiction
def predict_sentiment(text):
    pred = model.predict(text)
    return pred

def mlb_inverse(Pred) :
    
    pred = mlb.inverse_transform(Pred)[0]
    
    return pred 
# Créer l'interface utilisateur Streamlit
st.write('Entrez votre texte ci-dessous pour prédire son sentiment :')

# Ajouter un champ de saisie pour l'utilisateur
Title_input = st.text_input('Titre du Texte à classifier', '')
Body_input= st.text_input('Corps du Texte à classifier', '')



if st.button('Prédire'):
    # Faire une prédiction sur le texte saisi par l'utilisateur
    
    # Afficher la prédiction dans Streamlit
    
    data = {"Title" : [Title_input], "Body": [Body_input]}
    data = main(data)
    matrix_prediction = predict_sentiment(data)
    prediction = mlb_inverse(matrix_prediction)
    for idx, pred in enumerate (prediction) :
        st.write('Le tag numéro',idx,' prédit :', pred)

st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    color: #333333;
}
.stButton {
    background-color: #4CAF50;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)


# Ajouter un texte explicatif
st.write('Cet outil utilise un modèle de classification de sentiment NLP préalablement entraîné sur un ensemble de données provenant de StackOverFlow.')
st.write('Il peut prédire des tags en fonction de son contenu. Essayez-le maintenant pour voir à quel point il est précis et utile !')
if st.button('Quitter'):
    st.experimental.stop() 
    
## cd C:\Users\cedri\OneDrive\Documents\Cours\OpenClassroom\Cours\Projet_5
##  streamlit run Randrianarivelo_Cedric_API_P5.py



