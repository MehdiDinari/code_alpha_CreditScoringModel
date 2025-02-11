import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Charger et préparer les données
data = pd.read_csv('bank.csv', sep=';')
data.dropna(inplace=True)
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
X = data[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous',
          'job_blue-collar', 'job_management', 'job_self-employed', 'job_services',
          'job_student', 'job_technician', 'marital_married', 'marital_single',
          'education_secondary', 'education_tertiary', 'default_yes', 'housing_yes',
          'loan_yes']]  # Features

y = LabelEncoder().fit_transform(data['y'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]


# Fonction pour afficher la page de prédiction
def show_predict_page():
    st.title("Prédiction de Catégorie de Crédit")

    input_data = {}
    input_data['age'] = st.number_input("Âge", min_value=18, max_value=100, value=30, step=1)
    input_data['balance'] = st.number_input("Solde Bancaire", value=0.0)
    input_data['duration'] = st.number_input("Durée du contact en secondes", value=0.0)
    input_data['campaign'] = st.number_input("Nombre de contacts lors de la campagne", min_value=1, value=1, step=1)
    input_data['pdays'] = st.number_input("Nombre de jours depuis dernier contact", value=-1, step=1)
    input_data['previous'] = st.number_input("Nombre de contacts précédents", value=0, step=1)

    input_data['housing_yes'] = st.selectbox("Avez-vous un prêt immobilier?", [0, 1])
    input_data['loan_yes'] = st.selectbox("Avez-vous un prêt personnel?", [0, 1])
    input_data['job_blue-collar'] = st.selectbox("Travailleur manuel?", [0, 1])
    input_data['job_management'] = st.selectbox("Poste de management?", [0, 1])
    input_data['job_self-employed'] = st.selectbox("Travailleur indépendant?", [0, 1])
    input_data['job_services'] = st.selectbox("Secteur des services?", [0, 1])
    input_data['job_student'] = st.selectbox("Étudiant?", [0, 1])
    input_data['job_technician'] = st.selectbox("Technicien?", [0, 1])
    input_data['marital_married'] = st.selectbox("Marié?", [0, 1])
    input_data['marital_single'] = st.selectbox("Célibataire?", [0, 1])
    input_data['education_secondary'] = st.selectbox("Éducation secondaire?", [0, 1])
    input_data['education_tertiary'] = st.selectbox("Éducation supérieure?", [0, 1])
    input_data['default_yes'] = st.selectbox("Avez-vous un défaut de crédit?", [0, 1])

    if st.button("Prédire"):
        user_array = np.array(list(input_data.values())).reshape(1, -1)
        user_scaled = scaler.transform(user_array)
        user_prob = model.predict_proba(user_scaled)[:, 1][0]

        category = \
        pd.cut([user_prob], bins=[0, 0.25, 0.5, 0.75, 1], labels=['Refusé', 'Peu acceptable', 'Moyen', 'Bon'])[0]
        st.write(f"Catégorie de crédit prédite : **{category}**")

        # Ajouter un point sur le graphique de clustering
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        user_pca = pca.transform(user_scaled)

        plt.figure(figsize=(8, 6))
        kmeans = KMeans(n_clusters=4, random_state=42)
        y_kmeans = kmeans.fit_predict(X_pca)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', edgecolor='k', alpha=0.7)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200,
                    label="Centroids")
        plt.scatter(user_pca[:, 0], user_pca[:, 1], c='black', marker='o', s=200, label="Votre Position")
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Clustering K-Means avec position utilisateur")
        plt.legend()
        st.pyplot(plt)

