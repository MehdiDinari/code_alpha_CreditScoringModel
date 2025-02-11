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


def show_explore_page():
    st.title("Exploration des Données")

    # Matrice de confusion
    plt.figure(figsize=(6, 4))
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Credit', 'Credit'],
                yticklabels=['No Credit', 'Credit'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Matrice de Confusion")
    st.pyplot(plt)

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe ROC-AUC")
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Importance des caractéristiques
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    feature_importance[:10].plot(kind='bar', color='blue')
    plt.title("Top 10 Features les plus importantes")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    st.pyplot(plt)

    # Réduction de dimension avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    # Clustering K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_kmeans = kmeans.fit_predict(X_pca)

    # Définition des catégories
    category_labels = {0: "Refusé", 1: "Peu acceptable", 2: "Moyen", 3: "Bon"}
    cluster_names = [category_labels[label] for label in y_kmeans]

    # Tracé du clustering
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200,
                label="Centroids")
    handles, _ = scatter.legend_elements()
    legend_labels = [category_labels[i] for i in range(4)]
    plt.legend(handles, legend_labels, title="Catégories")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.title("Clustering K-Means avec 4 catégories nommées")
    st.pyplot(plt)

    # Catégorisation des probabilités de prédiction
    categories = pd.cut(y_prob, bins=[0, 0.25, 0.5, 0.75, 1], labels=['Refusé', 'Peu acceptable', 'Moyen', 'Bon'])
    category_counts = categories.value_counts().sort_index()

    # Graphique en barres
    plt.figure(figsize=(6, 4))
    category_counts.plot(kind='bar', color=['red', 'orange', 'yellow', 'green'])
    plt.xlabel("Catégorie")
    plt.ylabel("Nombre de clients")
    plt.title("Répartition des prédictions en 4 catégories")
    plt.xticks(rotation=0)
    st.pyplot(plt)

    # Graphique en secteurs (camembert)
    plt.figure(figsize=(6, 6))
    category_counts.plot(kind='pie', autopct='%1.1f%%', colors=['red', 'orange', 'yellow', 'green'])
    plt.ylabel('')
    plt.title("Répartition des prédictions en 4 catégories")
    st.pyplot(plt)


# Exécuter la page d'exploration
if __name__ == "__main__":
    show_explore_page()
