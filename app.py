import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display 
import numpy as np
import plotly.graph_objects as go 
from ScrapingDonnees_st import *
from statistiques_descriptives_st import *
from ML_test import *
st.set_option("client.showErrorDetails", False)


st.set_page_config(
    page_title="Application Machine Learning et Web Scraping",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)



st.markdown("""
    <style>
    
    body {
        background-color: white; 
    }
    
    
    section[data-testid="stSidebar"] {
        background-color:rgb(51, 0, 102); 
    }
    
    
    [data-testid="stSidebar"] .css-1aumxhk {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        width: 14px; 
        height: 14px;
        background-color:rgb(96, 96, 96); 
            border: 4.5px solid rgb(8, 122, 197); 
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    }

    
    [data-testid="stSidebar"] .stRadio input:checked + label > div:first-child div {
        background-color:rgb(96, 96, 96) !important;
    }

    
    [data-testid="stSidebar"] .stRadio > div > label {
        font-size: 1px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.title("📚Sommaire📚")
option = st.sidebar.radio(
    "SECTIONS",
    [
        "📋 Présentation du projet de Machine Learning et Web Scraping",
        "🌐 Scraping et présentation des données",
        "📈 Statistiques descriptives",
        "💻 Modèles de Machine Learning",
    ]
)


# Graphiques : 

@st.cache_data
def donnees(annees_debut, annees_fin):
    donnees=Donnees(annees_debut,annees_fin)
    dfgd=donnees.get_dataframe()
    return dfgd

df=donnees(2003, 2024)

# Affichage des sections selon l'option choisie
if option =="📋 Présentation du projet de Machine Learning et Web Scraping":
    st.markdown("# Projet de Machine Learning et Web Scraping")
    st.text(" ")
    st.write("Le rôle des données dans le sport a connu une révolution majeure au cours des deux dernières décennies. Aux États-Unis, notamment dans des ligues comme la NBA (basketball), la NFL (football américain), la MLB (baseball), et même la NHL (hockey), les équipes ont adopté des approches data-driven pour optimiser leurs performances, affiner leurs stratégies et améliorer leur prise de décision.")
    st.write("Dans le basket, en particulier la NBA, l'utilisation de la data est devenue essentielle à tous les niveaux : analyse des performances des joueurs, prédiction des blessures, évaluation des adversaires, gestion des contrats ect...")
    st.write("Les franchises NBA et d'autres équipes sportives professionnelles disposent aujourd'hui de cellules de data analytics appelées parfois Data Departments, Analytics Teams, ou Performance Labs. Ces départements sont composés de data analysts, data scientists, statisticiens, et parfois même ingénieurs en IA, qui travaillent avec les coachs, managers, et scouts.")
    st.write("Ce projet vise à explorer et à analyser les statistiques de carrière NBA de LeBron James en s'appuyant sur des données collectées via le site ESPN et en les exploitant à l'aide de modèles de Machine Learning")
    st.write("L'objectif principal est de réaliser un scraping des données relatives aux performances de LeBron James, de les explorer à travers des analyses statistiques descriptives et d'appliquer des algorithmes de Machine Learning pour effectuer des prédictions au scoring sur ses performances futures.")
    st.write("Ce projet a pour finalité de visualiser et comprendre les performances passées de LeBron James à travers des analyses statistiques et d'utiliser les données obtenues pour entraîner des modèles de prédiction, permettant d'estimer ses performances dans les matchs futurs.")
    
    st.write("Le projet est structuré en trois grandes parties :")
    st.write("1. Scraping et présentation des données")
    st.write("2. Analyse statistique descriptive")
    st.write("3. Modèles de Machine Learning")
    st.text(" ")
    st.markdown("Afin de réaliser ce projet de Machine Learning et Web Scraping, nous avons utilisé les librairies suivantes :")
    st.write("- pandas")
    st.write("- BeautifulSoup")
    st.write("- requests")
    st.write("- streamlit")
    st.write("- seaborn")
    st.write("- matplotlib")
    st.write("- IPython")
    st.write("- numpy")
    st.write("- scikit-learn")
    st.write("- plotly")
    st.write("- rich")
    st.write("Le code source de ce projet est disponible sur [GitHub]") ######
    
    st.markdown("Le projet a été réalisé par Jawad Gherras et Gabriel Edinger, étudiants en Master 2 Mecen à l'Université de Tours.") 



elif option == "🌐 Scraping et présentation des données":
    sommaire = st.sidebar.radio("Sommaire", ["Présentation des données", "Scraping des données", "Présentation du dataframe"])
    ## Présentation des données
    if sommaire == "Présentation des données":
        st.header("Statistiques de carrière de Lebron James")
        st.text(" ")
        
        st.write(" Les données collectées sur le site ESPN concernent les statistiques de LeBron James en NBA.")
        st.write("Voici un aperçu des données collectées provenant du site ESPN, sur les images suivantes : ")
        
        st.image("IMGS/ESPN1.png")
        st.image("IMGS/ESPN2.png")
        st.text(" ")
        st.write("[Pour plus d'informations sur les statistiques de LeBron James, cliquez ici](https://www.espn.com/nba/player/gamelog/_/id/1966/type/nba/year/2025)")
    
    ## Scraping des données
    elif sommaire == "Scraping des données":
        st.write("""Le scraping des données de LeBron James se fait à partir du site ESPN en analysant la structure HTML des pages de statistiques de ses matchs. 
                 Le site présente les données sous forme de tableaux mensuels regroupés dans des blocs HTML identifiables par une classe spécifique ("mb5"). 
                 Chaque bloc correspond à un mois de la saison et contient un tableau listant les performances de chaque match joué par LeBron.""")
        
        st.write("""Le processus de scraping commence par l'envoi d'une requête à la page web via la librairie requests pour obtenir le contenu HTML de la page. Ensuite, 
                 le contenu est analysé avec BeautifulSoup pour localiser les blocs contenant les tableaux de statistiques. 
                 Le code parcourt chaque bloc de manière itérative pour extraire les informations des tableaux HTML, comme la date du match, 
                 l'équipe adverse, le score, les points marqués, le nombre de minutes jouées, etc.""")
        
        st.write("""D'autre part, afin d'obtenir les données correspondantes uniquement à celles dans le tableau du site ESPN et non les autres informations, 
        nous avons supprimer les entetes comme la saisons, les statistiques totales par mois ou les statitiques totales pour la saison.""")
        
        st.write("""De plus, nous avons modifié la colonne OPP constitué du nom de l'équipe adverse et d'"@" qui est utilisé pour signifier que le match joué, est à l'extérieur et "vs" pour un match à domicile.  
                Nous avons gardé cette colonne OPP mais nous avons simplement laisser le nom de l'équipe adverse. 
                Une colonne location est créée constitué soit de "Extérieur" si auparavant on avait "@" et "Domicile" si on avait "vs".""")
        
        st.write(""" Pour finir, nous créons un dataframe avec toutes les données pour chaque saisons à l'aide de Pandas. Egalement nous avons modifié les types de données pour les colonnes afin de pourvoir les utiliser dans nos analyses et modèles.
                Les données collectées ont été sauvegardées dans un fichier CSV.""")
        st.write("Pour plus de détails sur le code source, veuillez consulter le fichier 'app.py' sur [GitHub]")######
    
    
    
    ## Présentation du dataframe 
    elif sommaire == "Présentation du dataframe":
        st.write(f"La base de données possède {df.shape[0]} lignes et {df.shape[1]} colonnes issues du site ESPN.")
        
        from io import StringIO
        memoire = StringIO()
        df.info(buf=memoire)
        info_str = memoire.getvalue()
        st.write("Informations sur le DataFrame et les types de données :")
        st.text(info_str)
        
        st.write("Statistiques descriptives du DataFrame :")
        st.write(df.describe())
        st.text(" ")
        st.write("Les colonnes du DataFrame sont les suivantes :")
        st.write(df.columns)
        st.write("Voici un aperçu des données collectées :")
        st.dataframe(df)

elif option == "📈 Statistiques descriptives":
    
    st.header("Statistiques descriptives")
    st.markdown("#### Présentation de plusieurs statistiques descriptives")
    
    graphique = st.sidebar.selectbox("Choisissez un graphique", ["Points marqués et matchs joués",
                                                             "Réussite au tir", 
                                                             "Victoires et défaites selon la localisation", 
                                                             "Adversaires de LeBron James",
                                                             "Distribution des points marqués",
                                                             ])
    
    if graphique == "Points marqués et matchs joués":
        st.subheader("Somme des points marqués par LeBron James et par année")
        st.write(" ")
        plot_points_somme_plotly(df)
        st.subheader("Moyenne de points par match et par année")
        st.write(" ")
        plot_points_moyens_no_text(df)
        st.subheader("Nombre de matchs joués par par année")
        st.write(" ")
        matchs_par_annee(df)
        st.subheader("Victoires et défaites de LeBron James par année")
        st.write(" ")
        victoires_par_annee(df)


    elif graphique == "Réussite au tir":
        st.subheader("Barplot du pourcentage de réussite au tir (FG%) par année")
        st.write(" ")
        pourcentage_reussite_par_annee(df)
        st.subheader("Barplot du pourcentage de réussite à 3 points (3P%) par année")
        st.write(" ")
        pourcentage_tirs_3_points_par_annee(df)

    elif graphique == "Victoires et défaites selon la localisation":
        st.subheader("Tableau des résultats selon la localisation")
        tab_localisation(df)
        st.write(" ")
        st.subheader("Diagramme à barres des résultats selon la localisation")
        plot_localisation_plotly(df)

    elif graphique == "Adversaires de LeBron James":
        st.subheader("Adversaires préférés de LeBron James")
        points_par_equipe_adverse(df)
        st.subheader("Moyenne de points par match contre les équipes adverses")
        st.write(" ")
        st.subheader("Les équipes contre lesquelles LeBron James a marqué le plus de points en moyenne par match.")
        points_moyens_par_equipe(df)
        st.write(" ")
        st.write("LeBron James a affronté 30 équipes différentes en NBA. Voici les équipes qu'il a le plus affrontées.")
        equipe_plus_affronte(df)
        st.write(" ")
        st.subheader("Somme des points encaissés par Lebron James selon les équipes")
        somme_des_points_encaissés_par_equipes(df)
        st.write(" ")
        st.subheader("Moyenne des points encaissés par les Lebron James selon les équipes")
        moyennes_des_points_encaissés_par_equipes(df)    
    
    elif graphique == "Distribution des points marqués":
        st.subheader("Distribution des points marqués par LeBron James")
        st.write(" ")
        distribution_points(df)    

elif option == "💻 Modèles de Machine Learning":
    
    choix_sous_sous_section = st.sidebar.radio("Sous-sous-sections", ["Etude des variables", "Etude des modèles"])
    
    st.header("Etude des variables et modèles de Machine Learning")
    st.text(" ")
    
    if choix_sous_sous_section == "Etude des variables":
        Varibles_selection=st.selectbox("Etude des variables", ["Matrices de correlation", "Nuages de points",
                                            "Boxplot et graphique des points"])
        if Varibles_selection == "Matrices de correlation" :
            st.write("Matrice de correlation linéaire entre les variables FG%, MIN et PTS")
            st.write(" ")
            correlation_entre_variables_MIN_FG_PTS(df)
            st.write(" ")
            st.write("Matrice de correlation non linéaire entre les variables FG%, MIN et PTS")
            st.write(" ")
            correlation_non_lineaire(df)
            st.write(" ")
            st.write("Matrice de correlation linéaire entre toutes les variables")
            st.write(" ")
            correlation_lineaire_entre_variables(df)
            st.write(" ")
            st.write("Matrice de correlation non linéaire entre toutes les variables")
            st.write(" ")
            correlation_non_lineaire_entre_variables(df)
        elif Varibles_selection == "Nuages de points":  
            st.write("Nuage de points entre les variables PTS et MIN")
            st.write(" ") 
            nuage_de_points_MIN_PTS(df)
            st.write(" ")
            st.write("Nuage de points entre les variables PTS et FG%")
            st.write(" ")
            nuage_de_points_FG_PTS(df)
            st.write(" ")
            st.write("Nuage de points entre les variables PTS et la localisation")
            st.write(" ")
            nuage_de_points_Location_PTS(df)
            st.write(" ")
            st.write("Nuage de points entre les variables PTS et le resultat")
            st.write(" ")
            nuage_de_points_Result_PTS(df)
        elif Varibles_selection == "Boxplot et graphique des points":
            st.write("Boxplot des points marqués par LeBron James selon le résultat du match")
            st.write(" ")
            boxplot_Result_PTS(df)
            st.write(" ")
            st.write("Graphique de l'évolution des points marqués par LeBron James selon l'année et la localisation")
            st.write(" ")
            evolution_points_annees(df)
    
    if choix_sous_sous_section == "Etude des modèles":
        choix_modele=st.selectbox("Modèles de Machine Learning", ["Préparation données", "KNN", "Régression linéaire", "Regression polynomiale", "SVR", "Gradient-Boosting", "Random Forest", "MLP", "Comparaison des modèles"])
        
        if choix_modele == "Préparation données":
                st.write("Préparation des données avec la fonction 'prep_donnees' issue de train_test_split de sklearn")
                st.markdown("#### Les données sont divisées en données d'entraînement et données de test avec un ratio de 80/20 par défaut.")
                st.write(" ")
                st.write("Une fonction 'prep_donnees' a été créée pour préparer les données pour les modèles de Machine Learning. Pour cela, nous avons utilisé la fonction 'train_test_split' de la librairie scikit-learn. Pour l'appeler, il suffit de lui passer : test_size=0.2 (taille de l'échantillon de test), random_state=42 (graine)")
                st.write("La variable cible (y) est 'PTS' (points marqués par LeBron James). Les variables explicatives (X) sont les autres variables.")
                st.write(" ")
                st.write("Les données d'entraînement et de test sont stockées dans les variables X_train, X_test, y_train et y_test. L'initialisation se fait avec la fonction 'prep_donnees(test_size=0.2, random_state=42)'.")
        elif choix_modele=="KNN":
            st.write("Modèle KNN")
            st.write(" ")
            st.write("Le modèle KNN (K-Nearest Neighbors) (sans acp) ici est un modèle de régression qui attribue une prédit un point en fonction de la moyenne des valeurs des k points les plus proches.")
            st.write(" ")
            st.write("Le modèle KNN a été entraîné avec les données d'entraînement et testé avec les données de test. La métrique utilisée est le NMSE (Negative Mean Squared Error).")
            st.write(" ")
            st.write("Pour l'appeler, il vous faut utiliser la fonction knn_avec_ou_sans_ACP(). Elle possède les paramètres suivants : n_neighbors (nombre de voisins), acp (booléen pour activer ou non l'ACP), n_components (nombre de composantes pour l'ACP), test_size (taille de l'échantillon de test), random_state (graine) choisis ou par défaut. Les paramètres finaux sont ceux qui optimisent le score d'entrainement avec l'aide de la méthode GridSearchCV().")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: knn_avec_ou_sans_ACP(ACP=False, n_neighbors=[3, 5, 7, 9], weights=['uniform', 'distance'], p=[1, 2], ACP_composantes=[2, 3, 4, 5], test_size=0.2, random_state=42): ")
            st.write(" ")
            st.image("IMGS/Knn.png")
            st.write(" ")
            st.image("IMGS/Knn_acp.png")
            
        elif choix_modele=="Régression linéaire":
            st.write("Modèle de régression linéaire")
            st.write(" ")
            st.write("Le modèle de régression linéaire est ici un modèle de régression qui cherche à établir une relation linéaire entre la variable cible et les variables explicatives.")
            st.write(" ")
            st.write("Ici ce modèle est divisé en trois sous modèles, linéaire, lasso et ridge. La métrique utilisée est le NMSE (Negative Mean Squared Error).")
            st.write(" ")
            st.write("Pour l'appeler, il vous faut utiliser la fonction regressions_lineaires(). Elle possède les paramètres suivants : type_de_modele ('lineaire', 'lasso', 'ridge'), alpha (paramètre de régularisation), test_size (taille de l'échantillon de test), random_state (graine)  choisis ou par défaut. Les paramètres finaux sont ceux qui optimisent le score d'entrainement avec l'aide de la méthode GridSearchCV().")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: regressions_lineaires(type_de_modele='lineaire', alpha=[0.001, 0.01, 0.1, 1, 10], test_size=0.2, random_state=42): ")
            st.write(" ")
            st.write("Régression linéaire")
            st.image("IMGS/lr.png")
            st.write(" ")
            st.write("Régression lasso")
            st.image("IMGS/lasso.png")
            st.write(" ")
            st.write("Régression ridge")
            st.image("IMGS/ridge.png")
        
        elif choix_modele=="Regression polynomiale":
            st.write("Modèle de régression polynomiale")
            st.write(" ")
            st.write("Le modèle de régression polynomiale est ici un modèle de régression qui cherche à établir une relation polynomiale entre la variable cible et les variables explicatives.")
            st.write(" ")
            st.write("La métrique utilisée est le NMSE (Negative Mean Squared Error).")
            st.write(" ")
            st.write("Pour l'appeler, il vous faut utiliser la fonction regression_polynomiale(). Elle possède les paramètres suivants : type_de_modele (lineaire, lasso, ridge), alpha (paramètre de régularisation), le degres (degré du polynome), test_size (taille de l'échantillon de test), random_state (graine), test_size (la taille de l'échantillon de test) choisis ou par défaut. Les paramètres finaux sont ceux qui optimisent le score d'entrainement avec l'aide de la méthode GridSearchCV().")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: regression_polynomiales(df, type_de_modele='lineaire', alpha=[0.001, 0.01, 0.1, 1, 10], degres=[1, 2, 3], test_size=0.2, random_state=42): ")
            st.write(" ")
            st.image("IMGS/poly_lr.png")
            st.write(" ")
            st.image("IMGS/poly_lasso.png")
            st.write(" ")
            st.image("IMGS/poly_ridge.png")
        
        elif choix_modele=="SVR":
            st.write("Modèle de SVR")
            st.write(" ")
            st.write("Le modèle de SVR (Support Vector Regression) est ici un modèle de régression qui cherche à établir une relation entre la variable cible et les variables explicatives.")
            st.write(" ")
            st.write("La métrique utilisée est le NMSE (Negative Mean Squared Error).")
            st.write(" ")
            st.write("Pour l'appeler, il vous faut utiliser la fonction svr(). Elle possède les paramètres suivants : kernel ('linear', 'poly', 'rbf', 'sigmoid'), C (paramètre de régularisation), epsilon (paramètre de liberté), test_size (taille de l'échantillon de test), random_state (graine) choisis ou par défaut. Les paramètres finaux sont ceux qui optimisent le score d'entrainement avec l'aide de la méthode GridSearchCV().")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: support_vector_regression(df, C=[0.1, 1, 10, 100], epsilon=[0.1, 0.01, 0.001], kernel=['linear', 'poly', 'rbf'], test_size=0.2, random_state=42): ")
            st.write(" ")
            st.write("Modèle de SVR")
            st.image("IMGS/SVR.png")
        
        elif choix_modele=="Gradient-Boosting":
            st.write("Modèle de Gradient-Boosting")
            st.write(" ")
            st.write("Le modèle de Gradient-Boosting est ici un modèle de régression qui cherche à établir une relation entre la variable cible et les variables explicatives.")
            st.write(" ")
            st.write("La métrique utilisée est le NMSE (Negative Mean Squared Error).")
            st.write(" ")
            st.write("Pour l'appeler, il vous faut utiliser la fonction gradient_boosting(). Elle possède les paramètres suivants : nbr_estimateurs (nombre d'estimateurs), profondeur_max (max_depth), taux_apprentissage (learning rate), test_size (taille de l'échantillon de test), random_state (graine) choisis ou par défaut. Les paramètres finaux sont ceux qui optimisent le score d'entrainement avec l'aide de la méthode GridSearchCV().")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: gradient_boosting(df, nbr_estimateurs=[200, 300, 1000], taux_apprentissage=[0.1, 0.01, 0.001], profondeur_max=[3, 5, 7, 9], test_size=0.2, random_state=42): ")
            st.write(" ")
            st.image("IMGS/gradient_boosting.png")
        
        elif choix_modele=="Random Forest":
            st.write("Modèle de Random Forest")
            st.write(" ")
            st.write("Le modèle de Random Forest est ici un modèle de régression qui cherche à établir une relation entre la variable cible et les variables explicatives.")
            st.write(" ")
            st.write("La métrique utilisée est le NMSE (Negative Mean Squared Error).")
            st.write(" ")
            st.write("Pour l'appeler, il vous faut utiliser la fonction random_forest(). Elle possède les paramètres suivants : nb_estimators (nombre d'estimateurs), max_depth (profondeur_max), min_samples_split (nombre minimum d'individu pour une split), test_size (taille de l'échantillon de test), random_state (graine) choisis ou par défaut. Les paramètres finaux sont ceux qui optimisent le score d'entrainement avec l'aide de la méthode GridSearchCV().")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: random_forest(df,n_estimators=[200, 300, 400], max_depth=[3, 5, 7, 9], min_samples_split=[2, 3, 4], test_size=0.2, random_state=42): ")
            st.write(" ")
            st.image("IMGS/random_forest.png")
        
        elif choix_modele=="MLP":
            st.write("Modèle de MLP")
            st.write(" ")
            st.write("Le modèle de MLP (Multi-Layer Perceptron) est ici un modèle de régression qui cherche à établir une relation entre la variable cible et les variables explicatives.")
            st.write(" ")
            st.write("La métrique utilisée est le NMSE (Negative Mean Squared Error).")
            st.write(" ")
            st.write("Pour l'appeler, il vous faut utiliser la fonction reseaux_neurones(). Elle possède les paramètres suivants : couchidden_layer_sizeshe_cachee (nombre de neurones par couche et nombre de couches cachées), solver (solver pour l'optimisation des poids), fct_activation (fonction d'activation), test_size (taille de l'échantillon de test), random_state (graine) choisis ou par défaut. Les paramètres finaux sont ceux qui optimisent le score d'entrainement avec l'aide de la méthode GridSearchCV().")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: reseaux_neurones(df, hidden_layer_sizes=[(50,), (100,), (50,50)], fct_activation=['relu','tanh', 'logistic'], solver=['lbfgs', 'adam'], test_size=0.2, random_state=42': ")
            st.write(" ")
            st.image("IMGS/reseau_neurones.png")
        
        elif choix_modele=="Comparaison des modèles":
            st.write("Comparaison des modèles")
            st.write(" ")
            st.write("Pour comparer les modèles, nous avons utilisé la fonction 'comparaison_modeles'.")
            st.write(" ")
            st.write("Cette fonction prend en paramètre les modèles à comparer et les données d'entraînement et de test. Elle renvoie un DataFrame avec les scores de chaque modèle.")
            st.write(" ")
            st.write("Voici un exemple d'appel et de résultats: ccomparaison_resultats(df, modele_teste=None, test_size=0.2, random_state=42, parametres_knn=None, parametres_knn_acp=None, parametres_lineaire=None, parametres_poly_lr=None, parametres_poly_lasso=None, parametres_poly_ridge=None, parametres_svr=None, parametres_gbr=None, parametres_rfr=None, parametres_mlp=None): ")
            st.write(" ")
            st.image("IMGS/comparaison_resultats.png")
