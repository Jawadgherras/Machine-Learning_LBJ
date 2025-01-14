"""Fichier des modèles de Machine Learning pour les données de LeBron James."""
from maintest import ESPNWebScraper, Conversion_de_types, Donnees

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    dataframe_LB=Donnees(2004, 2024)
    df=dataframe_LB.get_dataframe()


# Analyse des corrélations et relations entre les variables
## Corrélation (linéaire et non linéaire) entre certaines variables explicatives et la variable à expliquer
def correlation_entre_variables_MIN_FG_PTS(df):
    """Matrice de corrélation linéaire entre les variables MIN, FG% et PTS."""
    colonnes_choisies = ["MIN","FG%", "PTS"]
    df_variables=df[colonnes_choisies]
    correlation_lineaire=df_variables.corr(method="pearson")
    sns.heatmap(correlation_lineaire, annot=False, cmap="coolwarm")
    plt.title("Matrice de corrélation des variables")
    plt.show()

def correlation_non_lineaire(df):
    """Matrice de corrélation non linéaire entre les variables MIN, FG% et PTS."""
    colonnes_choisies = ["MIN","FG%", "PTS"]
    df_variables=df[colonnes_choisies]
    correlation_non_lineaire=df_variables.corr(method="spearman")
    sns.heatmap(correlation_non_lineaire, annot=False, cmap="coolwarm")
    plt.title("Matrice de corrélation des variables")
    plt.show()

def correlation_lineaire_entre_variables(df):
    """Matrice de corrélation linéaire entre les variables MIN, FG%, 3P%, Location, Result, FT%, REB, AST, BLK, STL, PF, TO et PTS."""
    colonnes_choisies = ["MIN","FG%", "3P%", "Location", "Result", "FT%", "REB", "AST", "BLK", "STL", "PF", "TO", "PTS"]
    df_variables=df[colonnes_choisies]
    correlation_lineaire=df_variables.corr(method="pearson")
    sns.heatmap(correlation_lineaire, annot=False, cmap="coolwarm")
    plt.title("Matrice de corrélation des variables")
    plt.show()

def correlation_non_lineaire_entre_variables(df):
    """Matrice de corrélation linéaire entre les variables MIN, FG%, 3P%, Location, Result, FT%, REB, AST, BLK, STL, PF, TO et PTS."""
    colonnes_choisies = ["MIN","FG%", "3P%", "Location", "Result", "FT%", "REB", "AST", "BLK", "STL", "PF", "TO", "PTS"]
    df_variables=df[colonnes_choisies]
    correlation_lineaire=df_variables.corr(method="spearman")
    sns.heatmap(correlation_lineaire, annot=False, cmap="coolwarm")
    plt.title("Matrice de corrélation des variables")
    plt.show()


# Visualisation des relations entre les variables

## Nuage de points entre les points marqués et le temps de jeu
def nuage_de_points_MIN_PTS(df):
    """Nuage de points entre les points marqués et le temps de jeu."""
    sns.scatterplot(x="MIN", y="PTS", data=df, palette="colorblind")
    plt.title("Nuage de points entre les points marqués et le temps de jeu")
    plt.show()

## Nuage de points entre les points marqués et le pourcentage de réussite aux tirs
def nuage_de_points_FG_PTS(df):
    """Nuage de points entre les points marqués et le pourcentage de réussite aux tirs."""
    sns.scatterplot(x="FG%", y="PTS", hue="Location", data=df, palette="colorblind")
    plt.title("Nuage de points entre les points marqués et le pourcentage de réussite aux tirs")
    plt.show()


## Nuage de points entre les points marqués et le localisation du match
def nuage_de_points_Location_PTS(df):
    sns.scatterplot(x="Location", y="PTS", data=df)
    plt.title("Nuage de points entre les points marqués et la localisation du match")
    plt.show()

## Nuage de points entre les points marqués et le résultat du match
def nuage_de_points_Result_PTS(df):
    sns.scatterplot(x="Result", y="PTS", hue="Location", data=df, palette="viridis")
    plt.title("Nuage de points entre les points marqués et le résultat du match")
    plt.show()

## Boxplot des points marqués et du résultat du match
def boxplot_Result_PTS(df):
    sns.boxenplot(x="Result", y="PTS", data=df, palette="viridis")
    plt.title("Boxplot des points marqués et du résultat du match")
    plt.show()

## Evolution des points marqués en fonction des années
def evolution_points_annees(df):
    sns.lineplot(x="Year", y="PTS", hue="Location", data=df, palette='coolwarm', marker='o')
    plt.title("Evolution des points marqués en fonction des années")
    plt.show()


# Modèles de Machine Learning
## Séparation des données en données d'entrainement et de test
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

def prep_donnees(df, test_size=0.2, random_state=42):
    """Préparation des données pour les modèles de Machine Learning. Séparation de la variable à expliquer ("PTS") et des variables explicatives.
    Puis division des données en données d'entrainement et de test.
    
    Paramètres: 
        df: Un DataFrame issue de la classe Donnees.
        test_size: La proportion des données de test.
        random_state: Le seed pour la reproductibilité des résultats.
        
    Retourne:
        X_train, X_test, y_train, y_test: Les données d'entrainement et de test.
        tuple: (X_train, X_test, y_train, y_test)
            - X_train: Les variables explicatives d'entrainement.
            - X_test: Les variables explicatives de test.
            - y_train: La variable à expliquer d'entrainement.
            - y_test: La variable à expliquer de test.
    """

    X=df.drop(columns="PTS")
    y=df["PTS"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)




from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# KNN avec ou sans ACP:
def knn_avec_ou_sans_ACP(df, ACP=False, n_neighbors=[3, 5, 7, 9], weights=["uniform", "distance"], p=[1, 2], ACP_composantes=[2, 3, 4, 5], test_size=0.2, random_state=42):
    """Modèle de KNN avec ou sans ACP. Uitlisation de GridSearchCV pour trouver les meilleurs hyperparamètres."""

    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)

    if ACP:
            pipeline = Pipeline([
                ("pca", PCA()),
                ("knn", KNeighborsRegressor())
            ])
            parametres_knn = {
                "knn__n_neighbors": n_neighbors,
                "knn__weights": weights,
                "knn__p": p,
                "pca__n_components": ACP_composantes
                        }
    else:
        pipeline = KNeighborsRegressor()
        parametres_knn = {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "p": p
        }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=parametres_knn, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    
    meilleur_modele=grid_search.best_estimator_
    meilleurs_parametres=grid_search.best_params_
    meilleur_score=grid_search.best_score_
    
    y_pred=meilleur_modele.predict(X_test)
    test_mse=mean_squared_error(y_test, y_pred)
    test_R2=r2_score(y_test, y_pred)
    
    print(f"Meilleurs hyperparamètres trouvés : {meilleurs_parametres}")
    print(f"Meilleur score (négatif de l'erreur quadratique moyenne) : {meilleur_score}")
    print(f"Erreur quadratique moyenne sur les données de test : {test_mse}")
    print(f"R2 score sur les données de test : {test_R2}")
    print(f"Meilleur modèle : {meilleur_modele}")
    
    return meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele 

from sklearn.linear_model import LinearRegression, Lasso, Ridge
# Régressions linéaires
def regressions_lineaires(df, type_de_modele="lineaire", alpha=[0.001, 0.01, 0.1, 1, 10], test_size=0.2, random_state=42):
    """Modèles de régression linéaire, lasso et ridge. Utilisation de GridSearchCV pour trouver les meilleurs hyperparamètres."""
    if type_de_modele=="lineaire":
        modele=LinearRegression()
        parametres_lineaire={}
    elif type_de_modele=="lasso":
        modele=Lasso()
        parametres_lineaire={"alpha": alpha}
    elif type_de_modele=="ridge":
        modele=Ridge()
        parametres_lineaire={"alpha": alpha}
    else:
        raise ValueError("Le type de modèle doit être 'lineaire', 'lasso' ou 'ridge'.")

    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)
    
    grid_search=GridSearchCV(estimator=modele, param_grid=parametres_lineaire, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_seach=grid_search.fit(X_train, y_train)

    meilleur_modele=grid_search.best_estimator_
    meilleurs_parametres=grid_search.best_params_
    meilleur_score=grid_search.best_score_
    
    y_pred=meilleur_modele.predict(X_test)
    test_mse=mean_squared_error(y_test, y_pred)
    test_R2=r2_score(y_test, y_pred)
    
    print(f"Meilleurs hyperparamètres trouvés : {meilleurs_parametres}")
    print(f"Meilleur score (négatif de l'erreur quadratique moyenne) : {meilleur_score}")
    print(f"Erreur quadratique moyenne sur les données de test : {test_mse}")
    print(f"R2 score sur les données de test : {test_R2}")
    print(f"Meilleur modèle : {meilleur_modele}")
    
    return meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Régressions polynomiales
def regression_polynomiales(df, type_de_modele="lineaire", alpha=[0.001, 0.01, 0.1, 1, 10], degres=[1, 2, 3], test_size=0.2, random_state=42):
    """Régressions polynomiales avec GridSearchCV pour trouver les meilleurs hyperparamètres."""

    if type_de_modele=="lineaire":
        modele=make_pipeline(PolynomialFeatures(), LinearRegression())
        parametres_poly={"polynomialfeatures__degree": degres}
    elif type_de_modele=="lasso":
        modele=make_pipeline(PolynomialFeatures(), Lasso())
        parametres_poly={"polynomialfeatures__degree": degres, "lasso__alpha": alpha}
    elif type_de_modele=="ridge":
        modele=make_pipeline(PolynomialFeatures(), Ridge())
        parametres_poly={"polynomialfeatures__degree": degres, "ridge__alpha": alpha}
    else:
        raise ValueError("Le type de modèle doit être 'lineaire', 'lasso' ou 'ridge'.")

    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)
    
    grid_search=GridSearchCV(estimator=modele, param_grid=parametres_poly, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_search=grid_search.fit(X_train, y_train)
    
    meilleur_modele=grid_search.best_estimator_
    meilleurs_parametres=grid_search.best_params_
    meilleur_score=grid_search.best_score_
    
    y_pred=meilleur_modele.predict(X_test)
    test_mse=mean_squared_error(y_test, y_pred)
    test_R2=r2_score(y_test, y_pred)
    
    
    print(f"Meilleurs hyperparamètres trouvés : {meilleurs_parametres}")
    print(f"Meilleur score (négatif de l'erreur quadratique moyenne) : {meilleur_score}")
    print(f"Erreur quadratique moyenne sur les données de test : {test_mse}")
    print(f"R2 score sur les données de test : {test_R2}")
    print(f"Meilleur modèle : {meilleur_modele}")
    
    
    return meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele



from sklearn.svm import SVR
# Support Vector Regression
def support_vector_regression(df, C=[0.1, 1, 10, 100], epsilon=[0.1, 0.01, 0.001], kernel=["linear", "poly", "rbf"], test_size=0.2, random_state=42):
    """Support Vector Regression avec GridSearchCV pour trouver les meilleurs hyperparamètres."""
    
    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)
    
    svr=SVR()
    
    parametres_svr={"C": C, "epsilon": epsilon, "kernel": kernel}
    
    grid_search=GridSearchCV(estimator=svr, param_grid=parametres_svr, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    meilleur_modele=grid_search.best_estimator_
    meilleurs_parametres=grid_search.best_params_
    meilleur_score=grid_search.best_score_
    
    y_pred=grid_search.predict(X_test)
    test_mse=mean_squared_error(y_test, y_pred)
    test_R2=r2_score(y_test, y_pred)
    
    
    print(f"Meilleurs hyperparamètres trouvés : {meilleurs_parametres}")
    print(f"Meilleur score (négatif de l'erreur quadratique moyenne) : {meilleur_score}")
    print(f"Erreur quadratique moyenne sur les données de test : {test_mse}")
    print(f"R2 score sur les données de test : {test_R2}")
    print(f"Meilleur modèle : {meilleur_modele}")
    
    return meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele



from sklearn.ensemble import GradientBoostingRegressor

# Gradient-Boosting
def gradient_boosting(df, nbr_estimateurs=[200, 300, 1000], taux_apprentissage=[0.1, 0.01, 0.001], profondeur_max=[3, 5, 7, 9], test_size=0.2, random_state=42):
    """Gradient-Boosting avec GridSearchCV pour trouver les meilleurs hyperparamètres."""
    
    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)
    
    gbr=GradientBoostingRegressor()
    
    parametres_gbr={"n_estimators": nbr_estimateurs, "learning_rate": taux_apprentissage, "max_depth": profondeur_max}
    
    grid_grb=GridSearchCV(estimator=gbr, param_grid=parametres_gbr, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_grb.fit(X_train, y_train)
    
    meilleur_modele=grid_grb.best_estimator_
    meilleurs_parametres=grid_grb.best_params_
    meilleur_score=grid_grb.best_score_
    
    y_pred=grid_grb.predict(X_test)
    test_mse=mean_squared_error(y_test, y_pred)
    test_R2=r2_score(y_test, y_pred)
    
    print(f"Meilleurs hyperparamètres trouvés : {meilleurs_parametres}")
    print(f"Meilleur score (négatif de l'erreur quadratique moyenne) : {meilleur_score}")
    print(f"Erreur quadratique moyenne sur les données de test : {test_mse}")
    print(f"R2 score sur les données de test : {test_R2}")
    print(f"Meilleur modèle : {meilleur_modele}")
    
    
    return meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele


from sklearn.ensemble import RandomForestRegressor

# Random Forest
def random_forest(df,n_estimators=[200, 300, 400], max_depth=[3, 5, 7, 9], min_samples_split=[2, 3, 4], test_size=0.2, random_state=42):
    """Random Forest avec GridSearchCV pour trouver les meilleurs hyperparamètres."""
    
    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)
    
    rfr=RandomForestRegressor()
    
    parametres_rfr={"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split}
    
    grid_rfr=GridSearchCV(estimator=rfr, param_grid=parametres_rfr, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_rfr.fit(X_train, y_train)
    
    meilleur_modele=grid_rfr.best_estimator_
    meilleurs_parametres=grid_rfr.best_params_
    meilleur_score=grid_rfr.best_score_
    
    y_pred=grid_rfr.predict(X_test)
    test_mse=mean_squared_error(y_test, y_pred)
    test_R2=r2_score(y_test, y_pred)
    
    
    print(f"Meilleurs hyperparamètres trouvés : {meilleurs_parametres}")
    print(f"Meilleur score (négatif de l'erreur quadratique moyenne) : {meilleur_score}")
    print(f"Erreur quadratique moyenne sur les données de test : {test_mse}")
    print(f"R2 score sur les données de test : {test_R2}")
    print(f"Meilleur modèle : {meilleur_modele}")
    
    return meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele


from sklearn.neural_network import MLPRegressor

# Réseau de neurones
def reseaux_neurones(df, hidden_layer_sizes=[(50,), (100,), (50,50)], fct_activation=["relu","tanh", "logistic"], solver=["lbfgs", "adam"], test_size=0.2, random_state=42):
    """Réseau de neurones avec GridSearchCV pour trouver"""
    
    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)
    
    mlp=MLPRegressor()
    
    parametres_mlp={"hidden_layer_sizes": hidden_layer_sizes, "activation": fct_activation, "solver": solver}
    
    grid_mlp=GridSearchCV(estimator=mlp, param_grid=parametres_mlp, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_mlp.fit(X_train, y_train)
    
    meilleur_modele=grid_mlp.best_estimator_
    meilleurs_parametres=grid_mlp.best_params_
    meilleur_score=grid_mlp.best_score_
    
    y_pred=grid_mlp.predict(X_test)
    test_mse=mean_squared_error(y_test, y_pred)
    test_R2=r2_score(y_test, y_pred)
    
    print(f"Meilleurs hyperparamètres trouvés : {meilleurs_parametres}")
    print(f"Meilleur score (négatif de l'erreur quadratique moyenne) : {meilleur_score}")
    print(f"Erreur quadratique moyenne sur les données de test : {test_mse}")
    print(f"R2 score sur les données de test : {test_R2}")
    print(f"Meilleur modèle : {meilleur_modele}")
    
    return meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele

#reseaux_neurones(df, hidden_layer_sizes=[(50,), (100,), (50,50)], fct_activation=["relu","tanh", "logistic"], solver=["lbfgs", "adam"], test_size=0.2, random_state=42)



# Comparaison des résultats des modèles 
#erreurs={"KNN": test_knn, "KNN avec PCA": test_knn_pca, "Régression linéaire": test_lr, "Régression lasso": test_lasso, "Régression ridge": test_ridge, "Régression linéaire polynomiale": mean_squared_error(y_test, y_pred_lr_poly), "Régression lasso polynomiale": mean_squared_error(y_test, y_pred_lasso_poly), "Régression ridge polynomiale": mean_squared_error(y_test, y_pred_ridge_poly), "SVR": mean_squared_error(y_test, y_pred_svr), "Gradient-Boosting": mean_squared_error(y_test, y_pred_gbr), "Random Forest": mean_squared_error(y_test, y_pred_rfr), "Réseau de neurones": mean_squared_error(y_test, y_pred_mlp)}

from rich.console import Console
from rich.table import Table


def comparaison_resultats(df, modele_teste=None, test_size=0.2, random_state=42, parametres_knn=None, parametres_knn_acp=None, parametres_lineaire=None, parametres_poly_lr=None, parametres_poly_lasso=None, parametres_poly_ridge=None, parametres_svr=None, parametres_gbr=None, parametres_rfr=None, parametres_mlp=None):
    """Comparaison des résultats des modèles de Machine Learning selon les hyperparamètres et les scores dans un affichage tabulaire avec Rich.
    
    modele_teste: Liste des modèles à tester. Par défaut, tous les modèles sont testés. Il y a 12 modèles possibles: "knn", "knn_acp", "lr", "lasso", "ridge", "lr_poly", "lasso_poly", "ridge_poly", "svr", "gbr", "rfr", "mlp".
    
    """
    
    def prep_donnees(df, test_size=0.2, random_state=42):
        """Préparation des données pour les modèles de Machine Learning. Séparation de la variable à expliquer ("PTS") et des variables explicatives.
        Puis division des données en données d'entrainement et de test.
        
        Paramètres: 
            df: Un DataFrame issue de la classe Donnees.
            test_size: La proportion des données de test.
            random_state: Le seed pour la reproductibilité des résultats.
            
        Retourne:
            X_train, X_test, y_train, y_test: Les données d'entrainement et de test.
            tuple: (X_train, X_test, y_train, y_test)
                - X_train: Les variables explicatives d'entrainement.
                - X_test: Les variables explicatives de test.
                - y_train: La variable à expliquer d'entrainement.
                - y_test: La variable à expliquer de test.
        """

        X=df.drop(columns="PTS")
        y=df["PTS"]

        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    
    
    
    param_grid_knn = parametres_knn if parametres_knn is not None else {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "p": [1, 2]}
    param_grid_knn_acp=parametres_knn_acp if parametres_knn_acp is not None else {"knn__n_neighbors": [3, 5, 7, 9], "knn__weights": ["uniform", "distance"], "knn__p": [1, 2], "pca__n_components": [2, 3, 4, 5]}
    param_grid_lr = parametres_lineaire if parametres_lineaire is not None else {}
    param_grid_lasso = parametres_lineaire if parametres_lineaire is not None else {"alpha": [0.001, 0.01, 0.1, 1, 10]}
    param_grid_ridge = parametres_lineaire if parametres_lineaire is not None else {"alpha": [0.001, 0.01, 0.1, 1, 10]}
    param_grid_poly_lr = parametres_poly_lr if parametres_poly_lr is not None else {"polynomialfeatures__degree": [1, 2, 3]}
    param_grid_poly_lasso = parametres_poly_lasso if parametres_poly_lasso is not None else {"lasso__alpha": [0.001, 0.01, 0.1, 1, 10], "polynomialfeatures__degree": [1, 2, 3]}
    param_grid_poly_ridge = parametres_poly_ridge if parametres_poly_ridge is not None else {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10], "polynomialfeatures__degree": [1, 2, 3]}
    param_grid_svr = parametres_svr if parametres_svr is not None else {"C": [0.1, 1, 10, 100], "epsilon": [0.1, 0.01, 0.001], "kernel": ["linear", "poly", "rbf"]}
    param_grid_grb=parametres_gbr if parametres_gbr is not None else {"n_estimators": [200, 300, 1000], "learning_rate": [0.1, 0.01, 0.001], "max_depth": [3, 5, 7, 9]}
    param_grid_rfr=parametres_rfr if parametres_rfr is not None else {"n_estimators": [200, 300, 400], "max_depth": [3, 5, 7, 9], "min_samples_split": [2, 3, 4]}
    param_grid_mlp=parametres_mlp if parametres_mlp is not None else {"hidden_layer_sizes": [(50,), (100,), (50,50)], "activation": ["relu","tanh", "logistic"], "solver": ["lbfgs", "adam"]}
    
    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=test_size, random_state=random_state)
    
    
    modeles_valides = ["knn", "knn_acp", "lr", "lasso", "ridge", 
                       "lr_poly", "lasso_poly", "ridge_poly", 
                       "svr", "gbr", "rfr", "mlp"]
    
    if modele_teste is not None:
        modeles_invalides = [modeles for modeles in modele_teste if modeles not in modeles_valides]
        if modeles_invalides:
            raise ValueError(f"Modèles invalides : {modeles_invalides}. Les modèles valides sont : {modeles_valides}.")
    
    modeles_etudies=[]
    
    if modele_teste is None or "knn" in modele_teste:
        modeles_etudies.append(("KNN", KNeighborsRegressor(), param_grid_knn))
    if modele_teste is None or "knn_acp" in modele_teste:
        modeles_etudies.append(("KNN avec ACP", Pipeline([("pca", PCA()), ("knn", KNeighborsRegressor())]), param_grid_knn_acp))
    if modele_teste is None or "lr" in modele_teste:
        modeles_etudies.append(("Régression linéaire", LinearRegression(), param_grid_lr))
    if modele_teste is None or "lasso" in modele_teste:
        modeles_etudies.append(("Lasso", Lasso(), param_grid_lasso))
    if modele_teste is None or "ridge" in modele_teste:
        modeles_etudies.append(("Ridge", Ridge(), param_grid_ridge))
    if modele_teste is None or "lr_poly" in modele_teste:
        modeles_etudies.append(("Régression polynomiale", make_pipeline(PolynomialFeatures(), LinearRegression()), param_grid_poly_lr))
    if modele_teste is None or "lasso_poly" in modele_teste:
        modeles_etudies.append(("Lasso polynomiale", make_pipeline(PolynomialFeatures(), Lasso()), param_grid_poly_lasso))
    if modele_teste is None or "ridge_poly" in modele_teste:
        modeles_etudies.append(("Ridge polynomiale", make_pipeline(PolynomialFeatures(), Ridge()), param_grid_poly_ridge))
    if modele_teste is None or "svr" in modele_teste:
        modeles_etudies.append(("SVR", SVR(), param_grid_svr))
    if modele_teste is None or "gbr" in modele_teste: 
        modeles_etudies.append(("Gradient-Boosting", GradientBoostingRegressor(), param_grid_grb))
    if modele_teste is None or "rfr" in modele_teste:
        modeles_etudies.append(("Random Forest", RandomForestRegressor(), param_grid_rfr))
    if modele_teste is None or "mlp" in modele_teste:
        modeles_etudies.append(("Réseau de neurones", MLPRegressor(), param_grid_mlp))

    console=Console()
    table = Table(title="Comparaison des résultats des modèles étudés", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Modèles", justify="left")
    table.add_column("Meilleurs paramètres", justify="center")
    table.add_column("Score données de Test (MSE)", justify="center")
    table.add_column("Score données de Test (R2)", justify="center")
    table.add_column("Meilleur modèle", justify="center")

    resultats = []
    
    for nom_du_modele, modele, param_grid in modeles_etudies:
        grid_search = GridSearchCV(estimator=modele, param_grid=param_grid, cv=6, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        meilleur_modele = grid_search.best_estimator_
        meilleurs_parametres = grid_search.best_params_
        meilleur_score = grid_search.best_score_
        
        y_pred=meilleur_modele.predict(X_test)
        test_mse=mean_squared_error(y_test, y_pred)
        test_R2=r2_score(y_test, y_pred)

        
        resultats.append([nom_du_modele, meilleurs_parametres, test_mse, test_R2, meilleur_modele])


        table.add_row(nom_du_modele, str(meilleurs_parametres), str(test_mse), str(test_R2), str(meilleur_modele))
        
        
    console.print(table)
    
    return resultats


#comparaison_resultats(df, modele_teste=None, test_size=0.2, random_state=42, parametres_knn=None, parametres_knn_acp=None, parametres_lineaire=None, parametres_poly_lr=None, parametres_poly_lasso=None, parametres_poly_ridge=None, parametres_svr=None, parametres_gbr=None, parametres_rfr=None, parametres_mlp=None)