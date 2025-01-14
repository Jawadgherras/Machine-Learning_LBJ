from maintest import ESPNWebScraper, Conversion_de_types, Donnees, plot_localisation_plotly, plot_points_moyens_no_text, plot_points_somme_plotly, victoires_par_annee, points_moyens_par_equipe, pourcentage_reussite_par_annee, points_par_equipe_adverse, somme_des_points_encaissés_par_equipes, moyennes_des_points_encaissés_par_equipes, distribution_points
from ML_test import *
from unittest.mock import patch, Mock
import pytest
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

#Test la classe ESPNWebScraper()

def test_initialisation_classe():
    scraper = ESPNWebScraper()
    assert scraper.headers["User-Agent"].startswith("Mozilla")
    assert len(scraper.days_of_week) == 7
    assert scraper.all_data == []
    assert scraper.headers_table == []


url = "https://www.espn.com/nba/player/gamelog/_/id/1966/type/nba/year/2023"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    with open("mock_espn_page.html", "w", encoding="utf-8") as f:
        f.write(response.text)  
    print("Le fichier a été sauvegardé avec succès.")
else:
    print(f"Erreur lors de la récupération de la page. Code de statut : {response.status_code}")


@patch("requests.get")
def test_recuperation_donnees_annee(mock_get):
    scraper = ESPNWebScraper()

    mock_response = MagicMock()
    mock_response.status_code = 200
    with open("mock_espn_page.html", "r") as f:
        mock_response.content = f.read()
    mock_get.return_value = mock_response

    data = scraper.recuperation_donnees_annee(2023)
    assert len(data) > 0  
    mock_get.assert_called_with(
        "https://www.espn.com/nba/player/gamelog/_/id/1966/type/nba/year/2023",
        headers=scraper.headers,
    )


#def test_extraction_lignes_tableau():
 #   scraper = ESPNWebScraper()

  #  with open("mock_table_data.html", "r", encoding="utf-8") as f:
   #     soup = BeautifulSoup(f.read(), "html.parser")
    #    table = soup.find("table", class_="Table")  

    #if table:
     #   data = scraper.extraction_lignes_tableau(table, 2023)
      #  assert len(data) > 0  
    #else:
     #   print("Tableau non trouvé dans le fichier HTML.")


def test_recuperation_donnees_annee_status_code():
    """Test que la méthode retourne une liste vide de données si le status code est différent de 200"""
    scraper = ESPNWebScraper()
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = scraper.recuperation_donnees_annee(2023)
        assert result == []

def test_recuperation_donnees_annee_mauvaise_structure_html():
    """Test que la méthode retourne une liste vide de données si le contenu de la page n'est pas une structure HTML"""
    scraper = ESPNWebScraper()
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html></html>"
        mock_get.return_value = mock_response

        result = scraper.recuperation_donnees_annee(2023)
        assert result == []

def test_extraction_donnees_mois_vide():
    """Test que la méthode retourne une liste vide de données si la liste de mois est vide et que le scaping se fait dans les divs :<div class='mb5'></div>"""
    scraper = ESPNWebScraper()
    soup = BeautifulSoup("<div class='mb5'></div>", "html.parser")
    mois = soup.find_all("div", class_="mb5")
    result = scraper.extraction_donnees_mois(mois, 2023)
    assert result == []



def test_processus_ligne():
    scraper = ESPNWebScraper()
    ligne = ["Wed", "@TeamX", "32", "10-15", "5-7", "7-9", "10", "5", "1", "2", "3", "1", "32"]
    result = scraper.processus_ligne(ligne, 2023)
    assert result[-1] == "Extérieur"


def test_donnees_entetes():
    scraper = ESPNWebScraper()
    scraper.all_data = [["2023", "Wed", "@TeamX", "Extérieur"]]
    scraper.headers_table = ["Year", "Day", "Opponent", "Location"]

    assert scraper.get_data() == [["2023", "Wed", "@TeamX", "Extérieur"]]
    assert scraper.get_headers() == ["Year", "Day", "Opponent", "Location"]


def test_get_data():
    scraper = ESPNWebScraper()
    scraper.all_data = [["2023", "vs Opponent", "30", "8", "3", "5"]]
    
    data = scraper.get_data()
    assert len(data) > 0


def test_get_headers():
    scraper = ESPNWebScraper()
    scraper.headers_table = ["Year", "Opponent", "PTS", "REB"]

    headers = scraper.get_headers()
    assert len(headers) > 0 
    assert headers[0] == "Year"



# Test de la classe Conversion_de_types()
from pandas.testing import assert_frame_equal

import pandas as pd
import pytest

@pytest.fixture
def dataframe_test():
    """Fixture pour créer un DataFrame pour le test de la classe Conversion_de_types."""
    data = {
        "Year": [2021, 2022],
        "Jour_de_sem": [0, 1],
        "Mois": [1, 2],
        "Jour": [1, 2],
        "OPP_ATL": [1, 0],
        "Location": ["1", "0"], 
        "Result": ["1", "0"],  
        "PTS-OPP": ["110.5", "105.7"],  
        "MIN": ["38.5", "36.2"],  
        "FG%": [50.2, 47.8],
        "PTS": [30, 25]
    }
    return pd.DataFrame(data)

def test_conversion_des_types(dataframe_test):
    """Teste que les types des colonnes sont correctement convertis."""
    conversion = Conversion_de_types(dataframe_test)
    conversion.convertir_types()
    df_converti = conversion.get_dataframe()

    types_attendus = {
        "Year": int,
        "Jour_de_sem": int,
        "Mois": int,
        "Jour": int,
        "OPP_ATL": int,
        "Location": int,
        "Result": int,
        "PTS-OPP": float,
        "MIN": float,
        "FG%": float,
        "PTS": float
    }


    for colonne, type_attendu in types_attendus.items():
        assert df_converti[colonne].dtype == type_attendu

    assert df_converti["Location"].iloc[0] == 1
    assert df_converti["PTS-OPP"].iloc[0] == 110.5
    assert df_converti["MIN"].iloc[1] == 36.2



# Test de la classe Donnees()
from unittest.mock import MagicMock

def test_modifier_colonnes():
    
    df_mock = pd.DataFrame({
        "Year": [2023, 2023],
        "Date": ["Tue 01/10", "Wed 01/11"],
        "OPP": ["vs BOS", "@ MIA"],
        "Result": ["W110-100", "L98-96"],
        "Location": ["Domicile", "Extérieur"],
        "MIN": ["35", "38"],
        "FG": ["10-15", "9-20"],
        "FG%": ["66.7", "45.0"],
        "3PT": ["3-5", "1-4"],
        "3P%": ["60.0", "25.0"],
        "FT": ["7-8", "4-5"],
        "FT%": ["87.5", "80.0"],
        "REB": ["10", "8"],
        "AST": ["8", "7"],
        "BLK": ["2", "1"],
        "STL": ["1", "2"],
        "PF": ["4", "3"],
        "TO": ["3", "5"],
        "PTS": ["30", "23"]
    })

    donnees = Donnees(2023, 2023)
    donnees.df = df_mock  
    donnees._modifier_colonnes()  

    
    assert "PTS-OPP" in donnees.df.columns
    assert donnees.df.loc[0, "PTS-OPP"] == 100
    assert donnees.df.loc[1, "PTS-OPP"] == 98

    assert "Location" in donnees.df.columns
    assert donnees.df.loc[0, "Location"] == 1
    assert donnees.df.loc[1, "Location"] == 0
    
    assert "Result" in donnees.df.columns
    assert donnees.df.loc[0, "Result"] == 1
    assert donnees.df.loc[1, "Result"] == 0
    
    assert "Tentatives_3PT" in donnees.df.columns
    assert donnees.df.loc[0, "Tentatives_3PT"] == 5
    assert donnees.df.loc[1, "Tentatives_3PT"] == 4
    
    assert "Reussis_FG" in donnees.df.columns
    assert donnees.df.loc[0, "Reussis_FG"] == 10
    assert donnees.df.loc[1, "Reussis_FG"] == 9
    
    assert "Reussis_FT" in donnees.df.columns
    assert donnees.df.loc[0, "Reussis_FT"] == 7
    assert donnees.df.loc[1, "Reussis_FT"] == 4
    
    assert "Jour_de_sem" in donnees.df.columns
    assert donnees.df.loc[0, "Jour_de_sem"] == 1
    assert donnees.df.loc[1, "Jour_de_sem"] == 2
    
    assert "Mois" in donnees.df.columns
    assert donnees.df.loc[0, "Mois"] == 1
    assert donnees.df.loc[1, "Mois"] == 1
    
    assert "Jour" in donnees.df.columns
    assert donnees.df.loc[0, "Jour"] == 10
    assert donnees.df.loc[1, "Jour"] == 11


def test_deplace_colonne():
    df_mock = pd.DataFrame({
        "Year": [2023, 2023],
        "Date": ["Tue 01/10", "Wed 01/11"],
        "OPP": ["vs BOS", "@ MIA"],
        "Result": ["W110-100", "L98-96"],
        "Location": ["Domicile", "Extérieur"],
        "MIN": ["35", "38"],
        "FG": ["10-15", "9-20"],
        "FG%": ["66.7", "45.0"],
        "3PT": ["3-5", "1-4"],
        "3P%": ["60.0", "25.0"],
        "FT": ["7-8", "4-5"],
        "FT%": ["87.5", "80.0"],
        "REB": ["10", "8"],
        "AST": ["8", "7"],
        "BLK": ["2", "1"],
        "STL": ["1", "2"],
        "PF": ["4", "3"],
        "TO": ["3", "5"],
        "PTS": ["30", "23"]
    })

    donnees = Donnees(2023, 2023)
    donnees.df = df_mock  
    
    assert donnees.df.columns[0] == "Year"
    assert donnees.df.columns[1] == "Date"
    assert donnees.df.columns[2] == "OPP"
    assert donnees.df.columns[3] == "Result"
    assert donnees.df.columns[4] == "Location"
    
    
    donnees.df=donnees.deplace_colonne(donnees.df, "Location", donnees.df.columns.get_loc("Result"))
    
    assert donnees.df.columns[3] == "Location"
    assert donnees.df.columns[4] == "Result"


def test_supprimer_valeurs_opp():
    df_mock = pd.DataFrame({
        "Year": [2023, 2023, 2023],
        "Date": ["Tue 01/10", "Wed 01/11", "Thu 01/12"],
        "OPP": ["vs BOS", "@ MIA", "WEST*"],
        "Result": ["W110-100", "L98-96", "W100-90"],
        "Location": ["Domicile", "Extérieur", "Domicile"],
        "MIN": ["35", "38", "40"],
        "FG": ["10-15", "9-20", "12-20"],
        "FG%": ["66.7", "45.0", "60.0"],
        "3PT": ["3-5", "1-4", "2-5"],
        "3P%": ["60.0", "25.0", "40.0"],
        "FT": ["7-8", "4-5", "4-5"],
        "FT%": ["87.5", "80.0", "80.0"],
        "REB": ["10", "8", "9"],
        "AST": ["8", "7", "6"],
        "BLK": ["2", "1", "1"],
        "STL": ["1", "2", "2"],
        "PF": ["4", "3", "3"],
        "TO": ["3", "5", "4"],
        "PTS": ["30", "23", "28"]
    })
    
    donnees = Donnees(2023, 2023)
    donnees.df = df_mock
    
    
    assert len(donnees.df) == 3
    
    
    ligne_à_supprimer = ["WEST*"]
    
    donnees.supprimer_valeurs_opp(ligne_à_supprimer)
    
    assert len(donnees.df) == 2
    
    assert "vs BOS" in donnees.df["OPP"].values
    assert "@ MIA" in donnees.df["OPP"].values
    assert "WEST*" not in donnees.df["OPP"].values


def test_get_dataframe():
    df_mock = pd.DataFrame({
        "Year": [2023, 2023],
        "Date": ["Tue 01/10", "Wed 01/11"],
        "OPP": ["vs BOS", "@ MIA"],
        "Result": ["W110-100", "L98-96"],
        "Location": ["Domicile", "Extérieur"],
        "MIN": ["35", "38"],
        "FG": ["10-15", "9-20"],
        "FG%": ["66.7", "45.0"],
        "3PT": ["3-5", "1-4"],
        "3P%": ["60.0", "25.0"],
        "FT": ["7-8", "4-5"],
        "FT%": ["87.5", "80.0"],
        "REB": ["10", "8"],
        "AST": ["8", "7"],
        "BLK": ["2", "1"],
        "STL": ["1", "2"],
        "PF": ["4", "3"],
        "TO": ["3", "5"],
        "PTS": ["30", "23"]
    })
    
    donnees = Donnees(2023, 2023)
    donnees.df = df_mock
    
    assert donnees.get_dataframe().equals(df_mock)



def test_exportation_csv(tmp_path):
    
    df_mock = pd.DataFrame({"Year": [2023, 2023], "Result": ["W110-100", "L95-100"]})

    donnees = Donnees(2023, 2023)
    donnees.df = df_mock

    file_path = tmp_path / "test_output.csv"
    donnees.exportation_CSV(file_path)

    df_exporte = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(df_mock, df_exporte)


# Test modèles de ML
from sklearn.model_selection import train_test_split 
from ML_test import prep_donnees, knn_avec_ou_sans_ACP

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



#Test de la fonction prep_donnees
@pytest.fixture
def donnees_mock():
    """Création de données fictives pour les tests. Les données doivent contenir forcément une colonne 'PTS'."""
    data = {
        "Feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "PTS": [15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    }
    return pd.DataFrame(data)



def test_prep_donnees_sorties(donnees_mock):
    """Test des sorties de la fonction prep_donnees. Vérifie les les proportions dans les échantillons d'entrainement et de test. Mais aussi la présence de la colonne 'PTS'."""
    df = donnees_mock
    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=0.2, random_state=42)
    
    
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    assert "PTS" not in X_train.columns, "X_train ne devrait pas contenir la colonne 'PTS'."
    assert "PTS" not in X_test.columns, "X_test ne devrait pas contenir la colonne 'PTS'."


def test_prep_donnees_reproductibilite(donnees_mock):
    """Test de la reproductibilité avec le même random_state. Vérifie que les résultats sont identiques grace à la graine."""
    df = donnees_mock
    result1 = prep_donnees(df, test_size=0.2, random_state=42)
    result2 = prep_donnees(df, test_size=0.2, random_state=42)
    
    
    for r1, r2 in zip(result1, result2):
        pd.testing.assert_frame_equal(r1, r2) if isinstance(r1, pd.DataFrame) else pd.testing.assert_series_equal(r1, r2)

def test_prep_donnees_changement_test_size(donnees_mock):
    """Test de la modification du paramètre test_size. Pour vérifier que la proportion change correctement lorsque test_size change."""
    df = donnees_mock
    X_train, X_test, y_train, y_test = prep_donnees(df, test_size=0.3, random_state=42)
    
    assert len(X_train) == 7
    assert len(X_test) == 3


# Test de la fonction knn_avec_ou_sans_ACP()
def test_knn_sans_acp(donnees_mock):
    """Test de la fonction avec ACP=False."""
    df = donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele = knn_avec_ou_sans_ACP(df, ACP=False)
    
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert isinstance(test_R2, float)
    assert test_R2>=0
    assert test_mse>=0

def test_knn_avec_acp(donnees_mock):
    """Test de la fonction avec ACP=True. Vérifie que les meilleurs paramètres, le meilleur score et le MSE de test sont bien retournés."""
    df = donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = knn_avec_ou_sans_ACP(df, ACP=True)
    
    assert isinstance(meilleurs_parametres, dict)
    assert "pca__n_components" in meilleurs_parametres
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert isinstance(test_R2, float)
    assert test_R2>=0

def test_erreur_parametres_invalides(donnees_mock):
    """Test pour des paramètres invalides. Ici n_neighbors est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        knn_avec_ou_sans_ACP(df, ACP=False, n_neighbors=["invalid"])

# Test de la fonction regressions_lineaires()

def test_regressions_lineaire(donnees_mock):
    """Test de la fonction regressions_lineaires. Vérifie que les meilleurs paramètres, le meilleur score et le MSE de test sont bien retournés."""
    df=donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = regressions_lineaires(df, type_de_modele="lineaire", test_size=0.2, random_state=42)
    
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert test_mse>=0
    assert isinstance(test_R2, float)
    assert test_R2>=0
    
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = regressions_lineaires(df, type_de_modele="lasso", alpha=[0.001, 0.01, 0.1, 1, 10], test_size=0.2, random_state=42)
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert test_mse>=0
    assert isinstance(test_R2, float)
    assert test_R2>=0
    
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = regressions_lineaires(df, type_de_modele="ridge", alpha=[0.001, 0.01, 0.1, 1, 10], test_size=0.2, random_state=42)
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert test_mse>=0
    assert isinstance(test_R2, float)
    assert test_R2>=0


def test_regressions_lineaire_parametres_invalides_lasso(donnees_mock):
    """Test pour des paramètres invalides. Ici alpha est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        regressions_lineaires(df, type_de_modele="lasso", alpha=["invalide"], test_size=0.2, random_state=42)


def test_regressions_lineaire_parametres_invalides_ridge(donnees_mock):
    """Test pour des paramètres invalides. Ici alpha est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        regressions_lineaires(df, type_de_modele="ridge", alpha=["invalide"], test_size=0.2, random_state=42)

#Test de la fonction regression_polynomiales()
def test_regression_polynomiales(donnees_mock):
    """Test de la fonction regression_polynomiales. Vérifie que les meilleurs paramètres, le meilleur score et le MSE de test sont bien retournés."""
    df=donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = regression_polynomiales(df, type_de_modele="lineaire", degres=[2, 3, 4], test_size=0.2, random_state=42)
    
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert "polynomialfeatures__degree" in meilleurs_parametres
    assert test_mse>=0
    assert isinstance(test_R2, float)
    assert test_R2>=0
    
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = regression_polynomiales(df, type_de_modele="lasso", degres=[2, 3, 4], alpha=[0.001, 0.01, 0.1, 1, 10], test_size=0.2, random_state=42)
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert "polynomialfeatures__degree" in meilleurs_parametres
    assert "lasso__alpha" in meilleurs_parametres
    assert test_mse>=0
    assert isinstance(test_R2, float)
    assert test_R2>=0
    
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = regression_polynomiales(df, type_de_modele="ridge",degres=[2, 3, 4], alpha=[0.001, 0.01, 0.1, 1, 10], test_size=0.2, random_state=42)
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert "polynomialfeatures__degree" in meilleurs_parametres
    assert "ridge__alpha" in meilleurs_parametres
    assert test_mse>=0
    assert meilleur_score<=0
    assert isinstance(test_R2, float)
    assert test_R2>=0

def test_regression_polynomiales_parametres_invalides(donnees_mock):
    """Test pour des paramètres invalides. Ici degres est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        regression_polynomiales(df, type_de_modele="lineaire", degres=["invalide"], test_size=0.2, random_state=42)

def test_regression_polynomiales_parametres_invalides_lasso(donnees_mock):
    """Test pour des paramètres invalides. Ici alpha est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        regression_polynomiales(df, type_de_modele="lasso", degres=[2, 3, 4], alpha=["invalide"], test_size=0.2, random_state=42)

def test_regression_polynomiales_parametres_invalides_ridge(donnees_mock):
    """Test pour des paramètres invalides. Ici alpha est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        regression_polynomiales(df, type_de_modele="ridge", degres=[2, 3, 4], alpha=["invalide"], test_size=0.2, random_state=42)

#Test de la fonction support_vector_regression()
def test_support_vector_regression(donnees_mock):
    """Test de la fonction support_vector_regression. Vérifie que les meilleurs paramètres, le meilleur score et le MSE de test sont bien retournés."""
    df=donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = support_vector_regression(df, C=[0.1, 1, 10, 100], epsilon=[0.1, 0.01, 0.001], kernel=["linear", "poly", "rbf"], test_size=0.2, random_state=42)
    
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert "C" in meilleurs_parametres
    assert "epsilon" in meilleurs_parametres
    assert "kernel" in meilleurs_parametres
    assert test_mse>=0
    assert meilleur_score<=0
    assert isinstance(test_R2, float)
    assert test_R2>=0

def test_support_vector_regression_parametres_invalides(donnees_mock):
    """Test pour des paramètres invalides. Ici C est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        support_vector_regression(df, C=["invalide"], epsilon=[0.1, 0.01, 0.001], kernel=["linear", "poly", "rbf"], test_size=0.2, random_state=42)

#Test de la fonction gradeint_boosting()

def test_gradient_boosting(donnees_mock):
    """Test de la fonction gradient_boosting. Vérifie que les meilleurs paramètres, le meilleur score et le MSE de test sont bien retournés."""
    df=donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = gradient_boosting(df, nbr_estimateurs=[200, 300, 1000], taux_apprentissage=[0.1, 0.01, 0.001], profondeur_max=[3, 5, 7, 9], test_size=0.2, random_state=42)
    
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert "n_estimators" in meilleurs_parametres
    assert "learning_rate" in meilleurs_parametres
    assert "max_depth" in meilleurs_parametres
    assert test_mse>=0
    assert isinstance(test_R2, float)
    assert test_R2>=0

def test_gradient_boosting_parametres_invalides(donnees_mock):
    """Test pour des paramètres invalides. Ici nbr_estimateurs est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        gradient_boosting(df, nbr_estimateurs=["invalide"], taux_apprentissage=[0.1, 0.01, 0.001], profondeur_max=["invalide", 5, 7, 9], test_size=0.2, random_state=42)

#Test de la fonction random_forest()

def test_random_forest(donnees_mock):
    """Test de la fonction random_forest. Vérifie que les meilleurs paramètres, le meilleur score et le MSE de test sont bien retournés."""
    df=donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = random_forest(df,n_estimators=[200, 300, 400], max_depth=[3, 5, 7, 9], min_samples_split=[2, 3, 4], test_size=0.2, random_state=42)
    
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert "n_estimators" in meilleurs_parametres
    assert "max_depth" in meilleurs_parametres
    assert "min_samples_split" in meilleurs_parametres
    assert test_mse>=0
    assert meilleur_score<=0
    assert isinstance(test_R2, float)
    assert test_R2>=0

def test_random_forest_parametres_invalides(donnees_mock):
    """Test pour des paramètres invalides. Ici n_estimators est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        random_forest(df, n_estimators=["invalide"], max_depth=[3, 5, 7, 9], min_samples_split=["Invalide", 3, 4], test_size=0.2, random_state=42)

#Test de la fonction reseaux_neurones()
def test_reseaux_neurones(donnees_mock):
    """Test de la fonction reseaux_neurones. Vérifie que les meilleurs paramètres, le meilleur score et le MSE de test sont bien retournés."""
    df=donnees_mock
    meilleurs_parametres, meilleur_score, test_mse, test_R2, meilleur_modele  = reseaux_neurones(df, hidden_layer_sizes=[(50,), (100,), (50,50)], fct_activation=["relu","tanh", "logistic"], solver=["lbfgs", "adam"], test_size=0.2, random_state=42)
    
    assert isinstance(meilleurs_parametres, dict)
    assert isinstance(meilleur_score, float)
    assert isinstance(test_mse, float)
    assert "hidden_layer_sizes" in meilleurs_parametres
    assert "activation" in meilleurs_parametres
    assert "solver" in meilleurs_parametres
    assert test_mse>=0
    assert meilleur_score<=0
    assert isinstance(test_R2, float)
    assert test_R2>=0

def test_reseaux_neurones_parametres_invalides(donnees_mock):
    """Test pour des paramètres invalides. Ici hidden_layer_sizes est une liste de str."""
    df = donnees_mock
    with pytest.raises(ValueError):
        reseaux_neurones(df, hidden_layer_sizes=["invalide"], fct_activation=["relu","tanh", "logistic"], solver=["lbfgs", "adam"], test_size=0.2, random_state=42)



#Test de la fonction de comparaison des résultats des modèles, comparaison_resultats()
def test_comparaison_resultats(donnees_mock):
    """Test de la fonction comparaison_resultats. Vérifie que les résultats sont bien retournés."""
    df=donnees_mock
    
    resultats = comparaison_resultats(df, modele_teste=["knn", "mlp"])
    
    assert len(resultats) == 2
    assert resultats[0][0] == "KNN"
    assert resultats[1][0] == "Réseau de neurones"
    assert isinstance(resultats[0][0], str)
    assert isinstance(resultats[1][0], str)
    assert isinstance(resultats[0][1], dict)
    assert isinstance(resultats[1][1], dict)
    assert isinstance(resultats[0][2], float)
    assert isinstance(resultats[1][2], float)
    assert isinstance(resultats[0][3], float)
    assert isinstance(resultats[1][3], float)
    assert resultats[0][3]>=0
    assert resultats[0][2]>=0
    assert resultats[1][2]>=0
    assert resultats[1][3]>=0

def test_comparaison_resultats_modele_invalide(donnees_mock):
    """Test pour un modèle invalide. Ici modele_teste contient un modèle non existant."""
    df = donnees_mock
    with pytest.raises(ValueError, match="Modèles invalides : .*"):
        comparaison_resultats(df, modele_teste=["invalide"])