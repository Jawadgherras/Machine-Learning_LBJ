import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

class ESPNWebScraper:
    """Classe pour scraper les données de LeBron James sur le site ESPN.com."""
    def __init__(self):
        """Attributs de la classe ESPNWebScraper."""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] 
        self.all_data = [] 
        self.headers_table = [] 

    def recuperation_donnees_annee(self, year):
        """Récupère les données pour une année donnée."""
        print(f"Récupération des données pour l'année {year}...")
        url = f"https://www.espn.com/nba/player/gamelog/_/id/1966/type/nba/year/{year}"

        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print(f"Erreur : statut {response.status_code} pour l'année {year}.")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        months = soup.find_all("div", class_="mb5")

        if not months:
            print(f"Aucune donnée trouvée pour l'année {year}.")
            return []

        return self.extraction_donnees_mois(months, year)

    def extraction_donnees_mois(self, months, year):
        """Extrait les données des blocs mensuels trouvés."""
        year_data = []
        
        for month in reversed(months):
            events_table = month.find("div", class_="events_table")
            if not events_table:
                print(f"Bloc 'mb5' sans tableau ignoré pour l'année {year}.")
                continue

            table = events_table.find("table", class_="Table")
            if not table:
                print(f"Tableau introuvable dans le bloc 'events_table' pour l'année {year}.")
                continue

            year_data.extend(self.extraction_lignes_tableau(table, year))

        return year_data

    def extraction_lignes_tableau(self, table, year):
        """Extrait les lignes d'un tableau HTML."""
        rows_data = []

        if len(self.all_data) == 0:
            self.headers_table = [th.text.strip() for th in table.find("thead").find_all("th")]
            print(f"En-têtes trouvés : {self.headers_table}")

        rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
        for row in reversed(rows):
            cols = row.find_all("td", class_="Table__TD")
            cols = [col.text.strip() for col in cols]

            if cols and not any(cols[0].startswith(day) for day in self.days_of_week):
                print(f"Ligne ignorée : {cols}")
                continue

            cols = self.processus_ligne(cols, year)
            if cols:
                rows_data.append(cols)

        return rows_data

    def processus_ligne(self, cols, year):
        """Traite et nettoie une ligne extraite."""
        if len(cols) > 1:
            if cols[1].startswith("@"):
                location = "Extérieur"
                cols[1] = cols[1][1:] 
            elif cols[1].startswith("vs"):
                location = "Domicile"
                cols[1] = cols[1][2:]  
            else:
                location = "Inconnue" 

            cols.append(location)  

        if cols:
            cols.insert(0, year)
        
        return cols

    def scraping_donnees(self, start_year, end_year):
        """Effectue le scraping des données pour une plage d'années."""
        for year in range(start_year, end_year + 1):
            year_data = self.recuperation_donnees_annee(year)
            self.all_data.extend(year_data)

        print(f"Scraping terminé. {len(self.all_data)} lignes de données récupérées.")

    def get_data(self):
        """Retourne toutes les données récupérées."""
        return self.all_data

    def get_headers(self):
        """Retourne les en-têtes des tableaux."""
        return self.headers_table


## Conversion des types de données
class Conversion_de_types:
    """Classe pour convertir les types de données du DataFrame."""
    def __init__(self, df):
        """Le DataFrame est passé au constructeur pour être stocké comme attribut de l'objet."""
        self.df=df
    
    def convertir_types(self):
        """Modifie les types de colonnes pour correspondre à ceux définis."""
        type_des_colonnes = {
            "Year": int,
            "Jour_de_sem": int,
            "Mois": int,
            "Jour": int,
            "OPP_ATL": int,
            "OPP_BKN": int,
            "OPP_BOS": int,
            "OPP_CHA": int,
            "OPP_CHI": int,
            "OPP_CLE": int,
            "OPP_DAL": int,
            "OPP_DEN": int,
            "OPP_DET": int,
            "OPP_GS": int,
            "OPP_HOU": int,
            "OPP_IND": int,
            "OPP_LAC": int,
            "OPP_LAL": int,
            "OPP_MEM": int,
            "OPP_MIA": int,
            "OPP_MIL": int,
            "OPP_MIN": int,
            "OPP_NO": int,
            "OPP_NY": int,
            "OPP_OKC": int,
            "OPP_ORL": int,
            "OPP_PHI": int,
            "OPP_PHX": int,
            "OPP_POR": int,
            "OPP_SA": int,
            "OPP_SAC": int,
            "OPP_TOR": int,
            "OPP_UTAH": int,
            "OPP_WSH": int,
            "OPP": str,
            "Location": int, 
            "Result": int,
            "PTS-OPP": float,    
            "MIN": float,
            "Reussis_FG": int,
            "Tentatives_FG": int,
            "FG%": float,
            "Reussis_3PT": int,
            "Tentatives_3PT": int,
            "3P%": float,
            "Reussis_FT": int,
            "Tentatives_FT": int,
            "FT%": float,
            "REB": int,
            "AST": int,
            "BLK": int,
            "STL": int,
            "PF": int,
            "TO": int,
            "PTS": float,
        }
        
        for colonne, type_colonne in type_des_colonnes.items():
            if colonne in self.df.columns:
                type_actuel = self.df[colonne].dtype
                if type_actuel != type_colonne:
                    print(f"Conversion de {colonne} de {type_actuel} en {type_colonne}")
                    self.df[colonne] = self.df[colonne].astype(type_colonne)
                        
    def get_dataframe(self):
        """Retourne le DataFrame modifié."""
        return self.df



# Classe pour l'initialisation du DataFrame
class Donnees:
    """Classe pour initialiser le DataFrame avec les données récupérées."""
    def __init__(self, start_year, end_year):
        """Initialisation de la classe : scraping des données, création du DataFrame et convertion des types."""
        
        self.scraper = ESPNWebScraper()  # Instance de la classe de scraping
        self.scraper.scraping_donnees(start_year, end_year)  # Scrape les données pour les années données
        self.scraper.get_headers().insert(0, "Year")  
        self.scraper.get_headers().append("Location")
        
        # Créer un DataFrame à partir des données scrappées
        self.df = pd.DataFrame(self.scraper.get_data(), columns=self.scraper.get_headers())
        
        # Renommer la première colonne en "YEAR"
        self.df.rename(columns={self.df.columns[0]: "Year"}, inplace=True)
        
        # Change le noms des équipes qui ont déménagé
        
        equipe_relocalisees = {"SEA": "OKC", "NJ" : "BKN"}
        
        self.df["OPP"] = self.df['OPP'].replace(equipe_relocalisees)
        
        # Modification de l'ordre des colonnes et ajout de colonnes
        self.df = self._ajuster_colonnes()
        
        # Supprimer les valeurs spécifiques dans la colonne "OPP"
        valeurs_a_supprimer=['WEST*','STE*','GIA*', 'DUR*', 'EAST*']
        self.supprimer_valeurs_opp(valeurs_a_supprimer) 
        
        # Modifier les colonnes et ajouter les nouvelles colonnes
        self._modifier_colonnes()
        
        # Encoder les valeurs de la colonne "Location" et "Result" en binaire
        self._encoder_resultat_localisation()
        
        # Conversion des types de données
        self.conversion = Conversion_de_types(self.df)
        self.conversion.convertir_types()

    def _ajuster_colonnes(self):
        """Ajoute et ajuste l'ordre des colonnes du DataFrame"""
        if "Location" not in self.df.columns:
            self.df["Location"] = None  # Si la colonne Location n'existe pas, on l'ajoute avec une valeur nulle
        
        # Placer la colonne "Location" en 3ème position
        self.df = self.deplace_colonne(self.df, 'Location', 3)
        return self.df

    def _modifier_colonnes(self):
        """Modifie et ajoute des colonnes spécifiques au DataFrame."""
        
        # Ajouter la colonne "PTS OPP" comme copie de la colonne "Result" avant modification
        #self.df['PTS-OPP'] = self.df['Result'].apply(lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) > 1 else None)
        self.df['PTS-OPP'] = self.df["Result"].apply(self.extraire_score_adversaire, axis=1)
        
        
        # Modifier la colonne "Result" pour ne conserver que "W" ou "L"
        self.df['Result'] = self.df['Result'].str.extract(r'^(W|L)')
        
        # Placer la colonne "PTS OPP" juste après "Result"
        self.df = self.deplace_colonne(self.df, 'PTS-OPP', self.df.columns.get_loc('Result') + 1)
        
        # Conversion de la colonnes Date en datetime
        self.df["Date"]=pd.to_datetime(self.df["Year"].astype(str) + " " + self.df["Date"], format="%Y %a %m/%d")
        
        # Création des nouvelles colonnes "Jour_de_sem", "Mois", et "Jour"
        self.df['Jour_de_sem'] = self.df['Date'].dt.weekday  # Lundi=0, Mardi=2, Mercredi=3, ... , Dimanche=6
        self.df['Mois'] = self.df['Date'].dt.month  # Mois 1-12, Jan=1, Feb=2, ... , Dec=12
        self.df['Jour'] = self.df['Date'].dt.day  # Jour du mois, de 1 à 31
        
        #Supprimer la colonne "Date"
        self.df.drop(columns=["Date"], inplace=True)
        
        # Placer les colonnes "Jour_de_sem", "Mois", et "Jour" après la colonne "Year"
        self.df = self.deplace_colonne(self.df, 'Jour_de_sem', self.df.columns.get_loc('Year') + 1)  
        self.df = self.deplace_colonne(self.df, 'Mois', self.df.columns.get_loc('Year') + 2) 
        self.df = self.deplace_colonne(self.df, 'Jour', self.df.columns.get_loc('Year') + 3) 
        
        # Appliquer One-Hot Encoding sur la colonne 'OPP' pour créer les nouvelles colonnes
        self.df = pd.get_dummies(self.df, columns=["OPP"], prefix="OPP", drop_first=False)
        
        # Déplacer les colonnes One-Hot Encoding après la colonne "Jour"
        position_jour = self.df.columns.get_loc("Jour") + 1  
        for column in self.df.columns:
            if column.startswith("OPP_"):
                self.df = self.deplace_colonne(self.df, column, position_jour)
                position_jour += 1  
                
        # Extraction des tirs réussis et tentatives pour les colonnes FG, 3PT, et FT
        # "FG" : séparer les tirs au panier réussis et tentés
        self.df['Reussis_FG'], self.df['Tentatives_FG'] = zip(*self.df['FG'].apply(self.extraction_tirs))
        
        # "3PT" 
        self.df['Reussis_3PT'], self.df['Tentatives_3PT'] = zip(*self.df['3PT'].apply(self.extraction_tirs))
        
        # "FT" : séparer les lancers francs réussis et tentés
        self.df['Reussis_FT'], self.df['Tentatives_FT'] = zip(*self.df['FT'].apply(self.extraction_tirs))
        
        # Supprime les anciennes colonnes "FG", "FT", "3PT"
        self.df.drop(['FG', 'FT', '3PT'], axis=1, inplace=True)
        
        # Place les nouvelles colonnes 
        self.df = self.deplace_colonne(self.df, 'Reussis_FG', self.df.columns.get_loc('MIN') + 1)
        self.df = self.deplace_colonne(self.df, 'Tentatives_FG', self.df.columns.get_loc('MIN') + 2)
        self.df = self.deplace_colonne(self.df, 'Reussis_3PT', self.df.columns.get_loc('FG%') + 1)
        self.df = self.deplace_colonne(self.df, 'Tentatives_3PT', self.df.columns.get_loc('FG%') + 2)
        self.df = self.deplace_colonne(self.df, 'Reussis_FT', self.df.columns.get_loc('3P%') + 1)
        self.df = self.deplace_colonne(self.df, 'Tentatives_FT', self.df.columns.get_loc('3P%') + 2)

    def _encoder_resultat_localisation(self):
        """Encode les valeurs de la colonne 'Location' & 'Result' en binaire."""
        
        # Encoder les valeurs de la colonne "Location" en binaire
        self.df['Location'] = self.df['Location'].map({"Domicile": 1, "Extérieur": 0})
        
        # Encoder les valeurs de la colonne "Result" en binaire
        self.df['Result'] = self.df['Result'].map({"W": 1, "L": 0})


    def extraire_score_adversaire(self, result, axis=None):
        """Extraire le score adverse à partir du résultat, en tenant compte de la victoire ou défaite."""
        if isinstance(result, str):
            if result.startswith('W'):
                match = re.search(r'W(\d+)-(\d+)', result)  
                if match:
                    team_score, opponent_score = map(int, match.groups())
                    return opponent_score  
            elif result.startswith('L'):
                match = re.search(r'L(\d+)-(\d+)', result) 
                if match:
                    team_score, opponent_score = map(int, match.groups())
                    return team_score  
        print(f"Format inattendu dans la colonne 'Result': {result}")  
        return None  

    def extraction_tirs(self, x):
        """Extrait les réussis et les tentatives à partir du format des tirs "TT-TR"."""
        if isinstance(x, str):
            try:
                reussis, tentatives = map(int, x.split('-'))
                return reussis, tentatives
            except ValueError:
                return 0, 0
        return 0, 0

    def deplace_colonne(self, df, nom_colonne, nouvelle_position):
            """Déplace une colonne à une nouvelle position."""
            columns = list(df.columns)
            columns.insert(nouvelle_position, columns.pop(columns.index(nom_colonne)))
            return df[columns]

    def supprimer_valeurs_opp(self, valeurs_a_supprimer):
        """
        Supprime les lignes où la colonne 'OPP' contient certaines valeurs spécifiques.
        :param valeurs_a_supprimer: Liste des valeurs à supprimer de la colonne 'OPP'.
        """
        
        self.df = self.df[~self.df["OPP"].isin(valeurs_a_supprimer)]

    def get_dataframe(self):
        """Retourne le DataFrame final prêt à l'emploi."""
        return self.df

    def exportation_CSV(self, nom_fichier):
        """Exporte le DataFrame en fichier CSV."""
        self.df.to_csv(nom_fichier, index=False)
        print(f"Les données ont été exportées dans {nom_fichier}")

def initialisation_donnees(start_year, end_year):
    """Initialisation des données dans un DataFrame."""
    donnees = Donnees(start_year, end_year)
    df =donnees.get_dataframe()
    return df