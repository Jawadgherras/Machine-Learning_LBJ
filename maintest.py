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
        
        
        # Modifier la colonne "Result" pour ne conserver que "W" ou "L" puis en 1 ou 0
        self.df['Result'] = self.df['Result'].str.extract(r'^(W|L)')
        self.df['Result'] = self.df['Result'].map({"W": 1, "L": 0})
        
        # Encode les valeurs de la colonne "Location" en binaire
        self.df['Location'] = self.df['Location'].map({"Domicile": 1, "Extérieur": 0})
        
        # Placer la colonne "PTS OPP" juste après "Result"
        self.df = self.deplace_colonne(self.df, 'PTS-OPP', self.df.columns.get_loc('Result') + 1)
        
        # Conversion de la colonnes Date en datetime
        self.df["Date"]=pd.to_datetime(self.df["Year"].astype(str) + " " + self.df["Date"], format="%Y %a %m/%d")
        
        # Création des nouvelles colonnes "Jour_de_sem", "Mois", et "Jour"
        self.df['Jour_de_sem'] = self.df['Date'].dt.weekday  # Lundi=0, Mardi=2, Mercredi=3, ... , Dimanche=6
        self.df['Mois'] = self.df['Date'].dt.month  # Mois 1-12, Jan=1, Fev=2, ... , Dec=12
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



from plotly import graph_objects as go
def plot_localisation_plotly(df):
    
    """Graphique en barres des résultats des matchs de LeBron James par localisation à l'aide de la librairie Plotly"""
    
    # Créer un tableau avec le nombre de matchs gagnés et perdus par localisation
    tableau_victoire = df[df['Result'] == 1].groupby('Location').size().reset_index(name='Matchs_gagnés')
    tableau_defaite = df[df['Result'] == 0].groupby('Location').size().reset_index(name='Matchs_perdus')
    tableau_final = pd.merge(tableau_victoire, tableau_defaite, on='Location', how="outer")
    
    fig = go.Figure()


    fig.add_trace(go.Bar(
        x=tableau_final['Location'],
        y=tableau_final['Matchs_gagnés'],
        name='Victoires',
        marker=dict(color='lightgreen', line=dict(width=0))
    ))


    fig.add_trace(go.Bar(
        x=tableau_final['Location'],
        y=tableau_final['Matchs_perdus'],
        name='Défaites',
        marker=dict(color='tomato', line=dict(width=0))
    ))


    fig.update_layout(
        title="Résultats des matchs de LeBron James par localisation",
        xaxis_title="Localisation",
        yaxis_title="Nombre de matchs",
        barmode='stack',  
        template="plotly_white",
        xaxis=dict(tickangle=-45, gridcolor=None, showgrid=True, tickmode='array', tickvals=[0, 1], ticktext=["Extérieur", "Domicile"]),
        yaxis=dict(gridcolor=None, showgrid=True),
        margin=dict(l=50, r=50, t=50, b=50),
        height=600,
        bargap=0.1,
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5", 
    )
    fig.show()


"""Graphiques interactifs avec Plotly pour les statistiques descriptives de LeBron James"""
import plotly.graph_objects as go
import pandas as pd


def plot_points_moyens_no_text(df):
    """Diagramme en barres des moyennes de points par match de LeBron James par saison à l'aide de la librairie Plotly"""
    
    df_moyenne = df.groupby('Year')['PTS'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_moyenne['Year'],
        y=df_moyenne['PTS'],
        marker=dict(
            color=df_moyenne['PTS'], 
            colorscale='Cividis',
            colorbar=dict(title="Moyenne des points marqués"),  
        ), text=df_moyenne["PTS"].round(1), textposition="outside"
    ))

    fig.update_layout(
        title="Moyenne de points par match de LeBron James par saison",
        xaxis_title="Année",
        yaxis_title="Moyenne de points par match",
        template="plotly_white",
        xaxis=dict(tickangle=45, gridcolor=None, showgrid=True),
        yaxis=dict(gridcolor=None, showgrid=True),
        margin=dict(l=50, r=50, t=50, b=50),
        height=600,
        bargap=0.1,
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5",  
    )
    #fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    fig.show()




def plot_points_somme_plotly(df):
    """Diagramme en barres de la somme des points par saison de LeBron James à l'aide de la librairie Plotly"""
    
    df_somme = df.groupby('Year')['PTS'].sum().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_somme['Year'],
        y=df_somme['PTS'],
        marker=dict(
            color=df_somme['PTS'], 
            colorscale='Blues',
            colorbar=dict(title="Somme des points marqués"),
        ),text=df_somme["PTS"].round(1), textposition="outside", 
    ))

    fig.update_layout(
        title="Somme des points par match de LeBron James par saison",
        xaxis_title="Année",
        yaxis_title="Points totaux par saison",
        template="plotly_white",
        xaxis=dict(tickangle=45, gridcolor=None, showgrid=True),
        yaxis=dict(gridcolor=None, showgrid=True),
        margin=dict(l=50, r=50, t=50, b=50),
        height=600,
        bargap=0.1,
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5",
    )
    #fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    fig.show()



import plotly.graph_objects as go
import pandas as pds


def plot_localisation_plotly(df):
    
    """Graphique en barres des résultats des matchs de LeBron James par localisation à l'aide de la librairie Plotly"""
    
    # Créer un tableau avec le nombre de matchs gagnés et perdus par localisation
    tableau_victoire = df[df['Result'] == 1].groupby('Location').size().reset_index(name='Matchs_gagnés')
    tableau_defaite = df[df['Result'] == 0].groupby('Location').size().reset_index(name='Matchs_perdus')
    tableau_final = pd.merge(tableau_victoire, tableau_defaite, on='Location', how="outer")
    
    fig = go.Figure()


    fig.add_trace(go.Bar(
        x=tableau_final['Location'],
        y=tableau_final['Matchs_gagnés'],
        name='Victoires',
        marker=dict(color='lightgreen')
    ))


    fig.add_trace(go.Bar(
        x=tableau_final['Location'],
        y=tableau_final['Matchs_perdus'],
        name='Défaites',
        marker=dict(color='tomato')
    ))


    fig.update_layout(
        title="Résultats des matchs de LeBron James par localisation",
        xaxis_title="Localisation",
        yaxis_title="Nombre de matchs",
        barmode='stack',  
        template="plotly_white",
        xaxis=dict(tickangle=-45, gridcolor=None, showgrid=True),
        yaxis=dict(gridcolor=None, showgrid=True),
        margin=dict(l=50, r=50, t=50, b=50),
        height=600,
        bargap=0.1,
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5", 
    )
    fig.show()



import plotly.graph_objects as go

def victoires_par_annee(df):
    """Graphique en barres du nombre de victoires par saison (en %) de LeBron James à l'aide de la librairie Plotly"""

    groupe_victoires = df.groupby("Year").agg(
        total_matchs=("Result", "count"),
        victoires=("Result", lambda x: (x == 1).sum())
    ).reset_index()

    # Calcule le pourcentage de victoires
    groupe_victoires["percentage_victoire"] = (groupe_victoires["victoires"] / groupe_victoires["total_matchs"]) * 100


    fig = go.Figure(data=[
        go.Bar(
            x=groupe_victoires["Year"],
            y=groupe_victoires["percentage_victoire"],
            text=groupe_victoires["percentage_victoire"].round(1),
            texttemplate="%{text}%",  
            textposition="outside",
            marker=dict(
                color=groupe_victoires["percentage_victoire"],
                colorscale="PuBu",  
                colorbar=dict(
                    title="Pourcentage de victoires (%)",
                    titlefont=dict(size=14), 
                    tickfont=dict(size=12)
                            )
                        )
            )
        ])


    fig.update_layout(
        title="Pourcentage de victoires par année (LeBron James)",
        xaxis=dict(title="Année", tickangle=45, gridcolor=None, showgrid=True),
        yaxis=dict(title="Pourcentage de victoires (%)", gridcolor=None, showgrid=True),
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5",
        hovermode="closest" 
    )
    fig.show()





def pourcentage_reussite_par_annee(df):
    """Graphique interactif du pourcentage de réussite au tir (FG%) par année de LeBron James à l'aide de la librairie Plotly"""
    
    # Calculer le pourcentage de réussite moyen par année dans un dataframe
    groupe = df.groupby("Year").agg(
        fg_percentage_moy=("FG%", "mean")
    ).reset_index()


    fig = go.Figure(data=[
        go.Scatter(
            x=groupe['Year'],
            y=groupe['fg_percentage_moy'],
            mode='lines+markers',  
            text=groupe['fg_percentage_moy'].round(2),  
            marker=dict(
                size=8, 
                color=groupe['fg_percentage_moy'], 
                colorscale="PuBu", 
                colorbar=dict(
                    title="Pourcentage de réussite (FG%)",  
                    titlefont=dict(size=14),  
                    tickfont=dict(size=12) 
                )
            ),
            line=dict(
                color="blue", 
                width=2  
            ),
            hovertemplate="Année: %{x}<br>FG%: %{y:.2f}%<extra></extra>" 
        )
    ])


    fig.update_layout(
        title="Pourcentage de réussite au tir (FG%) par année (LeBron James)",
        xaxis=dict(
            title="Année",
            tickangle=45,
            gridcolor=None, 
            showgrid=True  
        ),
        yaxis=dict(
            title="Pourcentage de réussite (FG%)",
            gridcolor=None,  
            showgrid=True 
        ),
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5",
        hovermode="closest" 
    )
    fig.show()


# Somme des points marqués selon les equipes préférées par LeBron James

def points_par_equipe_adverse(df):
    """Graphique en barres des points marqués par LeBron James contre les équipes adverses à l'aide de la librarie Plotly"""

    #points_par_equipe = df.groupby("OPP")["PTS"].sum().reset_index()
    #points_par_equipe = points_par_equipe.sort_values(by="PTS", ascending=False)

    # Code afin les équipes contre lesquelles LeBron James a marqué le plus de points (30 colonnes OPP_)
    points_par_equipe = []
    equipe_colonnes = [col for col in df.columns if col.startswith('OPP_')]
    
    for equipe_col in equipe_colonnes:
        donnes_equipe = df[df[equipe_col] == 1]
        total_points = donnes_equipe["PTS"].sum()
        points_par_equipe.append((equipe_col.replace('OPP_', ''), total_points))
    
    points_par_equipe_df = pd.DataFrame(points_par_equipe, columns=["Equipe", "Points"])
    points_par_equipe_df = points_par_equipe_df.sort_values(by="Points", ascending=False)


    colors = points_par_equipe_df["Points"]
    color_scale = "bluered"

    fig = go.Figure(
        data=[
            go.Bar(
                x=points_par_equipe_df["Equipe"],
                y=points_par_equipe_df["Points"],
                marker=dict(
                    color=colors,
                    colorscale=color_scale,
                    colorbar=dict(title="Points marqués"),
                ),
                text=points_par_equipe_df["Points"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Les équipes contre lesquelles LeBron James a marqué le plus de points",
        xaxis=dict(title="Équipe adverse", gridcolor=None, showgrid=True),
        yaxis=dict(title="Points marqués", gridcolor=None, showgrid=True),
        template="plotly_white",
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5",
    )
    fig.show()


# Moyenne des points par match contre les équipes adverses
def points_moyens_par_equipe(df):
    """Graphique en barres de la moyenne de points par match contre les équipes adverses à l'aide de la librairie Plotly"""
    

    points_moyens_par_equipe = []
    
    equipe_colonnes = [col for col in df.columns if col.startswith("OPP_")]
    
    for equipe_col in equipe_colonnes:
        donnes_equipe = df[df[equipe_col] == 1]
        moyenne_points = donnes_equipe["PTS"].mean()  
        points_moyens_par_equipe.append((equipe_col.replace('OPP_', ''), moyenne_points))
    
    points_moyens_par_equipe_df = pd.DataFrame(points_moyens_par_equipe, columns=["Equipe", "Points Moyens"])
    
    points_moyens_par_equipe_df = points_moyens_par_equipe_df.sort_values(by="Points Moyens", ascending=False)

    fig = go.Figure(data=[go.Bar(
        x=points_moyens_par_equipe_df["Equipe"],
        y=points_moyens_par_equipe_df["Points Moyens"],
        marker=dict(
            color=points_moyens_par_equipe_df["Points Moyens"],
            colorscale="Blues",
            colorbar=dict(title="Points Moyens")
        ),
        text=points_moyens_par_equipe_df["Points Moyens"].round(1),
        textposition="auto"
    )])
    
    
    fig.update_layout(
        title="Moyenne de points par match contre les équipes adverses",
        xaxis=dict(title="Équipe adverse", gridcolor=None, showgrid=True),
        yaxis=dict(title="Points Moyens", gridcolor=None, showgrid=True),
        template="plotly_white",
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5",
    )
    
    
    fig.show()



# Equipes qui ont le plus marqué de points contre LeBron James 
def somme_des_points_encaissés_par_equipes(df):
    """Graphique en barres des points marqués par LeBron James contre les équipes adverses à l'aide de la librarie Plotly"""

    #points_par_equipe = df.groupby("OPP")["PTS-OPP"].sum().reset_index()
    #points_par_equipe = points_par_equipe.sort_values(by="PTS-OPP", ascending=False)

    points_encaisses_par_equipe = []
    
    equipe_colonnes = [col for col in df.columns if col.startswith("OPP_")]
    
    for equipe_col in equipe_colonnes:
        donnes_equipe = df[df[equipe_col] == 1]
        somme_points = donnes_equipe["PTS-OPP"].sum()
        points_encaisses_par_equipe.append((equipe_col.replace('OPP_', ''), somme_points))
    
    points_encaisses_par_equipe_df = pd.DataFrame(points_encaisses_par_equipe, columns=["Equipe", "Points"])
    
    points_encaisses_par_equipe_df = points_encaisses_par_equipe_df.sort_values(by="Points", ascending=False)
    
    colors = points_encaisses_par_equipe_df["Points"]
    color_scale = "bluered"
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=points_encaisses_par_equipe_df["Equipe"],
                y=points_encaisses_par_equipe_df["Points"],
                marker=dict(
                    color=colors,
                    colorscale=color_scale,
                    colorbar=dict(title="Points marqués"),
                ),
                text=points_encaisses_par_equipe_df["Points"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Les équipes qui ont marqué le plus de points contre LeBron James",
        xaxis=dict(title="Équipe adverse", gridcolor=None, showgrid=True),
        yaxis=dict(title="Points marqués", gridcolor=None, showgrid=True),
        template="plotly_white",
        title_font=dict(size=20),
        plot_bgcolor="#d7d5d5",
    )
    fig.show()


# Moyenne des points marqués des équipes adverses contre LeBron James
def moyennes_des_points_encaissés_par_equipes(df):
    """Graphique en barres de la moyenne de points par match contre les équipes adverses à l'aide de la librairie Plotly"""
    
    #moyenne_points_par_equipe = df.groupby("OPP")["PTS-OPP"].mean().reset_index()
    #moyenne_points_par_equipe.columns = ["OPP", "Points Moyens"]
    #moyenne_points_par_equipe = moyenne_points_par_equipe.sort_values(by="Points Moyens", ascending=False)
    
    points_moyens_encaisses_par_equipe = []
    
    
    equipe_colonnes = [col for col in df.columns if col.startswith("OPP_")]
    
    
    for equipe_col in equipe_colonnes:
        donnes_equipe = df[df[equipe_col] == 1]
        moyenne_points = donnes_equipe["PTS-OPP"].mean()  
        points_moyens_encaisses_par_equipe.append((equipe_col.replace('OPP_', ''), moyenne_points))
    
    
    points_moyens_encaisses_par_equipe_df = pd.DataFrame(points_moyens_encaisses_par_equipe, columns=["Equipe", "Points Moyens"])
    
    
    points_moyens_encaisses_par_equipe_df = points_moyens_encaisses_par_equipe_df.sort_values(by="Points Moyens", ascending=False)
    

    fig=go.Figure(data=[go.Bar(
        x=points_moyens_encaisses_par_equipe_df["Equipe"],
        y=points_moyens_encaisses_par_equipe_df["Points Moyens"],
        marker=dict(
            color=points_moyens_encaisses_par_equipe_df["Points Moyens"],
            colorscale="Blues",
            colorbar=dict(title="Points Moyens")
        ),
        text=points_moyens_encaisses_par_equipe_df["Points Moyens"].round(1),
        textposition="auto"
    )])
    fig.update_layout(title="Moyenne de points encaissés contre les équipes adverses",
                        xaxis=dict(title="Équipe adverse", gridcolor=None, showgrid=True),
                        yaxis=dict(title="Points Moyens", gridcolor=None, showgrid=True),
                        template="plotly_white",
                        title_font=dict(size=20),
                        plot_bgcolor="#d7d5d5",)
    fig.show()

# Histogramme des points marqués par LeBron James

def distribution_points(df):
    """Histogramme des points marqués par LeBron James dans sa carrière à l'aide de la librairie Plotly"""
    fig=go.Figure(data=[go.Histogram(x=df["PTS"], nbinsx=50, marker=dict(color="grey", line=dict(color="black", width=1)))])
    
    fig.update_layout(title="Distribution des points marqués par LeBron James", 
                            xaxis=dict(title="Points marqués", gridcolor=None, showgrid=True),
                            yaxis=dict(title="Nombre de matchs", gridcolor=None, showgrid=True))
    fig.show()
