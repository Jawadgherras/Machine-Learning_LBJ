from ScrapingDonnees_st import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from plotly import graph_objects as go


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
        #plot_bgcolor="#d7d5d5",  
    )
    st.plotly_chart(fig, use_container_width=True)




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
        #plot_bgcolor="#d7d5d5",
    )
    #fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    st.plotly_chart(fig, use_container_width=True)



def tab_localisation(df):
    """Tableau des victoires et défaites selon la localisation (Home/Away)."""

    df['Location'] = df['Location'].replace({1: 'Home', 0: 'Away'})

    tableau_victoire = df[df['Result'] == 1].groupby('Location').size().reset_index(name='Matchs_gagnés')

    tableau_defaite = df[df['Result'] == 0].groupby('Location').size().reset_index(name='Matchs_perdus')

    tableau_final = pd.merge(tableau_victoire, tableau_defaite, on='Location', how="outer")

    tableau_final = tableau_final.fillna(0)

    st.write("### Résultats des matchs selon la localisation")
    st.write(tableau_final)
    

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
        #plot_bgcolor="#d7d5d5", 
    )
    st.plotly_chart(fig, use_container_width=True)



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
        #plot_bgcolor="#d7d5d5",
        hovermode="closest" 
    )
    st.plotly_chart(fig, use_container_width=True)





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
        #plot_bgcolor="#d7d5d5",
        hovermode="closest" 
    )
    st.plotly_chart(fig, use_container_width=True)


def pourcentage_tirs_3_points_par_annee(df):
    """Graphique interactif du pourcentage de réussite au tir à 3 points (3P%) par année de LeBron James à l'aide de la librairie Plotly"""
    groupe=df.groupby("Year").agg(
        Trois_pts_percentage_moy=("3P%", "mean")
    ).reset_index()
    
    fig=go.Figure(data=[go.Scatter(
        x=groupe["Year"],
        y=groupe["Trois_pts_percentage_moy"],
        mode="lines+markers",
        text=groupe["Trois_pts_percentage_moy"].round(2),
        marker=dict(
            size=8,
            color=groupe["Trois_pts_percentage_moy"],
            colorscale=[
                [0, "red"],
                [0.33, "orange"],
                [0.66, "yellow"],
                [1, "green"]
                        ],
            colorbar=dict(title="Pourcentage de réussite (3PT%)", titlefont=dict(size=14), tickfont=dict(size=12)
            ),
        ),
        line=dict(color="green", width=2),
        hovertemplate="Année: %{x}<br>3PT%: %{y:.2f}%<extra></extra>"
        )])
    fig.update_layout(
        title="Pourcentage de réussite au tir à 3PT par année ",
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
        #plot_bgcolor="#d7d5d5",
        hovermode="closest" 
    )
    st.plotly_chart(fig, use_container_width=True)



def points_par_equipe_adverse(df):
    """Graphique en barres des points marqués par LeBron James contre les équipes adverses à l'aide de la librarie Plotly"""

    #points_par_equipe = df.groupby("OPP")["PTS"].sum().reset_index()
    #points_par_equipe = points_par_equipe.sort_values(by="PTS", ascending=False)

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
        #plot_bgcolor="#d7d5d5",
    )
    st.plotly_chart(fig, use_container_width=True)


def points_moyens_par_equipe(df):
    """Graphique en barres de la moyenne de points par match contre les équipes adverses à l'aide de la librairie Plotly"""
    

    points_moyens_par_equipe = []
    
    equipe_colonnes = [col for col in df.columns if col.startswith("OPP_")]
    
    for equipe_col in equipe_colonnes:
        donnees_equipe = df[df[equipe_col] == 1]
        moyenne_points = donnees_equipe["PTS"].mean()  
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
        #plot_bgcolor="#d7d5d5",
    )
    st.plotly_chart(fig, use_container_width=True)


def equipe_plus_affronte(df):
    """Graphique en barres des équipes contre lesquelles LeBron James a le plus affronté à l'aide de la librairie Plotly"""
    
    match_par_equipe = []
    
    equipe_colonnes = [col for col in df.columns if col.startswith("OPP_")]
    
    for equipe_col in equipe_colonnes:
        donnees_equipe = df[df[equipe_col] == 1]
        total_matchs = len(donnees_equipe) 
        match_par_equipe.append((equipe_col.replace('OPP_', ''), total_matchs))
    
    match_par_equipe_df = pd.DataFrame(match_par_equipe, columns=["Equipe", "Matchs_joués"])
    
    match_par_equipe_df = match_par_equipe_df.sort_values(by="Matchs_joués", ascending=False)
    
    colors = match_par_equipe_df["Matchs_joués"]
    color_scale = "Viridis"

    fig = go.Figure(
        data=[
            go.Bar(
                x=match_par_equipe_df["Equipe"],
                y=match_par_equipe_df["Matchs_joués"],
                marker=dict(
                    color=colors,
                    colorscale=color_scale,
                    colorbar=dict(title="Matchs joués"),
                ),
                text=match_par_equipe_df["Matchs_joués"],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Équipes que LeBron James a le plus affrontées",
        xaxis=dict(title="Équipe adverse", gridcolor=None, showgrid=True, tickangle=-45),
        yaxis=dict(title="Nombre de matchs joués", gridcolor=None, showgrid=True),
        template="plotly_white",
        title_font=dict(size=20),
        #plot_bgcolor="#d7d5d5",
    )
    st.plotly_chart(fig, use_container_width=True)


def somme_des_points_encaissés_par_equipes(df):
    """Graphique en barres des points marqués contre LeBron James par les équipes adverses à l'aide de la librarie Plotly"""

    #points_par_equipe = df.groupby("OPP")["PTS-OPP"].sum().reset_index()
    #points_par_equipe = points_par_equipe.sort_values(by="PTS-OPP", ascending=False)

    points_encaisses_par_equipe = []
    
    equipe_colonnes = [col for col in df.columns if col.startswith("OPP_")]
    
    for equipe_col in equipe_colonnes:
        donnees_equipe = df[df[equipe_col] == 1]
        somme_points = donnees_equipe["PTS-OPP"].sum()
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
        #plot_bgcolor="#d7d5d5",
    )
    st.plotly_chart(fig)


def moyennes_des_points_encaissés_par_equipes(df):
    """Graphique en barres de la moyenne de points par match encaissée par LBJ contre les équipes adverses à l'aide de la librairie Plotly"""
    
    #moyenne_points_par_equipe = df.groupby("OPP")["PTS-OPP"].mean().reset_index()
    #moyenne_points_par_equipe.columns = ["OPP", "Points Moyens"]
    #moyenne_points_par_equipe = moyenne_points_par_equipe.sort_values(by="Points Moyens", ascending=False)
    
    points_moyens_encaisses_par_equipe = []
    
    
    equipe_colonnes = [col for col in df.columns if col.startswith("OPP_")]
    
    
    for equipe_col in equipe_colonnes:
        donnees_equipe = df[df[equipe_col] == 1]
        moyenne_points = donnees_equipe["PTS-OPP"].mean()  
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
                        #plot_bgcolor="#d7d5d5"
    )
    st.plotly_chart(fig, use_container_width=True)


def distribution_points(df):
    """Histogramme des points marqués par LeBron James dans sa carrière à l'aide de la librairie Plotly"""
    fig=go.Figure(data=[go.Histogram(x=df["PTS"], nbinsx=50, marker=dict(color="grey", line=dict(color="black", width=1)))])
    
    fig.update_layout(title="Distribution des points marqués par LeBron James", 
                            xaxis=dict(title="Points marqués", gridcolor=None, showgrid=True),
                            yaxis=dict(title="Nombre de matchs", gridcolor=None, showgrid=True))
    st.plotly_chart(fig, use_container_width=True)

import plotly.graph_objects as go
import pandas as pd


def matchs_par_annee(df):
    """Graphique en barres du nombre de matchs joués par LeBron James par saison à l'aide de la librairie Plotly"""
    
    matchs_par_annee = df.groupby("Year").size().reset_index(name="Matchs_joués")
    
    fig = go.Figure(
        go.Bar(
            x=matchs_par_annee["Year"],
            y=matchs_par_annee["Matchs_joués"],
            marker=dict(
                color=matchs_par_annee["Matchs_joués"],
                colorscale="Blues"
            ),
            text=matchs_par_annee["Matchs_joués"],
            textposition="outside"
        )
    )
    
    fig.update_layout(
        title="Nombre de matchs joués par LeBron James par saison",
        xaxis_title="Année",
        yaxis_title="Nombre de matchs joués",
        template="plotly_white",
        xaxis=dict(tickangle=45),
        height=600,
        bargap=0.2
    )
    
    st.plotly_chart(fig, use_container_width=True)