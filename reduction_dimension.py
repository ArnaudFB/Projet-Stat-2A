import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_pca_correlation_graph
from mlxtend.data import iris_data
import plotly.express as px
from matplotlib.patches import Circle

# Import des données depuis le fichier .npy
power_maps_3D = np.load("power_maps.npy")

# Transformation en objet DataFrame panda
N = power_maps_3D.shape[1]
nappe_data = pd.DataFrame(power_maps_3D, columns = ["burn-up", "bore", "P_rel", "T_entree", "insertion_barres"] + list(range(N - 5)))
features = ['burn-up', 'bore', 'P_rel', 'T_entree'] + list(range(N - 5)) #Variables du réacteur
target = ['burn-up', 'bore', 'P_rel', 'T_entree', 'insertion_barres'] #Variables que l'on cherche à expliquer
# Attribution des values de nos variables pour les standardiser
x = nappe_data.loc[:, features].values
y = nappe_data.loc[:, target].values
x = StandardScaler().fit_transform(x)

# Réalisation de l'ACP
pca = PCA(n_components=3)
nappe_data_pca = pca.fit_transform(x)
principalDf = pd.DataFrame(data = nappe_data_pca
             , columns = ["Axe 1", "Axe 2", "Axe 3"])

print(pca.explained_variance_ratio_)
# Visualisation 2D de l'ACP
finalDf = pd.concat([principalDf, nappe_data[target]], axis = 1)

fig = px.scatter_matrix(
    finalDf,
    dimensions=["Axe 1", "Axe 2", "Axe 3"],
    color='burn-up'
 )
fig.update_traces(diagonal_visible=False)
fig.show()

fig = px.scatter_matrix(
    finalDf,
    dimensions=["Axe 1", "Axe 2", "Axe 3"],
    color='bore'
 )
fig.update_traces(diagonal_visible=False)
fig.show()
fig = px.scatter_matrix(
    finalDf,
    dimensions=["Axe 1", "Axe 2", "Axe 3"],
    color='P_rel'
 )
fig.update_traces(diagonal_visible=False)
fig.show()
fig = px.scatter_matrix(
    finalDf,
    dimensions=["Axe 1", "Axe 2", "Axe 3"],
    color='T_entree'
 )
fig.update_traces(diagonal_visible=False)
fig.show()
fig = px.scatter_matrix(
    finalDf,
    dimensions=["Axe 1", "Axe 2", "Axe 3"],
    color='insertion_barres'
 )
fig.update_traces(diagonal_visible=False)
fig.show()

"""
figure, correlation_matrix = plot_pca_correlation_graph(nappe_data_pca,
                                                        ["burn-up", "bore"],
                                                        dimensions=(1, 2),
                                                        figure_axis_size=10)
figure.show()
"""