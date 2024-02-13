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
nappe_data.dropna()
#nappe_data_filtered = nappe_data.loc[:, (nappe_data != 0).all(axis=0)]
#nappe_data_filtered = nappe_data_filtered.loc[~(nappe_data_filtered[list(range(N - 5))] == 0).all(axis=1)]

#N2 = nappe_data_filtered.values.shape[1]
#print([N,N2])
features = ['burn-up', 'bore', 'P_rel', 'T_entree'] + list(range(N - 5)) #Variables du réacteur
target = ['burn-up', 'bore', 'P_rel', 'T_entree', 'insertion_barres'] #Variables que l'on cherche à expliquer

# Attribution des values de nos variables pour les standardiser
x = nappe_data.loc[:, features]#.values
x.drop(x.loc[x.sum(axis=1)==0].index, inplace=True)
x.drop(columns=x.columns[x.sum()==0], inplace=True)
x.columns = x.columns.astype(str)
y = nappe_data.loc[:, target].values
x = StandardScaler().fit_transform(x)

# Statistiques decriptives des données
print([nappe_data[target].mean(), nappe_data[target].quantile([0.25, 0.5, 0.75]), nappe_data[target].max(), nappe_data[target].min()])
# print(x)

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

# Get the loadings (correlations between original variables and principal components)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a correlation circle plot
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the correlation circle
for variable in target:
    i = nappe_data.columns.get_loc(variable)
    x, y = loadings[i, 0], loadings[i, 1]
    ax.arrow(0, 0, x, y, color='r', alpha=0.5)
    ax.text(x, y, variable, color='b', fontsize=12)

# Adjust plot settings
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal', 'box')
ax.grid(True)
ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)
ax.set_title('Correlation Circle')

plt.show()