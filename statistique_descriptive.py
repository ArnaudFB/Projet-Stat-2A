import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import des données depuis le fichier .npy
power_maps_3D = np.load("power_maps.npy")

# Transformation en objet DataFrame panda
N = power_maps_3D.shape[1]
nappe_data = pd.DataFrame(power_maps_3D, columns = ["burn-up", "bore", "P_rel", "T_entree", "insertion_barres"] + list(range(N - 5)))
nappe_data.dropna()
# Visualiser les nappes selon une config barre mais différentes burn-up, température, puissance
#nappe_data = nappe_data.loc[nappe_data["insertion_barres"] == 5.0]

features = ['burn-up', 'bore', 'P_rel', 'T_entree', 'insertion_barres']
print(nappe_data[features].describe())
print(nappe_data[features].head(1))

pm = nappe_data.values[1, 5:].reshape((17*6, 17*6, 14))

# Inversion de l'axe vertical pour avoir les bonnes coordonées navales

pm = pm[::-1, :, :]

# Affichage de la nappe de puissance radiale située à la 7ième maille axiale

plt.imshow(pm[:, :, 2])
plt.colorbar()
plt.show()

for var in features :
    print(nappe_data[var].value_counts())

print(nappe_data[["P_rel", "bore"]].value_counts())
