import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import des données depuis le fichier .npy
power_maps_3D = np.load("power_maps.npy")

# Transformation en objet DataFrame panda
N = power_maps_3D.shape[1]
nappe_data = pd.DataFrame(power_maps_3D, columns = ["burn-up", "bore", "P_rel", "T_entree", "insertion_barres"] + list(range(N - 5)))
nappe_data.dropna()
#nappe_data = nappe_data.loc[nappe_data["insertion_barres"] == 5.0]

features = ['burn-up', 'bore', 'P_rel', 'T_entree', 'insertion_barres']
print(nappe_data[features].describe())
print(nappe_data[features].head(1))

for var in features :
    print(nappe_data[var].value_counts())

print(nappe_data[["P_rel", "bore"]].value_counts())

nappe_data_config_0 = nappe_data.loc[nappe_data["insertion_barres"] == 0.0]
nappe_data_config_1 = nappe_data.loc[nappe_data["insertion_barres"] == 1.0]
nappe_data_config_2 = nappe_data.loc[nappe_data["insertion_barres"] == 2.0]
nappe_data_config_3 = nappe_data.loc[nappe_data["insertion_barres"] == 3.0]
nappe_data_config_4 = nappe_data.loc[nappe_data["insertion_barres"] == 4.0]
nappe_data_config_5 = nappe_data.loc[nappe_data["insertion_barres"] == 5.0]
pm = nappe_data_config_0.values[1, 5:].reshape((17*6, 17*6, 14))

# Inversion de l'axe vertical pour avoir les bonnes coordonées navales

pm = pm[::-1, :, :]

# Affichage de la nappe de puissance radiale située sur toutes les mailles axiales
#for i in range(14) :
#    plt.imshow(pm[:, :, i])
#    plt.colorbar()
#    plt.title("Maille axiale n° " + str(i))
#    plt.show()


# Visualisation des nappes selon les config de barres de contrôle

pm_config_0 = nappe_data_config_0.loc[(nappe_data_config_0['P_rel'] == 1) &
                                      (nappe_data_config_0['bore'] == 0) &
                                      (nappe_data_config_0['burn-up'] == 24) &
                                      (nappe_data_config_0['T_entree'] == 220)].values[0, 5:].reshape((17*6, 17*6, 14))

# Inversion de l'axe vertical pour avoir les bonnes coordonées navales

pm_config_0 = pm_config_0[::-1, :, :]

# Affichage de la nappe de puissance radiale située à la 7ième maille axiale
for i in range(14) :
    plt.imshow(pm_config_0[:, :, i])
    plt.colorbar()
    plt.title("Config 0, P_rel 1, bore 0, burn-up 24, T_entree 220 . Maille axiale n° " + str(i))
    plt.show()


