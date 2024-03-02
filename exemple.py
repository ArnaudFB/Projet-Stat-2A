import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import des données depuis le fichier .npy 

power_maps_3D = np.load("power_maps.npy")

# Transformation en objet DataFrame panda 

N = power_maps_3D.shape[1]
df = pd.DataFrame(power_maps_3D, columns = ["burn-up", "bore", "P_rel", "T_entree", "insertion_barres"] + list(range(N - 5)))
print(power_maps_3D.shape)
# On sélectionne la nappe de puissance située à la 4eme ligne du DataFrame 

pm = df.values[4, 5:].reshape((17*6, 17*6, 14))
print(pm.shape)

# Inversion de l'axe vertical pour avoir les bonnes coordonées navales

pm = pm[::-1, :, :]

# Affichage de la nappe de puissance radiale située à la 7ième maille axiale 

plt.imshow(pm[:, :, 7])
plt.colorbar()
plt.show()

# On sélectionne la nappe de puissance sans les réflecteurs située en haut et en bas du coeur 

core = np.array((17*6, 17*6, 10))
core = pm[:, :, 2:14]

# Coordonnées des assemblages 

coord_assemblages = [[1, 2], [1, 3], 
             [2, 1], [2, 2], [2, 3], [2, 4],
             [3, 1], [3, 2], [3, 3], [3, 4],
                     [4, 2], [4, 3]]

# On recupère la nappe de puissance radiale située à la 7ième maille axiale et située en E2

x =  coord_assemblages[0]
assemb_E2 = core[17 * x[0]: 17 * (x[0] + 1), 17 * x[1]: 17 * (x[1] + 1), :]

plt.imshow(assemb_E2[:, :, 7])
plt.colorbar()
plt.show()

