

#     * feature name: Le nom de chaque colonne ou caractéristique dans l'ensemble de données.
# 
#     * year: L'année de fabrication ou de mise en circulation du véhicule.
# 
#     * selling_price: Le prix de vente du véhicule.
# 
#     * km_driven: Le nombre de kilomètres parcourus par le véhicule.
# 
#     * fuel: Le type de carburant utilisé par le véhicule (par exemple, essence, diesel, électrique, etc.).
# 
#     *  seller_type: Le type de vendeur (par exemple, vendeur individuel, concessionnaire, etc.).
# 
#     * transmission: Le type de transmission du véhicule (par exemple, manuelle, automatique).
# 
#     * owner: Le nombre de propriétaires précédents du véhicule.
# 
#     * mileage: La consommation de carburant du véhicule en termes de kilomètres par litre.
# 
#     * engine: La cylindrée du moteur du véhicule.
# 
#     * max_power: La puissance maximale du moteur du véhicule.
# 
#     * torque: Le couple du moteur du véhicule.
# 
#     * seats: Le nombre de places (sièges) dans le véhicule.
#     
# Les données originales proviennent de Vehicle dataset sur Kaggle URL: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?fbclid=IwAR1mCHhtXoBV8PZESSd6621bkUp7hTrrvmTid6VzgcAuTZJqtgw-zECtAXw

# # 1. Importation des packages

import pandas as pd  # Importe la bibliothèque pandas pour la manipulation de données sous forme de dataframes.
import numpy as np   # Importe la bibliothèque numpy pour la manipulation de tableaux (arrays).

import matplotlib.pyplot as plt  # Importe la bibliothèque matplotlib pour la visualisation de données.
import seaborn as sns  # Importe la bibliothèque seaborn pour la visualisation de données basée sur matplotlib.

# # 2. charger des données

data=pd.read_csv('Car.csv') # lire

# # 3. Visualisation de données

data.head() #Affichage des premières lignes des données

# Utilisation de la méthode .shape pour obtenir les dimensions du dataframe Dimensions des données (nombre de lignes et de colonnes
dimensions = data.shape

# Affichage des dimensions
print(dimensions)


data.info() #Informations sur les données :

data.describe()

data.describe(include=object)
#obtenir des statistiques descriptives spécifiques aux colonnes de type objet (chaînes de caractères)

data['seller_type'].value_counts()
 # Comptage des valeurs

data.isna().sum()

data.boxplot()

# ### Matrice de coleration
# La matrice de corrélation indique les valeurs de corrélation, qui mesurent le degré de relation linéaire entre chaque paire de variables.
import seaborn as sns

numeric_data = data.select_dtypes(include=['number'])
# matrice de correlation
correlation_matrix = numeric_data.corr()
#Diagramme de corrélation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

import matplotlib.pyplot as plt  # Importe la bibliothèque matplotlib pour la création de graphiques.
import seaborn as sns            # Importe la bibliothèque seaborn pour une visualisation améliorée.

def custom_bar_plot(data, col, figsize=(15, 7), rotation=0):

    plt.figure(figsize=figsize)  # Crée une nouvelle figure avec la taille spécifiée.

    # Crée un graphique à barres (countplot) en utilisant la colonne spécifiée et l'ordre des catégories.
    plot = sns.countplot(x=col, data=data, order=data[col].value_counts().index)

    plt.xticks(rotation=rotation)  # Fait pivoter les étiquettes sur l'axe des x (si rotation est non nul).

    # Ajoute des étiquettes au-dessus de chaque barre pour afficher le nombre de chaque catégorie.
    for p in plot.patches:
        plot.annotate(f'{p.get_height()}',
                      (p.get_x() + p.get_width() / 2.0, p.get_height()),
                      ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.xlabel(col)   # Définit le label de l'axe des x avec le nom de la colonne.
    plt.ylabel('Count')  # Définit le label de l'axe des y comme "Count".
    plt.title(f'Count of {col}')  # Définit le titre du graphique en fonction de la colonne spécifiée.
    plt.show()  # Affiche le graphique.

# Exemple d'utilisation
custom_bar_plot(data, 'fuel', figsize=(12, 6), rotation=45)  # Appelle la fonction .

data['fuel'].hist()

import pygwalker as pyg
walker = pyg.walk(
    data,
    spec="./chart_meta_0.json",    # this json file will save your chart state, you need to click save button in ui mannual when you finish a chart, 'autosave' will be supported in the future.
    use_kernel_calc=True,          # set `use_kernel_calc=True`, pygwalker will use duckdb as computing engine, it support you explore bigger dataset(<=100GB).
)




