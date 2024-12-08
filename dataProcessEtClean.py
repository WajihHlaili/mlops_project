
# # 1. Importation des packages

import pandas as pd  # Importe la bibliothèque pandas pour la manipulation de données sous forme de dataframes.
import numpy as np   # Importe la bibliothèque numpy pour la manipulation de tableaux (arrays).

import matplotlib.pyplot as plt  # Importe la bibliothèque matplotlib pour la visualisation de données.
import seaborn as sns  # Importe la bibliothèque seaborn pour la visualisation de données basée sur matplotlib.

# # 2. charger des données

data=pd.read_csv('Data_csv/Car.csv') # lire

# # 3. Data Processing and Cleaning

# Verifier duplication
data.drop_duplicates()


data.drop(['torque','seller_type'],axis=1,inplace=True)

data.head()

data['name'] = data['name'].str.split(" ", expand=True)[0]
# extrait la première partie de chaque valeur dans la colonne 'name' jusqu'à un espace.

def extract_column(data, col):
    data[col] = data[col].str.split(" ", expand=True)[0]

def convert_to_float(data, col):
    data[col] = pd.to_numeric(data[col])

def fill_missing_values(data, col):
    data[col].fillna(data[col].astype("float64").mean(), inplace=True)

extract_column(data,'name')
extract_column(data,'mileage')
extract_column(data,'engine')
extract_column(data,'max_power')

convert_to_float(data,'mileage')
convert_to_float(data,'engine')
convert_to_float(data,'max_power')

fill_missing_values(data,'mileage')
fill_missing_values(data,'engine')
fill_missing_values(data,'seats')
fill_missing_values(data,'max_power')

data.isna().sum()

data['year'].fillna(data['year'].mean(), inplace=True)
data['selling_price'].fillna(data['selling_price'].mean(), inplace=True)
data['km_driven'].fillna(data['km_driven'].mean(), inplace=True)


data.dropna(subset=['fuel', 'transmission', 'owner'], inplace=True)


data.isna().sum()

data.head()

# . Comptez le nombre d'occurrences de chaque catégorie dans la colonne 'name'.
name_counts = data['name'].value_counts()

#remplacez les catégories ayant un count < 15 par 'other'.
threshold = 15
name_counts['other'] = name_counts[name_counts < threshold].sum()
name_counts = name_counts[name_counts >= threshold]

# . Créez un graphique à barres pour la colonne 'name' mise à jour.
plt.figure(figsize=(12, 6))
plot = sns.barplot(x=name_counts.index, y=name_counts.values)
plt.xticks(rotation=90)
plt.xlabel('Car Name')
plt.ylabel('Count')
plt.title('Car Name Counts (with "other" category)')
plt.show()


data.sample(10)

import seaborn as sns
import matplotlib.pyplot as plt

# Sélection des colonnes à inclure dans la pairplot
columns_to_include = ['year', 'selling_price', 'km_driven', 'transmission']

# Création d'une palette de couleurs personnalisée

# Utilisation du style de base de seaborn pour une apparence plus esthétique
sns.set(style='whitegrid')

# Création de la pairplot avec la palette de couleurs personnalisée
pairplot = sns.pairplot(data[columns_to_include], hue='transmission', diag_kind='kde', kind='scatter', height=3.5)

# Titre de la pairplot
plt.suptitle("Relation entre l'année, le prix de vente, le kilométrage et la transmission des voitures")

# Affichage de la pairplot
plt.show()



# Création d'une palette de couleurs personnalisée
custom_palette = sns.color_palette("Set2")  # Vous pouvez choisir une palette qui vous convient

# Création d'un graphique à barres pour la colonne 'fuel'
plt.figure(figsize=(10, 6))
plot = sns.countplot(x='fuel', data=data, palette=custom_palette)

# Titre du graphique
plt.title("Distribution des types de carburant")

# Affichage des étiquettes de comptage au-dessus des barres
for p in plot.patches:
    plot.annotate(f'{p.get_height()}',
                  (p.get_x() + p.get_width() / 2.0, p.get_height()),
                  ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Ajout de commentaires sur les observations
plt.text(0.5, 3000, ".", ha='center', va='center', fontsize=12, color='blue')

# Affichage du graphique
plt.show()




# Création d'une palette de couleurs personnalisée
custom_palette = sns.color_palette("Set2")  # Vous pouvez choisir une palette qui vous convient

# Compter les occurrences de chaque type de carburant
fuel_counts = data['fuel'].value_counts()

# Définir un seuil pour regrouper les types moins fréquents en 'other'
threshold = 15
other_fuel_count = fuel_counts[fuel_counts < threshold].sum()

# Remplacer les types de carburant moins fréquents par 'other' dans le dataframe
data['fuel'] = data['fuel'].apply(lambda x: 'other' if fuel_counts[x] < threshold else x)

# Création d'un graphique à barres pour la colonne 'fuel'
plt.figure(figsize=(10, 6))
plot = sns.countplot(x='fuel', data=data, palette=custom_palette)

# Titre du graphique
plt.title("Distribution des types de carburant")

# Affichage des étiquettes de comptage au-dessus des barres
for p in plot.patches:
    plot.annotate(f'{p.get_height()}',
                  (p.get_x() + p.get_width() / 2.0, p.get_height()),
                  ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Commentaires plus détaillés sur les observations
plt.text(0.9, 3000, f"Les types de carburant Diesel et Petrol sont les plus courants.\n"
                     f"Les autres types ont été regroupés sous 'other' ({other_fuel_count} occurrences).",
         ha='center', va='center', fontsize=12, color='blue')

# Affichage du graphique
plt.show()


manuel = data[data['transmission']=='Manual']
automatique = data[data['transmission']=='Automatic']

# Sélection des colonnes à inclure dans la pairplot
columns_to_include = ['year', 'selling_price', 'km_driven', 'fuel']

# Filtrage des données pour les voitures automatiques
automatique = data[data['transmission'] == 'Automatic']

# Création d'une palette de couleurs personnalisée
custom_palette = sns.color_palette("husl")  # Vous pouvez choisir une palette qui vous convient

# Création de la pairplot pour les voitures automatiques en fonction du type de carburant
plt.figure(figsize=(12, 8))
pairplot = sns.pairplot(automatique[columns_to_include], hue='fuel', diag_kind='kde', kind='scatter',height=3.5)

# Titre de la pairplot
plt.suptitle("Relation entre l'année, le prix de vente, le kilométrage et le type de carburant des voitures automatiques")

# Affichage de la pairplot
plt.show()



sns.barplot(x='owner',y='selling_price',data=data,palette='spring')

# Supprimer les entrées correspondant aux "voitures d'essai" (test Driver Car)
data = data[~(data['owner'] == 'Test Drive Car')]

# Regrouper tous les autres types d'anciens propriétaires en "Third Owner & Above"
data['owner'] = data['owner'].apply(lambda x: x if x in ['First Owner', 'Second Owner'] else 'Third Owner & Above')

# Afficher un graphique à barres pour la colonne 'owner' mise à jour
sns.barplot(x='owner',y='selling_price',data=data,palette='spring')



# Calcul de la moyenne des kilomètres parcourus chaque année
km_mean = data.groupby('year')['km_driven'].mean()

# Filtrage des données pour les voitures manuelles et automatiques
manual = data[data['transmission'] == 'Manual']
automatic = data[data['transmission'] == 'Automatic']

# Création d'une figure avec deux sous-graphiques
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

# Sous-graphique 1 : Barres pour la moyenne des kilomètres parcourus chaque année
sns.barplot(x=km_mean.index, y=km_mean, ax=ax[0], palette='viridis')
ax[0].set_title('Moyenne des kilomètres parcourus chaque année')
ax[0].set_xlabel('Année')
ax[0].set_ylabel('Kilomètres parcourus')

# Sous-graphique 2 : Distribution des kilomètres parcourus pour les voitures manuelles et automatiques
sns.histplot(data=manual, x='km_driven', label='Manuelles', kde=True, ax=ax[1], color='skyblue')
sns.histplot(data=automatic, x='km_driven', label='Automatiques', kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Distribution des kilomètres parcourus')
ax[1].legend()

# Ajustements visuels
plt.tight_layout()

# Affichage du graphique
plt.show()


# Créer un box plot pour la colonne 'km_driven'
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['km_driven'], color='skyblue')

# Titre du graphique
plt.title('Box Plot des Kilomètres Parcourus')

# Affichage du graphique
plt.show()


# Calcul de la moyenne du prix de vente pour les voitures manuelles chaque année
year_mean_manual = data[data['transmission'] == 'Manual'].groupby('year')['selling_price'].mean()

# Calcul de la moyenne du prix de vente pour les voitures automatiques chaque année
year_mean_automatic = data[data['transmission'] == 'Automatic'].groupby('year')['selling_price'].mean()

# Création d'une figure avec deux sous-graphiques
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

# Sous-graphique 1 : Barres pour la moyenne du prix de vente des voitures manuelles chaque année
ax[0].bar(year_mean_manual.index, year_mean_manual, color='skyblue')
ax[0].set_title('Prix de vente moyen des voitures manuelles chaque année')
ax[0].set_xlabel('Année')
ax[0].set_ylabel('Prix de vente')

# Sous-graphique 2 : Barres pour la moyenne du prix de vente des voitures automatiques chaque année
ax[1].bar(year_mean_automatic.index, year_mean_automatic, color='salmon')
ax[1].set_title('Prix de vente moyen des voitures automatiques chaque année')
ax[1].set_xlabel('Année')
ax[1].set_ylabel('Prix de vente')

# Affichage du graphique
plt.show()




# Créer un box plot pour la distribution des prix de vente
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['selling_price'], color='skyblue')

# Titre du graphique
plt.title('Boîte à moustaches de la distribution des prix de vente')

# Afficher les outliers en utilisant des points
sns.stripplot(x=data['selling_price'], color='salmon', alpha=0.5)

# Affichage du graphique
plt.show()

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_cleaned = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data_cleaned

# Supprimer les outliers du prix de vente (selling_price)
data = remove_outliers_iqr(data, 'selling_price')

# Affichage d'un scatter plot pour visualiser la relation entre 'km_driven' et 'selling_price' en fonction de la transmission
sns.scatterplot(data=data, x="km_driven", y="selling_price", hue='transmission', style='transmission')
plt.title('Relation entre Kilométrage et Prix de Vente (par Transmission)')
plt.xlabel('Kilométrage')
plt.ylabel('Prix de Vente')
plt.show()

# Définition d'une fonction pour supprimer les outliers de 'km_driven'
def remove_outlier_km_driven(data):
    temp = pd.DataFrame()

    data_km_driven = data['km_driven']
    Q1 = data_km_driven.quantile(0.25)
    Q3 = data_km_driven.quantile(0.75)
    IQR = Q3 - Q1

    # Filtrage des outliers
    data_outlier = data_km_driven[(data_km_driven < (Q1 - 1.5 * IQR)) | (data_km_driven > (Q3 + 1.5 * IQR))]
    temp = pd.concat([temp, data_outlier])

    return data.drop(temp.index)

# Suppression des outliers de 'km_driven'
data = remove_outlier_km_driven(data)

# Affichage d'un scatter plot mis à jour pour visualiser la relation après la suppression des outliers
sns.scatterplot(data=data, x="km_driven", y="selling_price", hue='transmission', style='transmission')
plt.title('Relation entre Kilométrage et Prix de Vente (par Transmission) - Après suppression des outliers')
plt.xlabel('Kilométrage')
plt.ylabel('Prix de Vente')
plt.show()


# La suppression des outliers a permis d'améliorer la qualité des données et de rendre la relation entre le kilométrage et le prix de vente plus claire dans le scatter plot.

# Affichage d'un scatter plot pour visualiser la relation entre 'km_driven' et 'selling_price' en fonction de la transmission
sns.scatterplot(data=data, x="km_driven", y="selling_price", hue='transmission', style='transmission')
plt.title('Relation entre Kilométrage et Prix de Vente (par Transmission)')
plt.xlabel('Kilométrage')
plt.ylabel('Prix de Vente')
plt.show()

# Définition d'une fonction pour supprimer les outliers de 'selling_price' en fonction de l'année et de la transmission
def remove_outlier_selling_price(data):
    temp = pd.DataFrame()

    for year in sorted(data.year.unique()):
        for transmission in ['Manual', 'Automatic']:
            year_transmission_data = data[(data['year'] == year) & (data['transmission'] == transmission)]
            year_transmission_price = year_transmission_data['selling_price']

            Q1 = year_transmission_price.quantile(0.25)
            Q3 = year_transmission_price.quantile(0.75)
            IQR = Q3 - Q1

            # Filtrage des outliers spécifiques à chaque année et à chaque transmission
            outlier_data = year_transmission_data[(year_transmission_price < (Q1 - 1.5 * IQR)) | (year_transmission_price > (Q3 + 1.5 * IQR))]
            temp = pd.concat([temp, outlier_data])

    return data.drop(temp.index)

# Suppression des outliers de 'selling_price' en fonction de l'année et de la transmission
data = remove_outlier_selling_price(data)

# Affichage d'un scatter plot mis à jour pour visualiser la relation après la suppression des outliers
sns.scatterplot(data=data, x="km_driven", y="selling_price", hue='transmission', style='transmission')
plt.title('Relation entre Kilométrage et Prix de Vente (par Transmission) - Après suppression des outliers')
plt.xlabel('Kilométrage')
plt.ylabel('Prix de Vente')
plt.show()


def remove_outliers_iqr_by_year(data, column):
    temp = pd.DataFrame()

    for year in sorted(data['year'].unique()):
        year_data = data[data['year'] == year]
        year_values = year_data[column]

        Q1 = year_values.quantile(0.25)
        Q3 = year_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = year_data[(year_values < lower_bound) | (year_values > upper_bound)]
        temp = pd.concat([temp, outliers])

    return data.drop(temp.index)

# Supprimer les outliers du prix de vente (selling_price) en fonction de l'année
data = remove_outliers_iqr_by_year(data, 'selling_price')

# Créer un sous-ensemble de données en excluant les années antérieures à 2005
data = data[data['year'] > 2005]

# Calcul du prix de vente moyen des voitures manuelles chaque année
year_mean_manual = data[data['transmission'] == 'Manual'].groupby('year')['selling_price'].mean()

# Calcul du prix de vente moyen des voitures automatiques chaque année
year_mean_automatic = data[data['transmission'] == 'Automatic'].groupby('year')['selling_price'].mean()

# Création de deux sous-graphiques pour afficher les moyennes
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

# Sous-graphique 1 : Barres pour la moyenne du prix de vente des voitures manuelles chaque année
ax[0].bar(year_mean_manual.index, year_mean_manual, color='skyblue')
ax[0].set_title('Prix de vente moyen des voitures manuelles chaque année')
ax[0].set_xlabel('Année')
ax[0].set_ylabel('Prix de vente')

# Sous-graphique 2 : Barres pour la moyenne du prix de vente des voitures automatiques chaque année
ax[1].bar(year_mean_automatic.index, year_mean_automatic, color='salmon')
ax[1].set_title('Prix de vente moyen des voitures automatiques chaque année')
ax[1].set_xlabel('Année')
ax[1].set_ylabel('Prix de vente')

# Affichage du graphique
plt.show()


data.head()

data.drop(['mileage','seats'],axis=1,inplace=True) # D'apres le matrice corr

data.fuel.unique()

data.replace('Manual',2, inplace = True)
data.replace('Automatic',1, inplace = True)

data.replace('First Owner',0, inplace = True)
data.replace('Second Owner',1, inplace = True)
data.replace('Third Owner & Above',2, inplace = True)

data.replace('Diesel',1, inplace = True)
data.replace('Petrol',2, inplace = True)
data.replace('CNG',3, inplace = True)
data.replace('LPG',4, inplace = True)

#Changing types
data['transmission'] = data['transmission'].astype(int)
data.drop(columns=['name'], axis=1, inplace = True)

data.to_csv('./Data_csv/data_cleaned.csv',index=False)





