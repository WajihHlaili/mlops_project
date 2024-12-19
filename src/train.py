
import pandas as pd

import numpy as np   # Importe la bibliothèque numpy pour la manipulation de tableaux (arrays).

from sklearn.model_selection import train_test_split  # Importe la fonction train_test_split pour diviser les données en ensembles d'entraînement et de test.
#Model
from sklearn.tree import DecisionTreeRegressor  # Importe le modèle de régression par arbre de décision.
from sklearn.ensemble import RandomForestRegressor  # Importe le modèle de régression par forêt aléatoire.
from sklearn.ensemble import GradientBoostingRegressor  # Importe le modèle de régression par Gradient Boosting Regressor.
from sklearn.linear_model import LinearRegression, Lasso  # Importe les modèles de régression linéaire et de régression Lasso.
from sklearn.linear_model import ElasticNet

#metrique
from sklearn.metrics import classification_report, confusion_matrix  # Importe des métriques pour évaluer les performances des modèles.
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importe GridSearchCV pour effectuer une recherche des meilleurs hyperparamètres et ShuffleSplit pour diviser les données en ensembles de validation.
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import matplotlib.pyplot as plt  # Importe la bibliothèque matplotlib pour la visualisation de données.
import seaborn as sns  # Importe la bibliothèque seaborn pour la visualisation de données basée sur matplotlib.

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import mlflow
import datetime
import warnings
import mlflow.sklearn

pd.options.display.max_columns = None
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler  # Importe StandardScaler pour la normalisation des données.

version = "v1.0"
data_url = "data/processed/data_cleaned.csv"

import os
os.environ['MLFLOW_TRACKING_USERNAME']= "WajihHlaili"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "WajihHlaili888"


#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/WajihHlaili/my-first-repo.mlflow')
mlflow.set_experiment("CarPricePrediction_mlFlow-experiment")




data = pd.read_csv("data/processed/data_cleaned.csv")

data.head()

data.info()


X = data.drop(columns='selling_price', axis=1)
Y = data['selling_price']


np.random.seed(42)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


X_train.shape, X_test.shape


Y_train.shape, Y_test.shape


# 1. Linear Regression
# 2. Decision Tree Regressor
# 3. Random Forest Regressor
# 4. Gradient Boosting Regression

mlflow.sklearn.autolog(disable=True)

models = {"Linear Regression": LinearRegression(),
         "Decision Tree Regressor": DecisionTreeRegressor(),
         "Random Forest Regressor": RandomForestRegressor(),
         "Gradient Boosting Regression":GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, random_state=0)}


# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, Y_train, Y_test):
    """
    Fits and evaluates given machine learning models
    models: a dict of different Scikit-Learn machine learning models
    X_train: training data (no labels)
    X_test: testing data (no labels)
    Y_train : training labels
    Y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    #Make a dictionary to keep models scores
    model_scores = {}
    # loop through Models
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("data_url",data_url)
            mlflow.log_param("data_version",version)
            mlflow.log_param("input_rows",data.shape[0])
            mlflow.log_param("input_cols",data.shape[1])
            #model fitting and training
            mlflow.set_tag(key= "model",value=name)
            params = model.get_params()
            mlflow.log_params(params)
            model.fit(X_train, Y_train)
            train_features_name = f'{X_train=}'.split('=')[0]
            train_label_name = f'{Y_train=}'.split('=')[0]
            mlflow.set_tag(key="train_features_name",value= train_features_name)
            mlflow.set_tag(key= "train_label_name",value=train_label_name)
            predicted=model.predict(X_train)
            # Assuming y_true contains true target values and y_pred contains predicted target values
            mae = mean_absolute_error(Y_train, predicted)
            mse = mean_squared_error(Y_train, predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_train, predicted)
            Accuracy =  model.score(X_test, Y_test)
            mlflow.log_metric("Accuracy",Accuracy)
            mlflow.log_metric("MAE",mae)
            mlflow.log_metric("MSE",mse)
            mlflow.log_metric("RMSE",rmse)
            mlflow.log_metric("R2",r2)
            mlflow.sklearn.log_model(model,artifact_path="ML_models")
        


fit_and_score(models, X_train, X_test, Y_train, Y_test)

#Reading Pandas Dataframe from mlflow
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.Accuracy <1")
run_id = df_mlflow.loc[df_mlflow['metrics.Accuracy'].idxmax()]['run_id']
print(run_id)



scaller = StandardScaler()
x_train_sc = scaller.fit_transform(X_train)
x_test_sc = scaller.transform(X_test)
x_train_sc[0:10,:]


logged_model = f'runs:/{run_id}/ML_models'
model = mlflow.pyfunc.load_model(logged_model)

import joblib
joblib.dump(model, 'models/new_model.pkl')
mlflow.register_model(model_uri=logged_model, name='best_model')
print("Model logged to DagsHub.")



