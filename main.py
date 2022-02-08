import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.covariance import oas

# Stocker les données du challenge (X_train, y_train, X_test et des données supplémentaires)
train_data = pd.read_csv('X_train.csv', index_col=0)
train_labels = pd.read_csv('y_train.csv', index_col=0)
test_data = pd.read_csv('X_test.csv', index_col=0)
additionnal_data = pd.read_csv('supplementary_data_Vkoyn8z.csv', index_col=0)

# On utilise dans ce cas les données supplémentaires
CLASS_1 = []
CLASS_2 = []
CLASS_3 = []
CLASS_4 = []
for element in tqdm(train_data["ID_TARGET"].values):
    CLASS_1.append(additionnal_data.loc[element]["CLASS_LEVEL_1"])
    CLASS_2.append(additionnal_data.loc[element]["CLASS_LEVEL_2"])
    CLASS_3.append(additionnal_data.loc[element]["CLASS_LEVEL_3"])
    CLASS_4.append(additionnal_data.loc[element]["CLASS_LEVEL_4"])
train_data["CLASS_LEVEL_1"] = CLASS_1
train_data["CLASS_LEVEL_2"] = CLASS_2
train_data["CLASS_LEVEL_3"] = CLASS_3
train_data["CLASS_LEVEL_4"] = CLASS_4
train_data["RET_TARGET"] = train_labels.values

# On impute les données manquantes par la moyenne
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_data = imp.fit_transform(train_data)
columns = list(test_data.columns) + ["CLASS_LEVEL_1", "CLASS_LEVEL_2", "CLASS_LEVEL_3", "CLASS_LEVEL_4", "RET_TARGET"]
train_data = pd.DataFrame(train_data, columns=columns)

train_data = train_data.drop("ID_DAY", axis=1)


"""
# Scaling data with Robust Scaler model
scaler = RobustScaler()
scaler.fit(train_data.drop("RET_TARGET", axis=1))
scaled_X = scaler.transform(train_data.drop("RET_TARGET", axis=1))

# Run a PCA to have more reliable data
model = PCA()
coord = model.fit_transform(scaled_X)

# Feature selection
coord_new = SelectKBest(f_classif, k=60).fit_transform(coord, train_data["RET_TARGET"].values)
"""

# On split le DataSet en plusieurs DataFrames
X_train, X_test, y_train, y_test = train_test_split(train_data.drop("RET_TARGET", axis=1), train_data["RET_TARGET"],
                                                    test_size=0.1, random_state=42)

# Cross validation : ne pas run cette partie, ca prend des heures
cv = True
if cv:
    def weighted_accuracy(y_test, y_pred):
        y_abs = np.abs(y_test)
        norm = y_abs.sum()
        score = ((np.sign(y_pred) == np.sign(y_test)) * y_abs).sum() / norm
        return score

    # On utilise la métrique du challenge comme scoring functio,
    my_func = make_scorer(weighted_accuracy, greater_is_better=True)

    # On fait la liste des paramètres à tester
    params = {
        "n_neighbors": [100, 200]
    }



    from sklearn.neighbors import KNeighborsClassifier

    random_search = GridSearchCV(KNeighborsClassifier(), param_grid=params, n_jobs=-1, verbose=3, scoring=my_func)
    random_search.fit(X_train, np.sign(y_train))

# On utilise le meilleur modèle selon la cross validation
boost = XGBClassifier(colsample_bytree=0.2, gamma=0.1, learning_rate=0.1,
                      max_depth=10, min_child_weight=5, verbosity=3, n_estimators=100,
                      objective="binary:logistic", num_parallel_tree=6)

# On entraine le modèle, et on prédit
boost.fit(X_train, np.sign(y_train))
y_pred = boost.predict(X_test)


def weighted_accuracy(y_test, y_pred):
    y_abs = np.abs(y_test)
    norm = y_abs.sum()
    score = ((np.sign(y_pred) == np.sign(y_test)) * y_abs).sum() / norm
    return score


print(weighted_accuracy(y_test=y_test, y_pred=y_pred))

# Tout ce qui suit sert à créer le DataFrame à submit sur le site du challenge, pas très utile
CLASS_1 = []
CLASS_2 = []
CLASS_3 = []
CLASS_4 = []

for element in tqdm(test_data["ID_TARGET"].values):
    CLASS_1.append(additionnal_data.loc[element]["CLASS_LEVEL_1"])
    CLASS_2.append(additionnal_data.loc[element]["CLASS_LEVEL_2"])
    CLASS_3.append(additionnal_data.loc[element]["CLASS_LEVEL_3"])
    CLASS_4.append(additionnal_data.loc[element]["CLASS_LEVEL_4"])

test_data["CLASS_LEVEL_1"] = CLASS_1
test_data["CLASS_LEVEL_2"] = CLASS_2
test_data["CLASS_LEVEL_3"] = CLASS_3
test_data["CLASS_LEVEL_4"] = CLASS_4

test_data = test_data.drop("ID_DAY", axis=1)
test_data = imp.transform(test_data)

test_data = pd.DataFrame(test_data, columns=train_data.columns[0:-1])

y_pred_test = boost.predict(test_data)

sub = pd.DataFrame(y_pred_test)

sub.to_csv("/Users/victorkhamesi/Desktop/submitv3.csv")
