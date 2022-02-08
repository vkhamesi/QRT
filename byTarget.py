import pandas as pd
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.covariance import oas
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

# Stocker les données du challenge (X_train, y_train, X_test et des données supplémentaires)
train_data = pd.read_csv('X_train.csv', index_col=0)
train_labels = pd.read_csv('y_train.csv', index_col=0)
test_data = pd.read_csv('X_test.csv', index_col=0)
additionnal_data = pd.read_csv('supplementary_data_Vkoyn8z.csv', index_col=0)

# On calcule la matrice des Betas
idx_ret_features = np.where(train_data.columns.str.contains('RET'))[0]
init_ret_features = train_data.columns[idx_ret_features]
target_ret_features = 'RET_' + train_data['ID_TARGET'].astype(str).unique()
returns = {}
for day in tqdm(train_data.ID_DAY.unique()):
    u = train_data.loc[train_data.ID_DAY == day]
    a = u.iloc[0, idx_ret_features]
    b = train_labels[train_data.ID_DAY == day]['RET_TARGET']
    b.index = 'RET_' + u.ID_TARGET.astype(str)
    returns[day] = pd.concat([a, b])
returns = pd.DataFrame(returns).T.astype(float)
features = returns.columns
cov = pd.DataFrame(oas(returns.fillna(0))[0], index=features, columns=features)
beta = cov / np.diag(cov)
targets = train_data["ID_TARGET"].unique().astype(str)
targets = ["RET_" + target for target in targets]
beta = beta.loc[targets][train_data.columns[1:-1]]

# On regroupe dans un DataFrame X_train et y_train
train_data["RET_TARGET"] = train_labels.values

# On impute les données manquantes avec les valeurs moyennes
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_data = imp.fit_transform(train_data)
columns = list(test_data.columns) + ["RET_TARGET"]
train_data = pd.DataFrame(train_data, columns=columns)

# Sandbox (à effacer en cas de problème)
train_data.drop("ID_DAY", axis=1, inplace=True)
beta_rows = []
for target in tqdm(train_data["ID_TARGET"]):
    id_target = "RET_" + str(int(target))
    beta_rows.append(beta.loc[id_target])
betas = pd.concat(beta_rows, axis=1).transpose()
ret_fois_betas = betas.values * train_data.values[:, 0:-2]
train_data[train_data.columns[0:-2]] = ret_fois_betas

test_data.drop("ID_DAY", axis=1, inplace=True)
beta_rows = []
for target in tqdm(test_data["ID_TARGET"]):
    id_target = "RET_" + str(int(target))
    beta_rows.append(beta.loc[id_target])
betas = pd.concat(beta_rows, axis=1).transpose()
ret_fois_betas = betas.values * test_data.values[:, 0:-1]
test_data[test_data.columns[0:-1]] = ret_fois_betas

# On supprime les rangs où la target a un rendement nul (3 rangs)
train_data = train_data[train_data["RET_TARGET"] != 0]

# Séparer les données selon l'ID de la target
train_data_by_target = []
for target, df_target in train_data.groupby("ID_TARGET"):
    train_data_by_target.append(df_target)

# On impute les données manquantes par la moyenne dans les données de test
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
test_data = imp.fit_transform(test_data)

columns = train_data.columns[0:-1]
test_data = pd.DataFrame(test_data, columns=columns)
test_data.index = [267100 + i for i in range(0, 114468)]

# Séparer les données selon l'ID de la target (pour les données de test ici)
test_data_by_target = []
for target, df_target in test_data.groupby("ID_TARGET"):
    test_data_by_target.append(df_target)


# Définition de la métrique du challenge
def weighted_accuracy(y_test, y_pred):
    y_abs = np.abs(y_test)
    norm = y_abs.sum()
    score = ((np.sign(y_pred) == np.sign(y_test)) * y_abs).sum() / norm
    return score


# Créer une fonction de score SciKit Learn
my_func = make_scorer(weighted_accuracy, greater_is_better=True)

# On définit 5 modèles différents à entrainer sur chaque target
model_1 = RandomForestClassifier(max_depth=7)
model_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7))
model_3 = XGBClassifier(gamma=0.2, learning_rate=0.2, max_depth=14,
                        min_child_weight=10)
model_4 = GaussianNB()
model_5 = HistGradientBoostingClassifier(max_depth=14)

reg = LinearRegression()
""" Meilleur score : 74,2 % 
model_1 = RandomForestClassifier(max_depth=7)
model_2 = KNeighborsClassifier(n_neighbors=600)
model_3 = XGBClassifier(gamma=0.2, learning_rate=0.15, max_depth=7,
                        min_child_weight=5)
model_4 = GaussianNB()
model_5 = HistGradientBoostingClassifier(max_depth=7)
"""
scores_by_target = []
cpt = 1
n = len(train_data_by_target)
predictions = []
preds_on_test = pd.DataFrame(columns=["RET_TARGET"], index=test_data.index)

# Pour chaque target
for df in tqdm(train_data_by_target):
    # On split les données
    X_train, X_test, y_train, y_test = train_test_split(df.drop("RET_TARGET", axis=1),
                                                        df["RET_TARGET"],
                                                        test_size=0.5)

    # On scale les données (utile uniquement pour le KNN classifier)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # On entraine le modèle 1 et on calcule son score sur les données de test
    model_1.fit(X_train, np.sign(y_train))
    pred_1 = model_1.predict(X_test)
    score_1 = weighted_accuracy(y_test=y_test, y_pred=pred_1)

    # On entraine le modèle 2 et on calcule son score sur les données de test
    model_2.fit(X_train, np.sign(y_train))
    pred_2 = model_2.predict(X_test)
    score_2 = weighted_accuracy(y_test=y_test, y_pred=pred_2)

    # On entraine le modèle 3 et on calcule son score sur les données de test
    model_3.fit(X_train, np.sign(y_train))
    pred_3 = model_3.predict(X_test)
    score_3 = weighted_accuracy(y_test=y_test, y_pred=pred_3)

    # On entraine le modèle 4 et on calcule son score sur les données de test
    model_4.fit(X_train, np.sign(y_train))
    pred_4 = model_4.predict(X_test)
    score_4 = weighted_accuracy(y_test=y_test, y_pred=pred_4)

    # On entraine le modèle 5 et on calcule son score sur les données de test
    model_5.fit(X_train, np.sign(y_train))
    pred_5 = model_5.predict(X_test)
    score_5 = weighted_accuracy(y_test=y_test, y_pred=pred_5)

    # On établit un 6ème modèle qui est un vote entre les 5 modèles précédents
    pred_voting = np.sign((1/5) * (pred_1 + pred_2 + pred_3 + pred_4 + pred_5))
    score_voting = weighted_accuracy(y_test=y_test, y_pred=pred_voting)

    # On entraine la régression linéaire
    reg.fit(X_train, y_train)
    pred_6 = reg.predict(X_test)
    score_6 = weighted_accuracy(y_test=y_test, y_pred=pred_6)

    # On stocke les scores de chaque modèle
    scores = np.array([score_1, score_2, score_3, score_4, score_5, score_voting, score_6])
    scores_by_target.append(scores)

    # On crée le DataFrame à submit
    test = test_data_by_target[cpt-1]
    id_model = scores.argmax()
    ind = test.index
    test_scaled = scaler.transform(test)

    # Pour chaque target, on choisit le modèle le plus adapté
    if id_model == 0:
        preds_on_test.loc[ind] = pd.DataFrame(model_1.predict(test), index=ind)
    elif id_model == 1:
        preds_on_test.loc[ind] = pd.DataFrame(model_2.predict(test_scaled), index=ind)
    elif id_model == 2:
        preds_on_test.loc[ind] = pd.DataFrame(model_3.predict(test), index=ind)
    elif id_model == 3:
        preds_on_test.loc[ind] = pd.DataFrame(model_4.predict(test), index=ind)
    elif id_model == 4:
        preds_on_test.loc[ind] = pd.DataFrame(model_5.predict(test), index=ind)
    elif id_model == 5:
        preds_on_test.loc[ind] = pd.DataFrame(np.sign(0.2*(model_1.predict(test) + model_2.predict(test_scaled) + model_3.predict(test) +
                                              model_4.predict(test) + model_5.predict(test))), index=ind)
    elif id_model == 6:
        preds_on_test.loc[ind] = pd.DataFrame(reg.predict(test), index=ind)

    cpt += 1

scores_by_target = np.array(scores_by_target)

expected_scores = []
for i in range(len(scores_by_target)):
    expected_scores.append(scores_by_target[i].max())
expected_scores = np.array(expected_scores)
print("Expected score on test set :", expected_scores.mean())

# Cross Validation (à développer pour améliorer les performances)
depths = [4, 5, 7, 9]
gammas = [.1, .15, .2]
learning_rates = [.1, .15, .2]
scores_by_model = []
for depth in depths:
    for gamma in gammas:
        for learning_rate in learning_rates:
            model_1 = RandomForestClassifier(max_depth=depth)
            model_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
            model_3 = XGBClassifier(gamma=gamma, learning_rate=learning_rate, max_depth=depth,
                                    min_child_weight=5)
            model_4 = GaussianNB()
            model_5 = HistGradientBoostingClassifier(max_depth=depth)
            reg = LinearRegression()

            scores_by_target = []
            cpt = 1
            n = len(train_data_by_target)
            predictions = []

            for df in tqdm(train_data_by_target):
                X_train, X_test, y_train, y_test = train_test_split(df.drop("RET_TARGET", axis=1),
                                                                    df["RET_TARGET"],
                                                                    test_size=0.2, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                score_cv_1 = cross_val_score(model_1, X_train, np.sign(y_train), cv=3, scoring=my_func)
                print(score_cv_1.mean())

                score_cv_2 = cross_val_score(model_2, X_train_scaled, np.sign(y_train), cv=3, scoring=my_func)
                print(score_cv_2.mean())

                score_cv_3 = cross_val_score(model_3, X_train, np.sign(y_train), cv=3, scoring=my_func)
                print(score_cv_3.mean())

                score_cv_4 = cross_val_score(model_4, X_train, np.sign(y_train), cv=3, scoring=my_func)
                print(score_cv_4.mean())

                score_cv_5 = cross_val_score(model_5, X_train, np.sign(y_train), cv=3, scoring=my_func)
                print(score_cv_4.mean())

                reg.fit(X_train, y_train)
                score_cv_7 = weighted_accuracy(y_test, reg.predict(X_test))

                scores = np.array([score_cv_1.mean(), score_cv_2.mean(), score_cv_3.mean(),
                                   score_cv_4.mean(), score_cv_5.mean(), score_cv_7])
                scores_by_target.append(scores)
                cpt += 1

            scores_by_target = np.array(scores_by_target)
            expected_scores = []
            for i in range(len(scores_by_target)):
                expected_scores.append(scores_by_target[i].max())
            expected_scores = np.array(expected_scores)

            scores_by_model.append([[depth, gamma, learning_rate], expected_scores.mean()])

"""
Best hyperparameters are :
depth=7, gamma=0.2, learning_rate=0.15
"""
