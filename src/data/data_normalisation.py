import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')

# Initialisation du scaler
scaler = StandardScaler()

# Ajustement uniquement sur le jeu d'entraînement
X_train_scaled = scaler.fit_transform(X_train)

# Transformation du jeu de test avec les mêmes paramètres
X_test_scaled = scaler.transform(X_test)

# Sauvegarde des données normalisées
X_train_scaled.to_csv('data/preprocessed/X_train_scale.csv', index=False)
X_test_scaled.to_csv('data/preprocessed/X_test_scale.csv', index=False)
