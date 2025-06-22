import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    """
    Normalise-les datasets X_train et X_test
    :return: X_train_scale, X_test_scale
    """
    logger = logging.getLogger(__name__)
    logger.info('Préparation des données : split en train/test')

    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')

    # Initialisation du scaler
    scaler = StandardScaler()

    # Ajustement uniquement sur le jeu d'entraînement
    X_train_scaled = scaler.fit_transform(X_train)

    # Transformation du jeu de test avec les mêmes paramètres
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarde des données normalisées
    pd.DataFrame(X_train_scaled).to_csv('data/processed/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('data/processed/X_test_scaled.csv', index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
