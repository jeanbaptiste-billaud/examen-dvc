import logging
import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_folderpath', type=click.Path())
def main(input_filepath, output_folderpath):
    """Prépare les données à partir de raw.csv et les écrit dans data/processed/"""
    logger = logging.getLogger(__name__)
    logger.info('Préparation des données : split en train/test')

    # Chargement du CSV
    df_raw = pd.read_csv(input_filepath)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_raw)

    # Création du dossier de sortie si nécessaire
    os.makedirs(output_folderpath, exist_ok=True)

    # Sauvegarde des datasets
    save_dataframes(X_train, X_test, y_train, y_test, output_folderpath)


def split_data(df):
    # Séparation des features et de la cible
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate', 'date'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Sauvegarde des splits
    for file, name in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_file = os.path.join(output_folderpath, f'{name}.csv')
        file.to_csv(output_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
