import csv
import pandas as pd


def read_dataset(path):

    # Lee la primera línea del archivo CSV para obtener los nombres de las columnas
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)[0]

    # Divide la cadena de encabezados en una lista de nombres de columnas
    column_names = headers.split(',')

    # Lee el archivo CSV con los nombres de columnas obtenidos
    df = pd.read_csv(path)
    return df


