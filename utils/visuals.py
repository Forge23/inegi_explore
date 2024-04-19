import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import utils.processing as proc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os

def save_confusion_matrix(y_true, y_pred, target_names):
    # checar folder de salida
    proc.check_output_folder("output/matrix")
    print(classification_report(y_true, y_pred, target_names=target_names))
    confusion = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, fmt='d')

    # salvar figura como confusion_matrix.png
    plt.title("Matriz de Confusión")
    plt.savefig(f"output/matrix/confusion_matrix.png")
    plt.clf()
    plt.close()

def save_roc_curve(diabetes_y_test, diabetes_y_pred):
    # checar folder de salida
    proc.check_output_folder("output")
    # generar curva ROC
    new_fig = plt.figure()
    metrics.RocCurveDisplay.from_predictions(diabetes_y_test, diabetes_y_pred)
    # salvarla como curve_ROC.png
    plt.savefig("output/curve_ROC.png")
    plt.close(new_fig)

def save_histogram(data, column,hue_col=None):
    proc.check_output_folder("output/histograms")
    sns_plot = sns.histplot(x=column, data=data, hue=hue_col, kde=True)
    fig = sns_plot.get_figure()
    sns_plot.set(title=f"{column}")
    fig.savefig("output/histograms/histogram_" + column + ".png")
    plt.close()


def save_histograms(data,hue_col=None):
    proc.check_output_folder("output/histograms")
    for column in data.columns:
        save_histogram(data, column,hue_col)


def save_correlation(data, var1, var2):
    # Verificar si la carpeta "output" existe, si no, crearla
    proc.check_output_folder("output/scatterplots")

    # Calcular la correlación entre las dos variables
    correlation_value = data[var1].corr(data[var2])

    # Crear el gráfico de correlación
    sns.scatterplot(data=data, x=var1, y=var2)

    # Añadir el valor de correlación como parte del título
    plt.title(f"Correlación entre {var1} y {var2}: {correlation_value}")
    plt.xlabel(var1)
    plt.ylabel(var2)

    # Guardar el gráfico como imagen PNG
    plt.savefig(f"output/scatterplots/{var1}_{var2}_correlation_plot.png")

    # Limpiar la figura
    plt.clf()
    plt.close()

def save_specific_correlations_individual(data, columns, hue_col):
 
    correlations = data[columns].corr()
    

    correlations.to_csv("output/correlations/all_correlations.csv")
    
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            filename = f"output/scatterplots/{col1}_{col2}_correlation_plot.png"
            fig = plt.figure()
            sns.scatterplot(data=data, x=col1, y=col2, hue=hue_col)
            plt.savefig(filename)
            plt.close()

def save_all_correlations_one_image(data):
    proc.check_output_folder("output")
    fig = sns.pairplot(data, hue="radius_mean")
    fig.savefig("output/All_histograms.png")
    plt.close()
    pass

def save_specific_correlations_one_image(data, columns):
    # Create a pair plot for the specified columns
    fig = sns.pairplot(data[columns], hue="diagnosis_num")
    plt.savefig("output/All_histograms.png")
    plt.close()
    
  
def save_all_correlations(data, correlations):
    #checar si la carpeta output existe, si no, crearla
    proc.check_output_folder("output/correlations")
    #guardar todas las correlaciones en un solo archivo
    correlations.to_csv("output/correlations/all_correlations.csv")
    #crear el grafcio de correlaciones
    fig = px.imshow(correlations)
    #guardar todas las correlaciones en imágenes individuales png
    #Es importante evitar cifras duplicadas, la correlación A vs B es la misma que la correlación B vs A
    for i in range(correlations.shape[0]):
        for j in range(i, correlations.shape[1]):
            if i != j:
                save_correlation(data, correlations.columns[i], correlations.columns[j])

def save_specific_correlations(data, columns):
    #checar si la carpeta output existe, si no, crearla
    proc.check_output_folder("output/correlations")
    
    # Calcular las correlaciones para las columnas en específico
    correlations = data[columns].corr()
    
    #guardar las correlaciones en un solo archivo
    correlations.to_csv("output/correlations/specific_correlations.csv")

    # Crear el gráfico
    fig = px.imshow(correlations)
    
    #guardar todas las correlaciones en imágenes individuales png
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            save_correlation(data, columns[i], columns[j])


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    # count how many annotations are already present
    n = len([c for c in ax.get_children() if 
                  isinstance(c, matplotlib.text.Annotation)])
    pos = (.1, .9 - .1*n)
    # or make positions for every label by hand
    pos = (.1, .9) if kws['label'] == 'Yes' else (.1,.8)

    ax.annotate("{}: r = {:.2f}".format(kws['label'],r),
                xy=pos, xycoords=ax.transAxes)