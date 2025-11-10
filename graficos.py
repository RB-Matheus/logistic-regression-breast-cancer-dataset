import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression


def plot_confusion_matrix(cm, modelo): #classes é pega do modelo, ao que parece

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=modelo.classes_)
    disp.plot()
    plt.show()
    
    
def plot_roc_curve(modelo, X_test, y_test):
    svc_disp = RocCurveDisplay.from_estimator(modelo, X_test, y_test)
    plt.show()


def plot_precision_recall_curve(clf, X_test, y_test):
    
    LogisticRegression()
    PrecisionRecallDisplay.from_estimator(
    clf, X_test, y_test)
    plt.show()
    
def plot_pesos(modelo, X_treino):

    best_model = modelo.named_steps['reglog']

    pesos = best_model.coef_[0]
    feature_names = X_treino 

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, pesos)
    plt.title("Pesos das features da Regressão Logística (melhor modelo do GridSearch)")
    plt.xlabel("Peso (coeficiente)")
    plt.ylabel("Feature")
    plt.show()
    
def plot_desvio(resultados):#bizarro!!!!
    df_plot = resultados[['param_reglog__C', 'mean_test_score', 'std_test_score']]

    plt.figure(figsize=(8,5))
    plt.errorbar(df_plot['param_reglog__C'],
                df_plot['mean_test_score'],
                yerr=df_plot['std_test_score'],
                fmt='-o', capsize=5)
    plt.title('Média e Desvio Padrão da Acurácia por valor de C')
    plt.xlabel('Valor de C')
    plt.ylabel('Acurácia média (± desvio padrão)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()