from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

pd.set_option('display.max_colwidth', None)
dados = datasets.load_breast_cancer()

X = dados.data
y = dados.target

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=42)

modelo_parametros = {
    'reglog': {
        'pipeline': Pipeline([
            ('padronizador', StandardScaler()), 
            #('balanceador', SMOTE(random_state=42, k_neighbors=5)), 
            ('reglog', LogisticRegression(random_state=42))
        ]),
        'parametros': {
            # C = 1, l1_raio = 0.25, max_iter = 100, penalty = l2, solver = saga [sem smote foi melhor]
            # C = 10, max_iter = 175, penalty = l2, solver = liblinear (aceita l1 e l2)
            # C = 5, max_iter = 250, penalty = l2, solver = newton-cg (só aceita l2)
            # C = 1, max_iter = 250, penalty = l2, solver = sag (só aceita l2)
            # saga (demorou muito para convergir), depende de elastic_net, que depende de l1_ratio
            # C = 1, max_iter = 250, penalty = l2, solver = lbfgs [sem smote foi melhor]
            # C = 1, max_iter = 250, penalty = l2, solver = newton-cholesky (só aceita l2)
            'reglog__penalty': ['l2', 'l1', 'elasticnet'],
            'reglog__l1_ratio': [0.25, 0.15, 0.05],
            'reglog__C': [1, 2, 5, 100],
            'reglog__solver': ['saga'],
            'reglog__max_iter': [50, 100]
        }
    }
}

relatorios = []
for nome_modelo, configs_modelo in modelo_parametros.items():
    grid = GridSearchCV(configs_modelo['pipeline'], configs_modelo['parametros'], refit=True, cv=5, verbose=0, return_train_score=False)
    grid.fit(X_treino, y_treino)
    y_previstos = grid.predict(X_teste)
    relatorio_classificacao = classification_report(y_teste, y_previstos)
    matriz_confusao = confusion_matrix(y_teste, y_previstos)
    indice_melhor_configuracao = grid.best_index_
    relatorios.append({
        'modelo': nome_modelo,
        #'escore': grid.best_score_,
        'media_escores_melhor_configuracao': grid.cv_results_['mean_test_score'][indice_melhor_configuracao],
        'desvio_padrao_escores_melhor_configuracao': grid.cv_results_['std_test_score'][indice_melhor_configuracao],
        'hiperparametros_melhor_configuracao': grid.best_params_,
        #'previsoes_teste': y_previstos,
        'relatorio_classificacao': relatorio_classificacao,
        'matriz_confusao': matriz_confusao
    })

print('RELATÓRIO GERAL\n')
df = pd.DataFrame(relatorios, columns=['modelo', 'media_escores_melhor_configuracao', 'desvio_padrao_escores_melhor_configuracao'])
print(f'{df}\n\n')

print('MELHOR CONFIGURAÇÃO DE HIPERPARÂMETROS POR MODELO\n')
melhores_configuracoes = pd.DataFrame(relatorios, columns=['modelo', 'hiperparametros_melhor_configuracao'])
print(f'{melhores_configuracoes}\n\n')

print('-'*120)
for relatorio in relatorios:
    print(f'MODELO: {relatorio['modelo']}')
    #print(f'\nPrevisões do teste:\n {relatorio["previsoes_teste"]}')
    print(f'\nMATRIZ DE CONFUSÃO:\n\n{relatorio['matriz_confusao']}')
    print(f'\nRELATÓRIO DE CLASSIFICAÇÃO:\n\n{relatorio['relatorio_classificacao']}')
    print('-'*120)

#prints novos usando graficos abaixo

import graficos

#graficos.plot_confusion_matrix(matriz_confusao, grid.best_estimator_)

#graficos.plot_pesos(grid.best_estimator_, dados.feature_names)

#graficos.plot_precision_recall_curve(grid.best_estimator_, X_teste, y_teste)

#graficos.plot_roc_curve(grid.best_estimator_, X_teste, y_teste)

#graficos.plot_desvio(pd.DataFrame(grid.cv_results_)) #bizarro!!!!