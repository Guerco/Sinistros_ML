# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:46:33 2026

@author: BIEL
"""

########################

# Classificador
from sklearn.neighbors import KNeighborsClassifier

# Divisão dos dados em folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

knn_acuracias = []
knn_matrizes = []
knn_metricas = []

for indice_treinamento, indice_teste in kfold.split(
    previsores,
    np.zeros(shape=(previsores.shape[0], 1))
):
    
    # Melhor configuração do modelo
    knn_classificador = KNeighborsClassifier(
        n_neighbors=149,
        metric='minkowski',
        p=2
    )

    # Treinamento
    knn_classificador.fit(
        previsores.iloc[indice_treinamento],
        classe.iloc[indice_treinamento, 0]
    )

    # Teste
    knn_previsoes = knn_classificador.predict(
        previsores.iloc[indice_teste]
    )

    knn_acuracia = accuracy_score(
        classe.iloc[indice_teste, 0],
        knn_previsoes
    )

    knn_metricas.append(
        precision_recall_fscore_support(
            classe.iloc[indice_teste, 0],
            knn_previsoes
        )
    )

    knn_matrizes.append(
        confusion_matrix(
            classe.iloc[indice_teste, 0],
            knn_previsoes
        )
    )

    knn_acuracias.append(knn_acuracia)

################## Resultado Final ####################

# Matriz de confusão média
knn_matriz_media = np.mean(knn_matrizes, axis=0)
knn_matriz_desvio_padrao = np.std(knn_matrizes, axis=0)

# Métricas médias
knn_acuracias = np.asarray(knn_acuracias)
knn_acuracia_final_media = knn_acuracias.mean()
knn_acuracia_final_desvio_padrao = knn_acuracias.std()

knn_metricas_medias = np.mean(knn_metricas, axis=0)
knn_metricas_desvio_padrao = np.std(knn_metricas, axis=0)

# obs:
# cada linha: precisão, recall, f1score
# cada coluna: classe