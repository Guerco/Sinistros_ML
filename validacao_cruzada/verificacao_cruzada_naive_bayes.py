# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 2026

@author: Yuri Viana
"""

########################

# Classificador
from pre_processamento import previsores, classe
from sklearn.naive_bayes import GaussianNB

# Divisão dos dados em folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

nb_acuracias = []
nb_matrizes = []
nb_metricas = []

for indice_treinamento, indice_teste in kfold.split(
    previsores,
    np.zeros(shape=(previsores.shape[0], 1))
):

    # Modelo
    nb_classificador = GaussianNB()

    # Treinamento
    nb_classificador.fit(
        previsores.iloc[indice_treinamento],
        classe.iloc[indice_treinamento, 0]
    )

    # Teste
    nb_previsoes = nb_classificador.predict(
        previsores.iloc[indice_teste]
    )

    nb_acuracia = accuracy_score(
        classe.iloc[indice_teste, 0],
        nb_previsoes
    )

    nb_metricas.append(
        precision_recall_fscore_support(
            classe.iloc[indice_teste, 0],
            nb_previsoes
        )
    )

    nb_matrizes.append(
        confusion_matrix(
            classe.iloc[indice_teste, 0],
            nb_previsoes
        )
    )

    nb_acuracias.append(nb_acuracia)

################## Resultado Final ####################

nb_matriz_media = np.mean(nb_matrizes, axis=0)
nb_matriz_desvio_padrao = np.std(nb_matrizes, axis=0)

nb_acuracias = np.asarray(nb_acuracias)
nb_acuracia_final_media = nb_acuracias.mean()
nb_acuracia_final_desvio_padrao = nb_acuracias.std()

nb_metricas_medias = np.mean(nb_metricas, axis=0)
nb_metricas_desvio_padrao = np.std(nb_metricas, axis=0)

# obs:
# cada linha: precisão, recall, f1score
# cada coluna: classe