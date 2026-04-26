# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 2026

@author: Yuri Viana
"""

########################

# Dados
from pre_processamento import previsores, classe

# Classificador
from sklearn.svm import SVC

# Validação cruzada
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Balanceamento
from imblearn.under_sampling import RandomUnderSampler

import numpy as np

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

svm_acuracias = []
svm_matrizes = []
svm_metricas = []

for indice_treinamento, indice_teste in kfold.split(
    previsores,
    np.zeros(shape=(previsores.shape[0], 1))
):

    # ===============================
    # Separar treino e teste
    # ===============================
    X_train = previsores.iloc[indice_treinamento]
    y_train = classe.iloc[indice_treinamento, 0]

    X_test = previsores.iloc[indice_teste]
    y_test = classe.iloc[indice_teste, 0]

    # ===============================
    # BALANCEAMENTO CORRETO
    # (APENAS NO TREINO)
    # ===============================
    rus = RandomUnderSampler(random_state=0)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # ===============================
    # Modelo SVM
    # ===============================
    svm_classificador = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=1
    )

    # Treinamento
    svm_classificador.fit(X_train, y_train)

    # Teste
    svm_previsoes = svm_classificador.predict(X_test)

    # ===============================
    # Métricas
    # ===============================
    svm_acuracia = accuracy_score(y_test, svm_previsoes)

    svm_metricas.append(
        precision_recall_fscore_support(
            y_test,
            svm_previsoes,
            zero_division=0
        )
    )

    svm_matrizes.append(
        confusion_matrix(
            y_test,
            svm_previsoes
        )
    )

    svm_acuracias.append(svm_acuracia)

################## Resultado Final ####################

svm_matriz_media = np.mean(svm_matrizes, axis=0)
svm_matriz_desvio_padrao = np.std(svm_matrizes, axis=0)

svm_acuracias = np.asarray(svm_acuracias)
svm_acuracia_final_media = svm_acuracias.mean()
svm_acuracia_final_desvio_padrao = svm_acuracias.std()

svm_metricas_medias = np.mean(svm_metricas, axis=0)
svm_metricas_desvio_padrao = np.std(svm_metricas, axis=0)

# obs:
# cada linha: precisão, recall, f1score
# cada coluna: classe

