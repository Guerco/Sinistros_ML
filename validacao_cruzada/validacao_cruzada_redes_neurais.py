# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:15:59 2026

@author: BIEL
"""

########################

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:46:33 2026

@author: BIEL
"""

########################

# Classificador
from sklearn.neural_network import MLPClassifier

# Divisão dos dados em folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

mlp_acuracias = []
mlp_matrizes = []
mlp_metricas = []

for indice_treinamento, indice_teste in kfold.split(
    previsores,
    np.zeros(shape=(previsores.shape[0], 1))
):

    # Melhor configuração do modelo
    mlp_classificador = MLPClassifier(
        verbose=True,
        max_iter=1000,
        tol=0.000001,
        solver='adam',
        hidden_layer_sizes=[10],
        activation='relu',
        random_state=1
    )

    # Treinamento
    mlp_classificador.fit(
        previsores.iloc[indice_treinamento],
        classe.iloc[indice_treinamento, 0]
    )

    # Teste
    mlp_previsoes = mlp_classificador.predict(
        previsores.iloc[indice_teste]
    )

    mlp_acuracia = accuracy_score(
        classe.iloc[indice_teste, 0],
        mlp_previsoes
    )

    mlp_metricas.append(
        precision_recall_fscore_support(
            classe.iloc[indice_teste, 0],
            mlp_previsoes
        )
    )

    mlp_matrizes.append(
        confusion_matrix(
            classe.iloc[indice_teste, 0],
            mlp_previsoes
        )
    )

    mlp_acuracias.append(mlp_acuracia)

################## Resultado Final ####################

# Matriz de confusão média
mlp_matriz_media = np.mean(mlp_matrizes, axis=0)
mlp_matriz_desvio_padrao = np.std(mlp_matrizes, axis=0)

# Métricas médias
mlp_acuracias = np.asarray(mlp_acuracias)
mlp_acuracia_final_media = mlp_acuracias.mean()
mlp_acuracia_final_desvio_padrao = mlp_acuracias.std()

mlp_metricas_medias = np.mean(mlp_metricas, axis=0)
mlp_metricas_desvio_padrao = np.std(mlp_metricas, axis=0)

# obs:
# cada linha: precisão, recall, f1score
# cada coluna: classe