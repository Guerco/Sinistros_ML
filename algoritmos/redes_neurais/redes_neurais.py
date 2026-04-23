# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:13:31 2026

@author: BIEL
"""

import pandas as pd

# ===========================================================
#               Classificação com Redes Neurais
# ===========================================================

# Treinando
from sklearn.neural_network import MLPClassifier

_classificador =    MLPClassifier(
                        verbose = True,
                        max_iter = 1000,
                        tol = 0.000001,
                        solver = 'adam',
                        hidden_layer_sizes = [10],
                        activation = 'relu',
                        random_state = 1
                    ) 

_classificador.fit(previsores_treinamento, classe_treinamento)





# Testando
_previsoes = _classificador.predict(previsores_teste)

# Análise dos Resultados
from sklearn.metrics import confusion_matrix, accuracy_score

acuracia_teste = round(accuracy_score(classe_teste, _previsoes), 5)
matriz_teste = confusion_matrix(classe_teste, _previsoes)




# Resultados na base treinamento
_previsoes_treinamento = _classificador.predict(previsores_treinamento)
                                               
acuracia_treinamento = round(accuracy_score(classe_treinamento, _previsoes_treinamento), 5)
matriz_treinamento = confusion_matrix(classe_treinamento, _previsoes_treinamento)                                               



