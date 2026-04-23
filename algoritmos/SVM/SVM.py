# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:55:23 2026

@author: Yuri Viana
"""


import pandas as pd

################## Classificação com SVM ################## 

# Treinamento 
from sklearn.svm import SVC

_classificador =    SVC(
                        kernel = 'rbf', 
                        C = 1.0, 
                        gamma = 'scale', 
                        random_state = 1,
                        verbose = True
                    )

_classificador.fit(previsores_treinamento, classe_treinamento)

# Teste 
_previsoes = _classificador.predict(previsores_teste)




# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score

acuracia_teste = round(accuracy_score(classe_teste, _previsoes), 5)
matriz_teste = confusion_matrix(classe_teste, _previsoes)




# Resultados na base de treinamento, para verificar overfitting
_previsoes_treinamento = _classificador.predict(previsores_treinamento)
acuracia_treinamento = accuracy_score(classe_treinamento, _previsoes_treinamento)
matriz_treinamento = confusion_matrix(classe_treinamento, _previsoes_treinamento)
