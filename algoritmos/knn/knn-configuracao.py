# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:37:50 2026

@author: BIEL
"""

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

_training_acuracy = []
_test_acuracy = []

# Testando um range de valores pra K

# _neighbours_settings = range(1,51)
_neighbours_settings = range(1, 251, 2)

for _k in _neighbours_settings:
    # Construindo o modelo
    _classificador = KNeighborsClassifier(n_neighbors=_k, metric='minkowski', p=2)
    
    # Treinando
    _classificador.fit(previsores_treinamento, classe_treinamento)
    
    # Gravando o resultado para os dados de treinamento
    _training_acuracy.append(_classificador.score(
            previsores_treinamento, classe_treinamento
        ))
    
    # Gravando o resultado para os dados de treinamento
    _test_acuracy.append(_classificador.score(
            previsores_teste, classe_teste
        ))
    
    
plt.plot(_neighbours_settings, _training_acuracy, label="Acurácia no Treinamento")
plt.plot(_neighbours_settings, _test_acuracy, label="Acurácia no Teste")   
plt.ylabel("Acurácia")
plt.xlabel("N_Vizinhos")    
plt.legend
    