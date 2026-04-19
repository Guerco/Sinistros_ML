# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:55:23 2026

@author: Yuri Viana
"""


from configuracao import *

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

classificador = SVC(
    kernel='rbf',
    C=5,
    gamma='scale',
    random_state=0
)

# Treino
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste
previsoes = classificador.predict(previsores_teste)

# Resultados
acuracia = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

print("Acurácia:", acuracia)
print("Matriz de confusão:")
print(matriz)