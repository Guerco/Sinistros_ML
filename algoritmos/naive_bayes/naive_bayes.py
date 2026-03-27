# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:22:25 2026

@author: BIEL
"""

# ===========================================================
#               Classificação com Naive Bayes
# ===========================================================


from sklearn.naive_bayes import GaussianNB
_classificador = GaussianNB()

# Treinando (automaticamente o sklearn já faz a correlação laplaciana)
_classificador.fit(previsores_treinamento, classe_treinamento)

# Testando
previsoes = _classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score
test_score = round(accuracy_score(classe_teste, previsoes), 5)
test_matrix = confusion_matrix(classe_teste, previsoes)





# Resultados na base de treinamento, para verificar overfitting
# previsoes_treinamento = classificador.predict(previsores_treinamento)
# acuracia_treinamento = accuracy_score(classe_treinamento, previsoes_treinamento)
# matriz_treinamento = confusion_matrix(classe_treinamento, previsoes_treinamento)