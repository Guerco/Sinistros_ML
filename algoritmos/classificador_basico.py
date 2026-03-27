# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:04:34 2026

@author: BIEL
"""

# ===========================================================
#           Classificação por Classe Majoritária
# ===========================================================

# Resultado Mínimo
_contagem = classe_treinamento['gravidade_lesao'].value_counts()
_classe_majoritaria = _contagem.idxmax()

previsoes = pd.Series([_classe_majoritaria]).repeat(classe_teste.size)

# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score
test_score = round(accuracy_score(classe_teste, previsoes), 5)
test_matrix = confusion_matrix(classe_teste, previsoes)