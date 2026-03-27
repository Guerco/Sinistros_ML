# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:00:58 2026

@author: BIEL
"""

# ===========================================================
#                 Classificação por KNN
# ===========================================================


from sklearn.neighbors import KNeighborsClassifier

# n_neighbours é o valor de K
# Minkowski com p=2 = Distância Euclidiana

_valorK = 160

_classificador = KNeighborsClassifier(n_neighbors=_valorK, metric='minkowski', p=2)

# Treinando
_classificador.fit(previsores_treinamento, classe_treinamento)

# Testando
previsoes = _classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score
test_score = round(accuracy_score(classe_teste, previsoes), 5)
test_matrix = confusion_matrix(classe_teste, previsoes)






# ===========================================================
#                 Gráficos
# ===========================================================


#  Matriz de Confusão

from sklearn.metrics import ConfusionMatrixDisplay

_labels = [0, 1]
_nomes = ['Grave', 'Leve']

_disp = ConfusionMatrixDisplay(confusion_matrix=test_matrix, display_labels=_nomes)
_disp.plot()



#  ??

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

_pca = PCA(n_components=2)
X_reduzido = pca.fit_transform(previsores_teste)

plt.scatter(X_reduzido[:,0], X_reduzido[:,1], c=previsoes)
plt.title("Visualização 2D (PCA) - KNN")
plt.show()