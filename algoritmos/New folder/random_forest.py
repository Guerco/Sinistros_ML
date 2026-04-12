# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:37:09 2026

@author: joaov
"""


import pandas as pd

from sklearn.ensemble import RandomForestClassifier

classificador = RandomForestClassifier(n_estimators=300,
                                       max_depth=15,
                                       criterion='gini',
                                       random_state=0
                                       )
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score

acuracia_teste = round(accuracy_score(classe_teste, previsoes),5)
matriz_confusao = confusion_matrix(classe_teste, previsoes)

#visualizando importancia

# import matplotlib.pyplot as plt
# import numpy as np

# n_features = previsores.columns.size 
# plt.figure()
# plt.barh(range(n_features), classificador.feature_importances_,align='center')
# plt.yticks(np.arange(n_features),previsores.columns)
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.show()
