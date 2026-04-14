# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:03:33 2026

@author: Yuri Viana
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Base
# =============================================================================

base = pd.read_csv(
    'C:/Users/Yuri Viana/Downloads/acidentes_sp_2014_2025/sinistros_sp_pessoas_2025_balanceado.csv'
)

X = base.drop('gravidade_lesao', axis=1)
y = base['gravidade_lesao']

# =============================================================================
# 2. Separação treino/teste (75%/25%)
# =============================================================================

from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=69
)

# =============================================================================
# 3. Padronização
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

# =============================================================================
# 4. testando diferentes configurações
# =============================================================================

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

valores_C = [0.1, 1, 10]
valores_gamma = ['scale', 0.1, 0.01]

melhor_acuracia = 0
melhor_modelo = None
melhores_parametros = None

print("\n===== Testando... =====")

for c in valores_C:
    for g in valores_gamma:

        modelo = SVC(kernel='rbf', C=c, gamma=g)
        modelo.fit(X_treino, y_treino)

        y_pred = modelo.predict(X_teste)
        acc = accuracy_score(y_teste, y_pred)

        print(f"C={c}, gamma={g} → Acurácia: {acc:.6f}")

        if acc > melhor_acuracia:
            melhor_acuracia = acc
            melhor_modelo = modelo
            melhores_parametros = (c, g)

# =============================================================================
# 5. Resultado final (funciona pfv)
# =============================================================================

from sklearn.metrics import confusion_matrix

print("\n===== Melhor modelo =====")
print(f"C={melhores_parametros[0]}, gamma={melhores_parametros[1]}")
print(f"Acurácia: {melhor_acuracia:.6f}")

# Matriz de confusão final
y_pred_final = melhor_modelo.predict(X_teste)
matriz = confusion_matrix(y_teste, y_pred_final)

print("\nMatriz de Confusão:")
print(f"{matriz[0][0]}\t{matriz[0][1]}")
print(f"{matriz[1][0]}\t{matriz[1][1]}")

# =============================================================================
# 6. Matriz
# =============================================================================

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot()
plt.title("Matriz de Confusão - Melhor Modelo")
plt.show()