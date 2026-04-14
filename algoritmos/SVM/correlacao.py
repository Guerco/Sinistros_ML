# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:19:05 2026

@author: Yuri Viana
"""

# =============================================================================
#                 Matriz de Correlação
# =============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. base
# =============================================================================

base = pd.read_csv('C:/Users/Yuri Viana/Downloads/acidentes_sp_2014_2025/sinistros_sp_pessoas_2025_balanceado.csv')

print("Base carregada com", base.shape[0], "linhas e", base.shape[1], "colunas")

# =============================================================================
# 2. Matriz de correlação
# =============================================================================

corr = base.corr()

# =============================================================================
# 3. Plot da matriz
# =============================================================================

plt.figure(figsize=(12,10))

sns.heatmap(
    corr,
    vmin=-1,
    vmax=1,
    center=0,
    cmap='coolwarm',
    square=True
)

plt.title("Matriz de Correlação (Pearson)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# =============================================================================
# 4. Correlação com a variavel alvo
# =============================================================================

corr_target = corr['gravidade_lesao'].sort_values(ascending=False)

print("\nCorrelação com a variável alvo (gravidade_lesao):")
print(corr_target)

# =============================================================================
# 5. Identificar variáveis altamente correlacionadas
# =============================================================================

# Matriz absoluta
corr_abs = corr.abs()

# Triângulo superior
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

# Selecionar colunas com correlação alta
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]

print("\nVariáveis altamente correlacionadas (>|0.9|):")
print(to_drop)
