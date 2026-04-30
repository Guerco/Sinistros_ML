# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:52:42 2026

@author: Yuri Viana
"""

import pandas as pd

# ===============================
# carregando base
# ===============================
base = pd.read_csv(
    'C:/Users/Yuri Viana/Downloads/sinistros_sp_pessoas_2014-2025.csv',
    sep=';',
    encoding='latin-1'
)

# Padronizar nomes das colunas
base.columns = base.columns.str.strip().str.lower()

# VERIFICAR COLUNAS
print(base.columns)

# ===============================
# usar apenas coluna alvo
# ===============================
print(base['gravidade_lesao'].value_counts())

# Manter só leve e grave
base = base[base['gravidade_lesao'].isin(['LEVE', 'GRAVE'])]

print("Após filtro:")
print(base['gravidade_lesao'].value_counts())

print("Total de linhas:", len(base))

# ===============================
# separação
# ===============================
previsores = base.drop('gravidade_lesao', axis=1)
classe = base['gravidade_lesao']

# ===============================
# converter categóricos
# ===============================
from sklearn.preprocessing import LabelEncoder

for coluna in previsores.columns:
    if previsores[coluna].dtype == 'object':
        le = LabelEncoder()
        previsores[coluna] = le.fit_transform(previsores[coluna].astype(str))

# classe binária
le_classe = LabelEncoder()
classe = le_classe.fit_transform(classe)

# ===============================
# Tratar NaN
# ===============================
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
previsores = imputer.fit_transform(previsores)

# ===============================
# Train/Test
# ===============================
from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsores, classe, test_size=0.25, random_state=0
)

# ===============================
# Padronização
# ===============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)

print("Configuração OK!")
