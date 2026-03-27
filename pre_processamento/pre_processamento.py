# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:08:55 2026

@author: BIEL
"""

import pandas as pd

# =============================================================================
#                              Carregando a base
# =============================================================================

base = pd.read_csv('./pre_processamento/sinistros_sp_pessoas_2014-2025.csv', sep=';', encoding='latin-1')


_colunas_selecionadas = [
    'regiao_administrativa',
    'tipo_via',
    'tipo_veiculo_vitima',
    'tipo_de_vitima',
    'sexo',
    # 'faixa_etaria_legal',
    'idade',
    'profissao',
    'nacionalidade',
    'ano_sinistro',
    'mes_sinistro',
    'dia_sinistro',
    'gravidade_lesao'
]

base = base[_colunas_selecionadas]

# print(base.info())
# print(base.head())





# =============================================================================
#                     Tratando valores inválidos
# =============================================================================


#       Procurando as colunas que possuem algum valor faltante
pd.isnull(base).any()
base.isnull().sum()

base_tratamento = base.copy()

base_tratamento.drop(base_tratamento[base_tratamento['sexo'] == 'NAO DISPONIVEL'].index, inplace=True)

base_tratamento.drop(base_tratamento[base_tratamento['tipo_de_vitima'] == 'NAO DISPONIVEL'].index, inplace=True)
base_tratamento.drop(base_tratamento[base_tratamento['tipo_de_vitima'].isnull()].index, inplace=True)

base_tratamento.loc[base_tratamento['tipo_de_vitima'] == 'PEDESTRE', 'tipo_veiculo_vitima'] = 'NENHUM'
base_tratamento.drop(base_tratamento[base_tratamento['tipo_veiculo_vitima'].isnull()].index, inplace=True)

base_tratamento.drop(base_tratamento[base_tratamento['tipo_via'] == 'NAO DISPONIVEL'].index, inplace=True)

# base_tratamento.drop(base_tratamento[base_tratamento['faixa_etaria_legal'].isnull()].index, inplace=True)
# base_tratamento.drop(base_tratamento[base_tratamento['faixa_etaria_legal'] == 'NAO DISPONIVEL'].index, inplace=True)
base_tratamento.drop(base_tratamento[base_tratamento['idade'].isnull()].index, inplace=True)

base_tratamento.drop(base_tratamento[base_tratamento['profissao'].isin(['Nao informada', 'NÃO INFORMADO'])].index, inplace=True)
base_tratamento.drop(base_tratamento[base_tratamento['profissao'].isnull()].index, inplace=True)

base_tratamento.drop(base_tratamento[base_tratamento['nacionalidade'].isnull()].index, inplace=True)

base_tratamento.loc[base_tratamento['gravidade_lesao'] == 'FATAL', 'gravidade_lesao'] = 'GRAVE'
base_tratamento.drop(base_tratamento[base_tratamento['gravidade_lesao'] == 'NAO DISPONIVEL'].index, inplace=True)
base_tratamento['gravidade_lesao'].value_counts()

base_tratamento['nacionalidade'] = base_tratamento['nacionalidade'].str.strip()
base_tratamento.loc[base_tratamento['nacionalidade'] != 'BRASILEIRA', 'nacionalidade'] = 'ESTRANGEIRA'

# =============================================================================
#                     Separando dados em previsores e classe
# =============================================================================

_cols_previsores = [
    'regiao_administrativa',
    'tipo_via',
    'tipo_veiculo_vitima',
    'tipo_de_vitima',
    'sexo',
    # 'faixa_etaria_legal',
    'idade',
    'profissao',
    'nacionalidade',
    'ano_sinistro',
    'mes_sinistro',
    'dia_sinistro'
]
_cols_classe = ['gravidade_lesao']

previsores = base_tratamento[_cols_previsores].copy()
classe = base_tratamento[_cols_classe].copy()

# Após separar previsores e classe, resete o índice
previsores = previsores.reset_index(drop=True)
classe = classe.reset_index(drop=True)




# =============================================================================
#      Transformando as variáveis categóricas binárias em valores numéricos
# =============================================================================


from sklearn.preprocessing import LabelEncoder
# import numpy as np

_le_sexo            = LabelEncoder()
_le_gravidade       = LabelEncoder()
_le_via             = LabelEncoder()
_le_nacionalidade   = LabelEncoder()
_le_tp_vitima       = LabelEncoder()
_le_tp_profissao    = LabelEncoder()
_le_faixa_legal     = LabelEncoder()
# _le_tp_regiao       = LabelEncoder()

previsores.loc[:, 'sexo'] = _le_sexo.fit_transform(previsores.loc[:, 'sexo'])
previsores['sexo'] = previsores['sexo'].astype('int64')

previsores.loc[:, 'tipo_via'] = _le_via.fit_transform(previsores.loc[:, 'tipo_via'])
previsores['tipo_via'] = previsores['tipo_via'].astype('int64')

previsores.loc[:, 'nacionalidade'] = _le_nacionalidade.fit_transform(previsores.loc[:, 'nacionalidade'])
previsores['nacionalidade'] = previsores['nacionalidade'].astype('int64')

previsores.loc[:, 'tipo_de_vitima'] = _le_tp_vitima.fit_transform(previsores.loc[:, 'tipo_de_vitima'])
previsores['tipo_de_vitima'] = previsores['tipo_de_vitima'].astype('int64')

previsores.loc[:, 'profissao'] = _le_tp_profissao.fit_transform(previsores.loc[:, 'profissao'])
previsores['profissao'] = previsores['profissao'].astype('int64')

# previsores.loc[:, 'regiao_administrativa'] = _le_tp_regiao.fit_transform(previsores.loc[:, 'regiao_administrativa'])
# previsores['regiao_administrativa'] = previsores['regiao_administrativa'].astype('int64')

# previsores.loc[:, 'faixa_etaria_legal'] = _le_faixa_legal.fit_transform(previsores.loc[:, 'faixa_etaria_legal'])
# previsores['faixa_etaria_legal'] = previsores['faixa_etaria_legal'].astype('int64')

classe.loc[:, 'gravidade_lesao'] = _le_gravidade.fit_transform(classe.loc[:, 'gravidade_lesao'])
classe['gravidade_lesao'] = classe['gravidade_lesao'].astype('int64')





# =============================================================================
#      Transformar as variáveis categóricas (nominais) em variáveis dummy
# =============================================================================


from sklearn.preprocessing import LabelBinarizer
_lb = LabelBinarizer()

# Variavel regiao_administrativa
_variaveis_dummy = _lb.fit_transform(previsores['regiao_administrativa'])
_novas_variaveis_dummy = _lb.classes_
_df_variaveis_dummy = pd.DataFrame(_variaveis_dummy, columns=_novas_variaveis_dummy)
previsores = previsores.join(_df_variaveis_dummy)
previsores = previsores.drop('regiao_administrativa',axis=1)

# Variavel tipo_veiculo_vitima
_variaveis_dummy = _lb.fit_transform(previsores['tipo_veiculo_vitima'])
_novas_variaveis_dummy = _lb.classes_
_df_variaveis_dummy = pd.DataFrame(_variaveis_dummy, columns=_novas_variaveis_dummy)
previsores = previsores.join(_df_variaveis_dummy)
previsores = previsores.drop('tipo_veiculo_vitima',axis=1)

# Variavel faixa_etaria_legal
# _variaveis_dummy = _lb.fit_transform(previsores['faixa_etaria_legal'])
# _novas_variaveis_dummy = _lb.classes_
# _df_variaveis_dummy = pd.DataFrame(_variaveis_dummy, columns=_novas_variaveis_dummy)
# previsores = previsores.join(_df_variaveis_dummy)
# previsores = previsores.drop('faixa_etaria_legal',axis=1)

# Variavel tipo_de_vitima
# _variaveis_dummy = _lb.fit_transform(previsores['tipo_de_vitima'])
# _novas_variaveis_dummy = _lb.classes_
# _df_variaveis_dummy = pd.DataFrame(_variaveis_dummy, columns=_novas_variaveis_dummy)
# previsores = previsores.join(_df_variaveis_dummy)
# previsores = previsores.drop('tipo_de_vitima',axis=1)

# Variavel profissao
# _variaveis_dummy = _lb.fit_transform(previsores['profissao'])
# _novas_variaveis_dummy = _lb.classes_
# _df_variaveis_dummy = pd.DataFrame(_variaveis_dummy, columns=_novas_variaveis_dummy)
# previsores = previsores.join(_df_variaveis_dummy)
# previsores = previsores.drop('profissao',axis=1)

_cols_previsores = previsores.columns





# =============================================================================
#                     Balanceamento com Undersampling
# =============================================================================


_gravidade_count = base_tratamento['gravidade_lesao'].value_counts()
print('Classe grave:', _gravidade_count['LEVE'])
print('Classe leve:', _gravidade_count['GRAVE'])
print('Proportion:', round(_gravidade_count['LEVE'] / _gravidade_count['GRAVE'], 2), ': 1')
_gravidade_count.plot(kind='bar', title='Count (target)',color = ['#1F77B4', '#FF7F0E']);

from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# Aplicando Undersampling
_us = RandomUnderSampler(random_state=69)
previsores, classe = _us.fit_resample(previsores, classe)

# Exibindo as novas distribuições das classes
print(classe.value_counts())

# Exportar o DataFrame para um arquivo CSV
_df_balanceada = pd.concat([previsores, classe], axis=1)
_df_balanceada.to_csv('sinistros_sp_pessoas_balanceado.csv', index=False)





# =============================================================================
#                 Separando em base de testes e treinamento
# =============================================================================


#  usando 25% para teste
from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=69)






# =============================================================================
#                     Padronização dos dados
# =============================================================================


from sklearn.preprocessing import StandardScaler

_scaler = StandardScaler()
previsores_treinamento = _scaler.fit_transform(previsores_treinamento)
previsores_teste = _scaler.transform(previsores_teste)



















