# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:32:07 2026

@author: BIEL
"""

import pandas as pd
# import numpy  as mp

# ===========================================================
#                 Descobrindo Correlações
# ===========================================================

#    ( ! ) Executar antes da padronização
base_completa = pd.concat([previsores, classe], axis=1)

corr = base_completa.corr()



#   Plotar matriz de correlação de pearson
import seaborn as sns
_ax = sns.heatmap(
       corr,
       
       vmin = -1,
       vmax = 1,
       center = 0,
       
       cmap = sns.diverging_palette(20, 220, n = 200),
       
       square = True,
       xticklabels = True,
       yticklabels = True
    )

_ax.set_xticklabels(
        ax.get_xticklabels(), 
        rotation = 45, 
        horizontalalignment = 'right'
    )





####################### Removendo colunas correlacionadas #####################

# base = pd.read_csv('credit_data.csv')
previsores = previsores.drop('NENHUM', axis = 1)




