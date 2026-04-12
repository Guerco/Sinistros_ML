# -*- coding: utf-8 -*-
"""


@author: joão 
"""

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

training_acuracy = []
test_acuracy = []


altura_config = range(1, 12)

for altura in altura_config:
    # Construindo o modelo
    classificador = DecisionTreeClassifier(max_depth=altura,
                                           criterion='entropy',
                                           random_state=0
                                           ) 
    
    # Treinando
    classificador.fit(previsores_treinamento, classe_treinamento)
    
    # Gravando o resultado para os dados de treinamento
    training_acuracy.append(classificador.score(
            previsores_treinamento, classe_treinamento
        ))
    
    # Gravando o resultado para os dados de treinamento
    test_acuracy.append(classificador.score(
            previsores_teste, classe_teste
        ))
    
plt.figure()    
plt.plot(altura_config, training_acuracy, label="Acurácia no Treinamento")
plt.plot(altura_config, test_acuracy, label="Acurácia no Teste")   
plt.ylabel("Acurácia")
plt.xlabel("Altura")    
plt.legend()
plt.show()
    