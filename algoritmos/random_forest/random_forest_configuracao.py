# -*- coding: utf-8 -*-
"""
@author: joão 
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import itertools

# Parâmetros para testar
n_estimators_config = [50, 100, 200, 300]
max_depth_config    = range(5, 20)
criterion_config    = ['entropy', 'gini']

melhor_acuracia  = 0
melhor_parametros = {}
resultados = []

total = len(n_estimators_config) * len(max_depth_config) * len(criterion_config)
atual = 0

for n_est, depth, criterion in itertools.product(n_estimators_config, max_depth_config, criterion_config):
    atual += 1
    print(f"Testando {atual}/{total}: n_estimators={n_est}, max_depth={depth}, criterion={criterion}")
    
    classificador = RandomForestClassifier(n_estimators=n_est,
                                           max_depth=depth,
                                           criterion=criterion,
                                           random_state=0,
                                           n_jobs=-1)
    classificador.fit(previsores_treinamento, classe_treinamento)
    acuracia = round(accuracy_score(classe_teste, classificador.predict(previsores_teste)), 5)
    
    resultados.append({
        'n_estimators': n_est,
        'max_depth': depth,
        'criterion': criterion,
        'acuracia': acuracia
    })
    
    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_parametros = {
            'n_estimators': n_est,
            'max_depth': depth,
            'criterion': criterion
        }

print("\n========== RESULTADO FINAL ==========")
print(f"Melhor acurácia:   {melhor_acuracia}")
print(f"Melhores parâmetros: {melhor_parametros}")