📊 Previsão de Falência com Indices Financeiros
---

## 📌 Descrição
Este projeto aborda a **previsão de falência empresarial** usando o dataset **Company Bankruptcy Prediction** (UCI Repository). O objetivo foi treinar uma **Rede Neural Multicamadas (MLP, Keras/TensorFlow)** para classificar empresas entre **Não falida (0)** e **Falida (1)** a partir de **indicadores financeiros**.

O estudo envolveu: 

- **Análise Exploratória (EDA):** inspeção de tipos, contagens nulas/duplicadas e distribuição do alvo;  
- **Pré-processamento (pipeline):**  transformações **log1p**; remoção de variância zero (***VarianceThreshold***); escalonamento dos atributos (***Min-Max Sacale***);  
- **Modelagem:**  baseline com com *Logistic Regression* (`class_weight='balanced'`) para referência e **MLP** com arquitetura densa e saída sigmoid.
- **Treinamento com pesos de classe** (calculados a partir da frequência do `y_train`), **EarlyStopping** (*patience*=10, `monitor='val_auc'`) e **ReduceLROnPlateau** (*factor*=0.5, *patience*=3);

---

## 📊 Resultados

| Modelo                      | Accuracy | AUC-ROC | F1 (Classe 1) |
|-----------------------------|:--------:|:-------:|:-------------:|
| Regressão Logística (base)  |  0,8304  | 0,9496  |    0,2629     |
| **MLP**^1  | **0,9441** | 0,9419  |  **0,4602**   |

> Observação: Limiar escolhido por teste do F1.

---

## 📌 Conclusão
- A **MLP**, com **pesos de classe** e **ajuste de limiar** via validação, **superou o baseline** em **F1 da classe positiva**, mantendo **AUC elevada**.  
- O cenário de **desbalanceamento extremo** torna a **precisão** da classe positiva mais modesta, mas o **recall** é significativamente melhorado, tornando-se útil para **detecção precoce de risco**.  

- Em contextos reais, recomenda-se:  
  - analisar o **trade-off precisão/recall** conforme o custo de falsos positivos/negativos;  
  - ampliar a **curadoria de dados positivos** (classe 1);  
  - considerar **técnicas adicionais** como detecção de anomalias ou *threshold moving* por segmento.

---
