# Processo
- Normalizar dataset
    - Ex1.: campos faltantes podem ser substituidos pela média da coluna
    - Ex2.: transformar colunas não numericas em ENUMS (como a coluna sexo) para incorporar à análise
    - Ex3.: remover NaN, valores vazios por valores compreensiveis
- Aplicar diversos modelos de aprendizado a fim de obter resultados e insights de forma rápida
- Antes de mudar de algoritmo, tentar criar features derivadas das originais (feature engineering)
    - Para validar quais features são mais úteis, utiliza-se a técnica de 'univariate feature selection'
    - Sklearn tem a função 'SelectKBest' que pode auxilar neste processo
- Começar dos modelos mais simples (LinearRegression, RandomForests) aos mais complexos (NN)

# Feature engineering tips

There's still more work you can do in feature engineering:

- Try using features related to the cabins.
- See if any family size features might help -- do the number of women in a family make the whole family more likely to survive?
- Does the national origin of the passenger's name have anything to do with survival?

There's also a lot more we can do on the algorithm side:

- Try the random forest classifier in the ensemble.
- A support vector machine might work well with this data.
- We could try neural networks.
- Boosting with a different base classifier might work better.

And with ensembling methods:

- Could majority voting be a better ensembling method than averaging probabilities?

# Modelos

## Random Forests
Increase the number of trees in order to improve the accuracy. We can also tweak the min_samples_split and min_samples_leaf variables to reduce overfitting.

Another method that builds on decision trees is a gradient boosting classifier.

# References
- Random Forests; https://citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics/