#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[2]:


#%matplotlib inline
#from IPython.core.pylabtools import figsize
#figsize(12, 8)
#sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head()


# In[15]:


x = get_sample(athletes, 'height', 3000)
x


# In[16]:


sct.shapiro(x)


# In[17]:


sct.jarque_bera(x)


# In[19]:


x.plot.hist()


# In[22]:


# Considerando agora uma amostra de tamanho 3000 da coluna weight obtida com a função get_sample(). 
# Faça o teste de normalidade de D'Agostino-Pearson utilizando a função scipy.stats.normaltest(). Podemos afirmar que os pesos vêm de uma 
# distribuição normal ao nível de significância de 5%? Responda com um boolean (True ou False).

x1 = get_sample(athletes, 'weight', 3000)
x1


# In[23]:


sct.normaltest(x1)


# In[25]:


x1.plot.hist()


# In[35]:


#plotting dist, qq and boxplot
import statsmodels.api as sm
sample = get_sample(athletes, 'weight', n = 3000)
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
sns.distplot(sample, bins = 25, ax = ax[0])
sm.qqplot(sample, fit = True, line='45', scale = .5, ax = ax[1])
sns.boxplot(sample, orient = 'v', width = 0.20, ax = ax[2])
plt.show()


# In[28]:


x1log = np.log(x1)
x1log


# In[29]:


x1log.plot.hist()


# In[ ]:


# Realize uma transformação logarítmica em na amostra de weight da questão 3 e repita o mesmo procedimento. 
# Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (True ou False).


# In[30]:


sct.normaltest(x1log)


# In[36]:


#plotting dist, qq and boxplot
sample = np.log(get_sample(athletes, 'weight', n = 3000))
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
sns.distplot(sample, bins = 25, ax = ax[0])
sm.qqplot(sample, fit = True, line='45', scale = .5, ax = ax[1])
sns.boxplot(sample, orient = 'v', width = 0.20,  ax = ax[2])
plt.show()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    sample = get_sample(athletes, 'height', n = 3000)
    
    W, p_value = sct.shapiro(sample)
    print('W - ' + str(W))
    print('p_value - ' + str(p_value))
    
    return not(p_value < 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[ ]:


def q2():
    sample = get_sample(athletes, 'height', n = 3000)
    jb, p_value = sct.jarque_bera(sample)
    
    print('skew - ' + str(sample.skew()))
    print('kurt - ' + str(sample.kurtosis()))   
    print('jb - ' + str(jb))
    print('p_value - ' + str(p_value))
    
    return not(p_value < 0.05)


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[ ]:


def q3():
    sample = get_sample(athletes, 'weight', n = 3000)
    dap, p_value = sct.normaltest(sample)
    
    print('skew - ' + str(sample.skew()))
    print('dap - ' + str(dap))
    print('p_value - ' + str(p_value))
    
    return not(p_value < 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[ ]:


def q4():
    sample = np.log(get_sample(athletes, 'weight', n = 3000))
    dap, p_value = sct.normaltest(sample)
    
    print('skew - ' + str(sample.skew()))
    print('dap - ' + str(dap))
    print('p_value - ' + str(p_value))
    
    return not(p_value < 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# In[32]:


athletes.head()


# In[39]:


bra, can, usa = [athletes[athletes['nationality'] == nation] for nation in ['BRA', 'CAN', 'USA']]
bra.head()


# In[46]:


stat,p_value = sct.ttest_ind(usa['height'], can['height'], equal_var = False, nan_policy = 'omit') 
print(p_value)
not(p_value < 0.05) #médias iguais -> não rejeitar H0


# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[ ]:


def q5():
    stat, p_value = sct.ttest_ind(bra['height'], usa['height'], equal_var = False, nan_policy = 'omit') 
    print(p_value)
    return not(p_value < 0.05) #médias iguais -> não rejeitar H0


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[ ]:


def q6():
    stat, p_value = sct.ttest_ind(bra['height'], can['height'], equal_var = False, nan_policy = 'omit') 
    print(p_value)
    return not(p_value < 0.05)


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[ ]:


def q7():
    stat, p_value = sct.ttest_ind(usa['height'], can['height'], equal_var = False, nan_policy = 'omit') 
    return float(p_value.round(8))


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
