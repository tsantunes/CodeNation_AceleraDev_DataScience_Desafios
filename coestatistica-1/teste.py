import pandas as pd 
import numpy as np
import json


df = pd.read_csv("desafio1.csv")

#filtro dos dados necessarios
filtro =['pontuacao_credito','estado_residencia']
a = df[filtro]

#definindo o df_mean
df_mean = a.groupby(['estado_residencia']).count()

#calculando as metricas e adicionando ao df_mean
df_mean['media'] = a.groupby(['estado_residencia']).mean()
df_mean['mediana'] = a.groupby(['estado_residencia']).median()
df_mean['desvio_padrao'] = a.groupby(['estado_residencia']).std()
df_mean['moda'] = a.groupby(['estado_residencia']).agg(pd.Series.mode)

#filtrando o df_mean com as metricas necessarias
filtro2 = ['media','mediana','desvio_padrao','moda']
df_resposta = df_mean[filtro2]

#exportando os dados para arquivo JSON
resposta = df_resposta.to_json(orient='index')
arquivo_resposta = open ("submission","w")
arquivo_resposta.write(resposta)
arquivo_resposta.close()

