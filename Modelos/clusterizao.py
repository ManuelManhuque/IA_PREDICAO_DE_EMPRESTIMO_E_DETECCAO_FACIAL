import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Carregar o arquivo CSV
df = pd.read_csv('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.csv')

# Preparar os dados
X = df[['idade', 'tipo_conta', 'saldo_conta']]

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar o modelo K-means
kmeans = KMeans(n_clusters=3, random_state=42)

# Treinar o modelo
kmeans.fit(X_scaled)

# Adicionar as labels ao DataFrame original
df['cluster'] = kmeans.labels_

# Salvar os resultados dos clusters em um arquivo CSV
df.to_csv('resultados_clusters.csv', index=False)

# Salvar o modelo K-means em um arquivo pickle
with open('modelo_clusters.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Plotar os clusters em diferentes combinações de características
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(df['idade'], df['saldo_conta'], c=df['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Idade')
plt.ylabel('Saldo Conta')
plt.title('Clusters de Clientes (Idade vs Saldo Conta)')
plt.savefig('C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_idade_saldo.png')

plt.subplot(1, 3, 2)
plt.scatter(df['idade'], df['tipo_conta'], c=df['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Idade')
plt.ylabel('Tipo Conta')
plt.title('Clusters de Clientes (Idade vs Tipo Conta)')
plt.savefig('C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_idade_tipo_conta.png')

plt.subplot(1, 3, 3)
plt.scatter(df['tipo_conta'], df['saldo_conta'], c=df['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Tipo Conta')
plt.ylabel('Saldo Conta')
plt.title('Clusters de Clientes (Tipo Conta vs Saldo Conta)')
plt.savefig('C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_tipo_conta_saldo.png')

plt.tight_layout()
plt.show()

# Plotar os centros dos clusters
centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(df['idade'], df['saldo_conta'], c=df['cluster'], cmap='viridis', s=50, alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 2], marker='o', c='red', s=200, label='Centro do Cluster')
plt.xlabel('Idade')
plt.ylabel('Saldo Conta')
plt.title('Clusters de Clientes com Centros')
plt.legend()
plt.show()
