import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configurações de Diretório
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'img_nao_supervisionado')
os.makedirs(output_dir, exist_ok=True)

# 1. Carga do Parquet
print("🌍 Carregando dados para Clusterização...")
flights_ready_path = os.path.join(base_dir, 'parquet', 'flights_ready.parquet')
df = pd.read_parquet(flights_ready_path)

# 2. Agrupamento Geográfico para o K-Means
# Agrupamos por coordenadas para identificar "zonas de risco"
df_geo = df.groupby(['LATITUDE', 'LONGITUDE']).agg({
    'TARGET': 'mean',        # Frequência de atrasos
    'DISTANCE': 'mean'       # Distância média das rotas
}).reset_index()

# 3. K-Means
print("📌 Executando K-Means (Não Supervisionado)...")
X_clus = df_geo[['LATITUDE', 'LONGITUDE', 'TARGET']]
scaler = StandardScaler()
X_clus_scaled = scaler.fit_transform(X_clus)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_geo['CLUSTER'] = kmeans.fit_predict(X_clus_scaled)

# 4. Gráfico: Mapa de Clusters

plt.figure(figsize=(12, 7))
scatter = plt.scatter(df_geo['LONGITUDE'], df_geo['LATITUDE'], 
            c=df_geo['CLUSTER'], cmap='viridis', s=50, alpha=0.6, edgecolors='white')
plt.colorbar(scatter, label='Cluster ID')
plt.title('Clusters de Aeroportos por Risco Geográfico de Atraso', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'{output_dir}/mapa_clusters_geograficos.png')

print(f"✨ Modelagem Não Supervisionada concluída! Imagens em: {output_dir}")