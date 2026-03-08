import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
import time  # Importado para medir a performance da limpeza

# 1. Configuração de Diretório
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'img_randon_forest')
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

# 1. Carregamento Relâmpago
print("📂 Carregando dados do Parquet...")
flights_ready_path = os.path.join(base_dir, 'parquet', 'flights_ready.parquet')
df = pd.read_parquet(flights_ready_path)

# 2. Preparação de Categóricas (Label Encoding)
le = LabelEncoder()
for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
    df[col] = le.fit_transform(df[col])

# --- PARTE A: MODELAGEM SUPERVISIONADA (CLASSIFICAÇÃO) ---
print("🤖 Treinando Modelo Supervisionado (Random Forest)...")
X = df[['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'HOUR', 'DISTANCE']]
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Usamos class_weight='balanced' para lidar com o fato de que atrasos são minoria
rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f"\n✅ RESULTADO SUPERVISIONADO:")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred))

# --- PARTE B: MODELAGEM NÃO SUPERVISIONADA (CLUSTERIZAÇÃO) ---
print("\n🌍 Iniciando Clusterização Geográfica...")
# Agrupamos por aeroporto para não sobrecarregar o gráfico
df_geo = df.groupby(['LATITUDE', 'LONGITUDE']).agg({'TARGET': 'mean'}).reset_index()

X_clus = df_geo[['LATITUDE', 'LONGITUDE', 'TARGET']]
scaler = StandardScaler()
X_clus_scaled = scaler.fit_transform(X_clus)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_geo['CLUSTER'] = kmeans.fit_predict(X_clus_scaled)

# --- VISUALIZAÇÕES PARA O VÍDEO ---
print("📊 Gerando gráficos...")
os.makedirs(output_dir, exist_ok=True)

# 1. Matriz de Confusão
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Predição de Atraso')
plt.savefig(f'{output_dir}/matriz_confusao.png')

# 2. Mapa de Clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_geo['LONGITUDE'], df_geo['LATITUDE'], c=df_geo['CLUSTER'], cmap='viridis', s=30, alpha=0.6)
plt.title('Clusters de Aeroportos por Risco de Atraso')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig(f'{output_dir}/mapa_clusters_ml.png')



# Ver a importância das variáveis
importances = rf.feature_importances_
features = X.columns
df_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n--- Variáveis mais importantes para o Modelo ---")
print(df_imp)

# Gráfico final de Importância das Variáveis sem avisos
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=df_imp, 
    hue='Feature',      # Define a cor baseada na própria feature
    palette='magma', 
    legend=False        # Remove a legenda desnecessária
)
plt.title('O que mais causa atraso? (Feature Importance)', fontsize=14, fontweight='bold')
plt.xlabel('Importância Relativa (0 a 1)')
plt.ylabel('Variáveis do Modelo')
plt.savefig(f'{output_dir}/feature_importance_final.png', bbox_inches='tight')
plt.show()

# Fim da medição
end_time = time.time()
tempo_total = end_time - start_time

print(f"⏱️ Tempo total de Random Forest: {tempo_total:.2f} segundos")
print("="*50)


print("✨ Processo concluído! Gráficos salvos em 'goo_img/'.")