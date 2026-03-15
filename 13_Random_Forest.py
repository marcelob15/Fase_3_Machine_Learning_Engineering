import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Configuração de Diretório
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'img_randon_forest')
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

# 1. Carregamento do CSV Processado
print("📂 Carregando dados do CSV Processado...")
flights_ready_path = os.path.join(base_dir, 'csv_processado', 'flights_ready.csv')

if not os.path.exists(flights_ready_path):
    print(f"❌ Erro: Arquivo {flights_ready_path} não encontrado.")
else:
    # Definindo tipos para economizar memória durante a leitura do CSV
    optimized_dtypes = {
        'MONTH': 'int8',
        'DAY_OF_WEEK': 'int8',
        'AIRLINE': 'category',
        'ORIGIN_AIRPORT': 'category',
        'DESTINATION_AIRPORT': 'category',
        'HOUR': 'int8',
        'DISTANCE': 'float32',
        'LATITUDE': 'float32',
        'LONGITUDE': 'float32',
        'TARGET': 'int8'
    }
    
    df = pd.read_csv(flights_ready_path, dtype=optimized_dtypes)

    # 2. Preparação de Categóricas (Label Encoding)
    print("⚙️ Codificando variáveis categóricas...")
    le = LabelEncoder()
    for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
        df[col] = le.fit_transform(df[col].astype(str))

    # --- PARTE A: MODELAGEM SUPERVISIONADA (CLASSIFICAÇÃO) ---
    print("🤖 Treinando Modelo Supervisionado (Random Forest)...")
    # Colunas preditoras
    features_list = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'HOUR', 'DISTANCE']
    X = df[features_list]
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Modelo configurado para performance e equilíbrio de classes
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print(f"\n✅ RESULTADO SUPERVISIONADO:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred))

    # --- PARTE B: MODELAGEM NÃO SUPERVISIONADA (CLUSTERIZAÇÃO) ---
    print("\n🌍 Iniciando Clusterização Geográfica de Aeroportos...")
    # Agrupamos por coordenadas para identificar o risco médio por localização
    df_geo = df.groupby(['LATITUDE', 'LONGITUDE']).agg({'TARGET': 'mean'}).reset_index()

    scaler = StandardScaler()
    X_clus_scaled = scaler.fit_transform(df_geo[['LATITUDE', 'LONGITUDE', 'TARGET']])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_geo['CLUSTER'] = kmeans.fit_predict(X_clus_scaled)

    # --- VISUALIZAÇÕES ---
    print("📊 Gerando gráficos e artefatos...")

    # 1. Matriz de Confusão
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Predição de Atraso')
    plt.savefig(os.path.join(output_dir, 'matriz_confusao.png'))

    # 2. Mapa de Clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df_geo['LONGITUDE'], df_geo['LATITUDE'], c=df_geo['CLUSTER'], cmap='viridis', s=30, alpha=0.6)
    plt.colorbar(label='Cluster ID')
    plt.title('Clusters de Aeroportos por Risco Geográfico de Atraso')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'mapa_clusters_ml.png'))

    # 3. Importância das Variáveis
    importances = rf.feature_importances_
    df_imp = pd.DataFrame({'Feature': features_list, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    print("\n--- Variáveis mais importantes para o Modelo ---")
    print(df_imp)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', palette='magma', legend=False)
    plt.title('O que mais causa atraso? (Feature Importance)', fontsize=14, fontweight='bold')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Variáveis')
    plt.savefig(os.path.join(output_dir, 'feature_importance_final.png'), bbox_inches='tight')
    
    # Fim da medição
    end_time = time.time()
    tempo_total = end_time - start_time

    print("\n" + "="*50)
    print(f"⏱️ Tempo total de Processamento e ML: {tempo_total:.2f} segundos")
    print(f"✨ Processo concluído! Gráficos salvos em: {output_dir}")
    print("="*50)