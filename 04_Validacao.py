import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score
import os
import time  # Importado para medir a eficiência

# Configuração de base
base_dir = os.path.dirname(os.path.abspath(__file__))

def run_pre_modelagem():
    # Início da medição total
    start_total = time.time()
    
    # 1. Carga Ultra-Rápida
    print("📂 Carregando dados preparados do Parquet...")
    start_load = time.time()
    
    flights_processed_path = os.path.join(base_dir, 'parquet', 'flights_processed.parquet')
    
    if not os.path.exists(flights_processed_path):
        print(f"❌ Erro: Arquivo {flights_processed_path} não encontrado. Rode o script 03_ETL.py primeiro.")
        return

    df = pd.read_parquet(flights_processed_path)
    
    end_load = time.time()
    print(f"⏱️ Tempo de carregamento (Parquet): {end_load - start_load:.2f} segundos")

    # 2. Prepara Variáveis para Supervisionado
    print("⚙️ Preparando variáveis (Label Encoding)...")
    le = LabelEncoder()
    for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['TARGET', 'LATITUDE', 'LONGITUDE'])
    y = df['TARGET']

    # Divisão com amostragem estratificada para manter a proporção de atrasos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Modelo Supervisionado (Random Forest)
    print("🤖 Treinando Random Forest (Validação Preliminar)...")
    start_rf = time.time()
    
    # n_estimators=50 para um teste mais ágil nesta fase de validação
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    auc_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    
    end_rf = time.time()
    print(f"✅ Treinamento RF concluído em: {end_rf - start_rf:.2f} segundos")
    print(f"📈 ROC-AUC Preliminar: {auc_score:.4f}")

    # 4. Modelo Não Supervisionado (Clusterização de Aeroportos)
    print("🌍 Executando K-Means para clusterização geográfica...")
    start_km = time.time()
    
    # Agrupar pontos geográficos únicos
    df_geo = df[['LATITUDE', 'LONGITUDE']].drop_duplicates()
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(df_geo)
    df_geo['CLUSTER'] = kmeans.labels_
    
    end_km = time.time()
    print(f"✅ Clusterização concluída em: {end_km - start_km:.2f} segundos")

    # Fim da medição total
    end_total = time.time()
    
    print("\n" + "="*50)
    print("🏁 Modelagem Preliminar Concluída!")
    print(f"⏱️ Tempo total de execução: {end_total - start_total:.2f} segundos")
    print("="*50)

if __name__ == "__main__":
    run_pre_modelagem()