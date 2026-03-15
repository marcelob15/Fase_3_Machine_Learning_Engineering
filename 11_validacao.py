import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import os
import time

def run_pre_modelagem():
    start_total = time.time()
    
    # 1. Configuração do Caminho
    flights_processed_path = r"c:\Users\P\Documents\Doc\Aula_Fiep\Fase_3\Trabalho\Fase_3_Machine_Learning_Engineering\csv_processado\flights_ready.csv"
    
    print(f"📂 Carregando dados preparados do CSV: {flights_processed_path}")
    start_load = time.time()
    
    if not os.path.exists(flights_processed_path):
        print(f"❌ Erro: Arquivo não encontrado.")
        return

    optimized_dtypes = {
        'MONTH': 'int8',
        'DAY': 'int8',
        'DAY_OF_WEEK': 'int8',
        'AIRLINE': 'category',
        'ORIGIN_AIRPORT': 'category',
        'DESTINATION_AIRPORT': 'category',
        'ORIGIN_STATE': 'category',
        'SCHEDULED_HOUR': 'int8',
        'DISTANCE': 'float32',
        'ARRIVAL_DELAY': 'float32',
        'ORIGIN_LATITUDE': 'float32',
        'ORIGIN_LONGITUDE': 'float32',
        'TARGET': 'int8'
    }

    df = pd.read_csv(flights_processed_path, dtype=optimized_dtypes)
    end_load = time.time()
    print(f"⏱️ Tempo de carregamento: {end_load - start_load:.2f} segundos")

    # 2. Prepara Variáveis (Correção do Erro de String)
    print("⚙️ Preparando variáveis (Label Encoding)...")
    le = LabelEncoder()
    
    # LISTA ATUALIZADA: Incluindo 'ORIGIN_STATE' que estava causando o erro
    cols_to_encode = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ORIGIN_STATE']
    
    for col in cols_to_encode:
        if col in df.columns:
            # Converte para string antes para evitar problemas com tipos categóricos
            df[col] = le.fit_transform(df[col].astype(str))

    # Definindo as features (X) e o alvo (y)
    # Removemos colunas que não são preditoras ou que causam vazamento de dados (como ARRIVAL_DELAY)
    cols_to_drop = ['TARGET', 'ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE', 'ARRIVAL_DELAY']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['TARGET']

    print(f"   -> Features utilizadas: {list(X.columns)}")

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Modelo Supervisionado
    print("🤖 Treinando Random Forest (Validação Preliminar)...")
    start_rf = time.time()
    
    # Reduzi para 30 estimadores para ser ainda mais rápido no seu teste
    rf = RandomForestClassifier(n_estimators=30, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    end_rf = time.time()
    print(f"✅ Treinamento RF concluído em: {end_rf - start_rf:.2f} segundos")
    print(f"📈 ROC-AUC Preliminar: {auc_score:.4f}")

    # 4. Modelo Não Supervisionado (Clusterização)
    print("🌍 Executando K-Means para clusterização geográfica...")
    start_km = time.time()
    
    df_geo = df[['ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE']].drop_duplicates()
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(df_geo)
    
    end_km = time.time()
    print(f"✅ Clusterização concluída em: {end_km - start_km:.2f} segundos")

    end_total = time.time()
    print("\n" + "="*50)
    print("🏁 Modelagem Preliminar Concluída!")
    print(f"⏱️ Tempo total de execução: {end_total - start_total:.2f} segundos")
    print("="*50)

if __name__ == "__main__":
    run_pre_modelagem()