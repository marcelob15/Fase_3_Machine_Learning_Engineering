import pandas as pd
import numpy as np
import os
import time

# Configuração de base
base_dir = os.path.dirname(os.path.abspath(__file__))

def script_etl_limpeza():
    # Início da medição de tempo
    start_time = time.time()
    
    print("🚀 Iniciando Limpeza Pesada e Otimização (CSV -> CSV Processado)...")
    
    # 1. Carregamento Seletivo (Otimiza RAM carregando apenas o necessário)
    cols_necessarias = [
        'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
        'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY',
        'ARRIVAL_DELAY', 'DISTANCE', 'DIVERTED', 'CANCELLED'
    ]
    
    flights_path = os.path.join(base_dir, 'dataset', 'flights.csv')
    
    if not os.path.exists(flights_path):
        print(f"❌ Erro: Arquivo {flights_path} não encontrado.")
        return

    # Lendo o arquivo bruto
    df = pd.read_csv(flights_path, usecols=cols_necessarias, low_memory=False)

    airports_path = os.path.join(base_dir, 'dataset', 'airports.csv')
    df_airports = pd.read_csv(airports_path)[['IATA_CODE', 'LATITUDE', 'LONGITUDE']]

    # 2. Filtragem: Apenas voos que decolaram e não foram desviados
    print("🧹 Filtrando cancelamentos e desvios...")
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

    # 3. Criação da Target
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

    # 4. Engenharia de Variáveis
    # Transformando HHMM em apenas a hora (0-23)
    df['HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    
    # 5. Merge com Coordenadas
    print("🔗 Acoplando coordenadas geográficas...")
    df = df.merge(df_airports, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
    
    # 6. Seleção Final de Colunas
    cols_ml = [
        'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
        'DESTINATION_AIRPORT', 'HOUR', 'DISTANCE', 
        'LATITUDE', 'LONGITUDE', 'TARGET'
    ]
    df_final = df[cols_ml].dropna()

    # 7. Salvando em CSV (Alterado de Parquet para CSV)
    csv_dir = os.path.join(base_dir, 'csv_processado')
    os.makedirs(csv_dir, exist_ok=True)
    
    flights_ready_path = os.path.join(csv_dir, 'flights_ready.csv')
    
    print(f"💾 Salvando base otimizada com {len(df_final):,} linhas em CSV...")
    # index=False evita que uma coluna extra de ID seja criada no arquivo
    df_final.to_csv(flights_ready_path, index=False)
    
    # Fim da medição
    end_time = time.time()
    tempo_total = end_time - start_time

    print("\n" + "="*50)
    print(f"✅ Sucesso! O arquivo 'flights_ready.csv' está pronto.")
    print(f"⏱️ Tempo total da Limpeza Pesada: {tempo_total:.2f} segundos")
    print("="*50)

if __name__ == "__main__":
    script_etl_limpeza()