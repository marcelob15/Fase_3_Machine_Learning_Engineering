import pandas as pd
import numpy as np
import os
import time  # Importado para medir a eficiência do processo

# Configuração de base
base_dir = os.path.dirname(os.path.abspath(__file__))

def run_etl():
    # Início da medição de tempo
    start_time = time.time()
    
    print("🚀 Iniciando ETL (Extração, Transformação e Carga)...")
    
    # 1. Carregamento Otimizado (CSV original)
    dtypes = {
        'MONTH': 'int8', 'DAY_OF_WEEK': 'int8', 'AIRLINE': 'str',
        'ORIGIN_AIRPORT': 'str', 'DESTINATION_AIRPORT': 'str',
        'SCHEDULED_DEPARTURE': 'int32', 'ARRIVAL_DELAY': 'float32',
        'DISTANCE': 'float32', 'DIVERTED': 'int8', 'CANCELLED': 'int8'
    }
    
    flights_path = os.path.join(base_dir, 'dataset', 'flights.csv')
    
    if not os.path.exists(flights_path):
        print(f"❌ Erro: Arquivo {flights_path} não encontrado.")
        return

    df_flights = pd.read_csv(flights_path, dtype=dtypes, low_memory=False)
    
    airports_path = os.path.join(base_dir, 'dataset', 'airports.csv')
    df_airports = pd.read_csv(airports_path)
    
    airlines_path = os.path.join(base_dir, 'dataset', 'airlines.csv')
    df_airlines = pd.read_csv(airlines_path).rename(columns={'AIRLINE': 'AIRLINE_NAME'})

    # 2. Merges Essenciais
    print("🔗 Realizando cruzamento de tabelas (Merges)...")
    df = df_flights.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left').drop(columns=['IATA_CODE'])
    df = df.merge(df_airports[['IATA_CODE', 'AIRPORT', 'LATITUDE', 'LONGITUDE']], 
                  left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left').drop(columns=['IATA_CODE'])
    
    # 3. Limpeza e Engenharia de Features
    # Focamos em voos realizados para garantir a qualidade do treinamento de ML
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    
    # Criar Target binária: 1 para atraso > 15min (Padrão internacional FAA)
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Extrair Hora (Feature Engineering fundamental)
    df['SCHEDULED_HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    
    # Seleção de colunas críticas para reduzir o tamanho do arquivo final
    cols_to_keep = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
                    'DESTINATION_AIRPORT', 'SCHEDULED_HOUR', 'DISTANCE', 
                    'LATITUDE', 'LONGITUDE', 'TARGET']
    
    df_final = df[cols_to_keep].dropna()

    # 4. Salvar em Parquet (O segredo da performance)
    parquet_dir = os.path.join(base_dir, 'parquet')
    os.makedirs(parquet_dir, exist_ok=True) # Garante que a pasta existe
    
    flights_processed_path = os.path.join(parquet_dir, 'flights_processed.parquet')
    
    print(f"💾 Salvando {len(df_final):,} registros em formato Parquet...")
    df_final.to_parquet(flights_processed_path, index=False)
    
    # Fim da medição
    end_time = time.time()
    tempo_total = end_time - start_time

    print("\n" + "="*50)
    print(f"✅ Sucesso! Arquivo 'flights_processed.parquet' gerado.")
    print(f"⏱️ Tempo de execução do ETL: {tempo_total:.2f} segundos")
    print("="*50)

if __name__ == "__main__":
    run_etl()