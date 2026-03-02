import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def run_etl():
    print("Iniciando ETL (Extração, Transformação e Carga)...")
    
    # 1. Carregamento Otimizado
    dtypes = {
        'MONTH': 'int8', 'DAY_OF_WEEK': 'int8', 'AIRLINE': 'str',
        'ORIGIN_AIRPORT': 'str', 'DESTINATION_AIRPORT': 'str',
        'SCHEDULED_DEPARTURE': 'int32', 'ARRIVAL_DELAY': 'float32',
        'DISTANCE': 'float32', 'DIVERTED': 'int8', 'CANCELLED': 'int8'
    }
    
    flights_path = os.path.join(base_dir, 'dataset', 'flights.csv')
    df_flights = pd.read_csv(flights_path, dtype=dtypes, low_memory=False)
    
    airports_path = os.path.join(base_dir, 'dataset', 'airports.csv')
    df_airports = pd.read_csv(airports_path)
    
    airlines_path = os.path.join(base_dir, 'dataset', 'airlines.csv')
    df_airlines = pd.read_csv(airlines_path).rename(columns={'AIRLINE': 'AIRLINE_NAME'})

    # 2. Merges Essenciais
    df = df_flights.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left').drop(columns=['IATA_CODE'])
    df = df.merge(df_airports[['IATA_CODE', 'AIRPORT', 'LATITUDE', 'LONGITUDE']], 
                  left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left').drop(columns=['IATA_CODE'])
    
    # 3. Limpeza e Engenharia de Features
    # Focamos em voos realizados para ML
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    
    # Criar Target: 1 para atraso > 15min
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Extrair Hora
    df['SCHEDULED_HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    
    # Remover colunas desnecessárias para o ML para diminuir o arquivo
    cols_to_keep = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
                    'DESTINATION_AIRPORT', 'SCHEDULED_HOUR', 'DISTANCE', 
                    'LATITUDE', 'LONGITUDE', 'TARGET']
    
    df_final = df[cols_to_keep].dropna()

    # 4. Salvar em Parquet (O segredo da performance)
    flights_processed_path = os.path.join(base_dir, 'parquet', 'flights_processed.parquet')
    df_final.to_parquet(flights_processed_path, index=False)
    print("Sucesso! Arquivo 'flights_processed.parquet' gerado.")

run_etl()