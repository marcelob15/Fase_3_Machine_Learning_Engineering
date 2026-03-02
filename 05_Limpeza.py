import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def script_etl_limpeza():
    print("🚀 Iniciando Limpeza Pesada (CSV -> Parquet)...")
    
    # 1. Carregamento Seletivo (Otimiza RAM)
    cols_necessarias = [
        'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
        'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY',
        'ARRIVAL_DELAY', 'DISTANCE', 'DIVERTED', 'CANCELLED'
    ]
    
    # Lendo o arquivo bruto
    flights_path = os.path.join(base_dir, 'dataset', 'flights.csv')
    df = pd.read_csv(flights_path, usecols=cols_necessarias, low_memory=False)

    airports_path = os.path.join(base_dir, 'dataset', 'airports.csv')
    df_airports = pd.read_csv(airports_path)[['IATA_CODE', 'LATITUDE', 'LONGITUDE']]

    # 2. Filtragem: Apenas voos que decolaram e não foram desviados
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

    # 3. Criação da Target (Padrão FAA > 15 min)
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

    # 4. Engenharia de Variáveis
    df['HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    
    # 5. Merge com Coordenadas (Para a Clusterização Geográfica)
    df = df.merge(df_airports, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
    
    # 6. Seleção Final de Colunas para o ML
    # Removemos o que não é feature preditiva ou que causa erro
    cols_ml = [
        'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
        'DESTINATION_AIRPORT', 'HOUR', 'DISTANCE', 
        'LATITUDE', 'LONGITUDE', 'TARGET'
    ]
    df_final = df[cols_ml].dropna()

    # 7. Salvando em Parquet (MUITO mais rápido que CSV)
    flights_ready_path = os.path.join(base_dir, 'parquet', 'flights_ready.parquet')
    df_final.to_parquet(flights_ready_path, index=False)
    print(f"✅ Sucesso! {len(df_final):,} linhas salvas em 'flights_ready.parquet'.")

if __name__ == "__main__":
    script_etl_limpeza()