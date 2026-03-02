import pandas as pd
import numpy as np
import os
import time  # Importado para medir o tempo

def analisar_pontos_criticos(file_path):
    # Início da medição
    start_time = time.time()
    
    # Tipos de dados otimizados para economizar memória (Reduz o consumo em ~60%)
    dtypes = {
        'YEAR': 'int16', 'MONTH': 'int8', 'DAY': 'int8', 'DAY_OF_WEEK': 'int8',
        'AIRLINE': 'str', 'ORIGIN_AIRPORT': 'str', 'DESTINATION_AIRPORT': 'str',
        'DEPARTURE_DELAY': 'float32', 'ARRIVAL_DELAY': 'float32',
        'DIVERTED': 'int8', 'CANCELLED': 'int8'
    }

    print(f"🚀 Lendo {file_path}...")
    df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
    
    # --- ANÁLISE DE PONTOS CRÍTICOS ---
    
    # 1. Integridade dos Dados (Valores Nulos)
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df)) * 100
    
    # 2. Atrasos Críticos (Definidos como > 3 horas)
    atrasos_extremos = df[df['ARRIVAL_DELAY'] > 180].shape[0]
    taxa_atraso_extremo = (atrasos_extremos / len(df)) * 100
    
    # 3. Aeroportos Gargalos (Origem com maior média de atraso)
    min_volume = len(df) * 0.01
    stats_aeroporto = df.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].agg(['mean', 'count'])
    gargalos = stats_aeroporto[stats_aeroporto['count'] >= min_volume].sort_values(by='mean', ascending=False).head(10)
    
    # 4. Concentração de Cancelamentos
    taxa_cancelamento = df['CANCELLED'].mean() * 100
    
    # Fim da medição
    end_time = time.time()
    tempo_total = end_time - start_time

    print("\n" + "="*40)
    print("=== RELATÓRIO DE PONTOS CRÍTICOS ===")
    print("="*40)
    print(f"Total de registros: {len(df):,}")
    print(f"Taxa Global de Cancelamento: {taxa_cancelamento:.2f}%")
    print(f"Voos com Atraso Crítico (>3h): {atrasos_extremos:,} ({taxa_atraso_extremo:.2f}%)")
    print(f"\n⏱️ Tempo de processamento: {tempo_total:.2f} segundos")
    print("-" * 40)
    print("\n--- Top 10 Aeroportos Gargalo (Volume Significativo) ---")
    print(gargalos)
    
    return df, gargalos

# Execução
base_dir = os.path.dirname(os.path.abspath(__file__))
flights_path = os.path.join(base_dir, 'dataset', 'flights.csv')

# Verifica se o arquivo existe antes de tentar ler
if os.path.exists(flights_path):
    df_final, gargalos = analisar_pontos_criticos(flights_path)
else:
    print(f"❌ Erro: O arquivo não foi encontrado em {flights_path}")