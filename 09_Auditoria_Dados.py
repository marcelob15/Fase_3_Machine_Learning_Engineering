
import pandas as pd
import numpy as np
import os
import time

# Configuração para exibir todas as colunas/linhas no terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

def auditoria_dados_eda(file_path):
    start_time = time.time()
    print(f"🚀 Iniciando Auditoria Numérica em: {file_path}...")
    
    # Carregamento Otimizado
    dtypes = {
        'AIRLINE': 'str', 'ORIGIN_AIRPORT': 'str',
        'SCHEDULED_DEPARTURE': 'int32', 'ARRIVAL_DELAY': 'float32'
    }
    cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE', 'ARRIVAL_DELAY']
    
    # Ler arquivo
    df = pd.read_csv(file_path, usecols=cols, dtype=dtypes, low_memory=False)
    
    # Criar Features
    df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    df['HOUR'] = (df['SCHEDULED_DEPARTURE'] // 100).astype(int)
    
    print(f"Total de Linhas Processadas: {len(df):,}")
    
    print("\n" + "="*60)
    print("🔎 ANÁLISE 1: O MISTÉRIO DE EWR (RANKING DE AEROPORTOS)")
    print("="*60)
    
    # Agrupar por Aeroporto
    stats_apt = df.groupby('ORIGIN_AIRPORT').agg(
        Total_Voos=('IS_DELAYED', 'count'),
        Qtd_Atrasos=('IS_DELAYED', 'sum'),
        Pct_Atraso=('IS_DELAYED', 'mean'),
        Media_Minutos=('ARRIVAL_DELAY', 'mean')
    ).reset_index()
    
    # Transformar média em porcentagem (0.20 -> 20.0%)
    stats_apt['Pct_Atraso'] = stats_apt['Pct_Atraso'] * 100
    
    # Filtro de Volume (0.5% do total para ser considerado "Grande/Médio")
    corte_volume = len(df) * 0.005
    print(f"Corte de Volume Mínimo (0.5%): {int(corte_volume)} voos")
    
    df_filtrado = stats_apt[stats_apt['Total_Voos'] >= corte_volume].copy()
    
    # Top 15 Piores por % de Atraso
    top_15_pct = df_filtrado.sort_values('Pct_Atraso', ascending=False).head(15)
    
    print("\n--- TOP 15 AEROPORTOS (Por % de Atraso > 15min) ---")
    print(top_15_pct[['ORIGIN_AIRPORT', 'Total_Voos', 'Pct_Atraso', 'Media_Minutos']].to_string(index=False))
    
    # Checagem Específica: Onde estão os famosos Hubs?
    hubs_check = ['EWR', 'ORD', 'JFK', 'ATL', 'LAX', 'DFW', 'DEN']
    print(f"\n--- AUDITORIA ESPECÍFICA DOS HUBS ({', '.join(hubs_check)}) ---")
    # Filtra os hubs independente do volume
    df_hubs = stats_apt[stats_apt['ORIGIN_AIRPORT'].isin(hubs_check)].sort_values('Pct_Atraso', ascending=False)
    print(df_hubs[['ORIGIN_AIRPORT', 'Total_Voos', 'Pct_Atraso', 'Media_Minutos']].to_string(index=False))

    print("\n" + "="*60)
    print("🔎 ANÁLISE 2: COMPORTAMENTO HORÁRIO (EFEITO BOLA DE NEVE)")
    print("="*60)
    
    stats_hour = df.groupby('HOUR').agg(
        Total_Voos=('IS_DELAYED', 'count'),
        Pct_Atraso=('IS_DELAYED', 'mean'),
        Media_Minutos=('ARRIVAL_DELAY', 'mean')
    ).reset_index()
    
    stats_hour['Pct_Atraso'] = stats_hour['Pct_Atraso'] * 100
    
    print("--- Tabela Hora a Hora ---")
    print(stats_hour.to_string(index=False))
    
    # Identificar o pico exato
    pico = stats_hour.loc[stats_hour['Pct_Atraso'].idxmax()]
    print(f"\n🔻 PICO DE ATRASO: Às {int(pico['HOUR'])}h com {pico['Pct_Atraso']:.2f}% de probabilidade.")
    
    end_time = time.time()
    print(f"\n⏱️ Processamento concluído em {end_time - start_time:.2f}s")

# Execução
base_dir = os.path.dirname(os.path.abspath(__file__))
flights_path = os.path.join(base_dir, 'dataset', 'flights.csv')

if os.path.exists(flights_path):
    auditoria_dados_eda(flights_path)
else:
    print("Arquivo não encontrado.")