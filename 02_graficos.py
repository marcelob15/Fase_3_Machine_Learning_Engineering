import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Configuração de Diretório
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'img_eda')
os.makedirs(output_dir, exist_ok=True)

# 2. Configurações de Estilo para o Vídeo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150  # Melhora a resolução para o vídeo

def gerar_analise_visual(file_path):
    print(f"Iniciando processamento de {file_path}...")
    
    # Dtypes otimizados para o arquivo de 500MB
    dtypes = {
        'AIRLINE': 'str', 
        'ORIGIN_AIRPORT': 'str',
        'SCHEDULED_DEPARTURE': 'int32', 
        'ARRIVAL_DELAY': 'float32'
    }
    
    # Carregamos apenas o essencial para os gráficos
    cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE', 'ARRIVAL_DELAY']
    df = pd.read_csv(file_path, usecols=cols, dtype=dtypes, low_memory=False)
    
    # Carregar nomes das companhias
    airlines_path = os.path.join(base_dir, 'dataset', 'airlines.csv')
    df_airlines = pd.read_csv(airlines_path).rename(columns={'AIRLINE': 'AIRLINE_NAME'})
    
    # Preparação da Target (Atraso > 15 min) e Hora
    df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    df['HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    df = df.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')

    # --- GRÁFICO 01: DISTRIBUIÇÃO ---
    print("Salvando Gráfico 01...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ARRIVAL_DELAY'].dropna(), bins=100, color='#3498db', kde=False)
    plt.axvline(15, color='#e74c3c', linestyle='--', label='Corte de 15 min (Target)')
    plt.title('Gráfico 01: Distribuição de Atrasos na Chegada', fontsize=14, fontweight='bold')
    plt.xlabel('Minutos de Atraso')
    plt.ylabel('Volume de Voos')
    plt.xlim(-60, 200) 
    plt.legend()
    plt.savefig(f'{output_dir}/grafico_01_distribuicao.png', bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 02: COMPANHIAS ---
    print("Salvando Gráfico 02...")
    cia_delay = df.groupby('AIRLINE_NAME')['IS_DELAYED'].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=cia_delay.values,
        y=cia_delay.index,
        hue=cia_delay.index,
        palette='magma',
        legend=False
    )
    plt.title('Gráfico 02: Percentual de Atrasos por Companhia', fontsize=14, fontweight='bold')
    plt.xlabel('Percentual de Voos com Atraso > 15min (%)')
    plt.ylabel('')
    plt.savefig(f'{output_dir}/grafico_02_companhias.png', bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 03: AEROPORTOS ---
    print("Salvando Gráfico 03...")
    # Filtrar aeroportos com volume expressivo (evita outliers de aeroportos minúsculos)
    min_vol = len(df) * 0.005 
    apt_stats = df.groupby('ORIGIN_AIRPORT')['IS_DELAYED'].agg(['mean', 'count'])
    top_apts = apt_stats[apt_stats['count'] >= max(1, min_vol)].sort_values(by='mean', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_apts['mean'] * 100,
        y=top_apts.index,
        hue=top_apts.index,
        palette='viridis',
        legend=False
    )


    plt.title('Gráfico 03: Top 10 Aeroportos "Gargalo"', fontsize=14, fontweight='bold')
    plt.xlabel('Chance de Atraso na Partida (%)')
    plt.ylabel('Código IATA')
    plt.savefig(f'{output_dir}/grafico_03_aeroportos.png', bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 04: PADRÃO HORÁRIO ---
    print("Salvando Gráfico 04...")
    hora_delay = df.groupby('HOUR')['IS_DELAYED'].mean() * 100
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=hora_delay.index, y=hora_delay.values, marker='o', color='#e74c3c', linewidth=2)
    plt.fill_between(hora_delay.index, hora_delay.values, color='#e74c3c', alpha=0.1)
    plt.title('Gráfico 04: Atrasos Acumulados ao Longo do Dia', fontsize=14, fontweight='bold')
    plt.xlabel('Hora do Dia (Partida Programada)')
    plt.ylabel('% de Chance de Atraso')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/grafico_04_horario.png', bbox_inches='tight')
    plt.close()

    print(f"\nConcluído! Todos os gráficos estão em: {os.path.abspath(output_dir)}")

# Para rodar no seu arquivo completo, basta descomentar a linha abaixo:
flights_path = os.path.join(base_dir, 'dataset', 'flights.csv')
gerar_analise_visual(flights_path)
