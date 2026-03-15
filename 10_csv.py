import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# --- CONFIGURAÇÃO DE AMBIENTE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'csv_processado') # Alterado de Parquet para CSV
IMG_DIR = os.path.join(BASE_DIR, 'img_eda')

# Cria diretórios se não existirem
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Configuração visual
sns.set_theme(style="whitegrid")

def get_memory_usage_mb(df):
    """Calcula o uso real de memória RAM do DataFrame em MB."""
    return df.memory_usage(deep=True).sum() / 1024**2

def plotar_comparacao_memoria(mem_raw, mem_opt, output_path):
    """Gera o gráfico visual de otimização de memória."""
    plt.figure(figsize=(10, 6))
    
    valores = [mem_raw, mem_opt]
    labels = ['CSV Bruto (Raw)', 'CSV Otimizado']
    colors = ['#e74c3c', '#3498db'] # Vermelho e Azul
    
    bars = plt.bar(labels, valores, color=colors, width=0.6)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(valores)*0.01),
                 f'{height:.1f} MB',
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    reducao = 100 * (1 - mem_opt / mem_raw)
    
    plt.title(f'Otimização de Memória: Engenharia de Dados\nRedução de {reducao:.1f}% no consumo de RAM', fontsize=14)
    plt.ylabel('Uso de Memória RAM (MB)')
    plt.ylim(0, max(valores) * 1.15) # Dá um respiro no topo
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico de memória salvo em: {output_path}")

def run_etl_com_auditoria():
    start_global = time.time()
    print("🚀 [ETL] Iniciando Pipeline de Engenharia de Dados...")

    # ---------------------------------------------------------
    # 1. CARREGAMENTO BRUTO (RAW) - Para comparação
    # ---------------------------------------------------------
    flights_path = os.path.join(DATASET_DIR, 'flights.csv')
    
    if not os.path.exists(flights_path):
        print(f"❌ Erro: Arquivo {flights_path} não encontrado.")
        return

    print("📥 Carregando dados brutos (Raw)...")
    # Lemos sem otimização de dtype propositalmente para ver o peso original
    df_raw = pd.read_csv(flights_path, low_memory=False)
    
    memoria_antes = get_memory_usage_mb(df_raw)
    dtypes_antes = df_raw.dtypes.astype(str).to_dict()
    rows_antes = len(df_raw)
    
    print(f"   -> Memória Inicial: {memoria_antes:.2f} MB")
    print(f"   -> Linhas: {rows_antes:,}")

    # ---------------------------------------------------------
    # 2. TRANSFORMAÇÃO E LIMPEZA
    # ---------------------------------------------------------
    print("⚙️ Executando Transformações...")
    
    # 2.1 Carregar auxiliares
    df_airlines = pd.read_csv(os.path.join(DATASET_DIR, 'airlines.csv'))
    df_airports = pd.read_csv(os.path.join(DATASET_DIR, 'airports.csv'))
    
    # 2.2 Merges (Enriquecimento)
    df = df_raw.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
    df = df.rename(columns={'AIRLINE_y': 'AIRLINE_NAME', 'AIRLINE_x': 'AIRLINE'})
    df = df.drop(columns=['IATA_CODE'])
    
    # Merge apenas para pegar coordenadas da ORIGEM
    df = df.merge(df_airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE', 'STATE']], 
                  left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
    df = df.rename(columns={'LATITUDE': 'ORIGIN_LATITUDE', 'LONGITUDE': 'ORIGIN_LONGITUDE', 'STATE': 'ORIGIN_STATE'})
    df = df.drop(columns=['IATA_CODE'])

    # 2.3 Filtragem (Business Logic)
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

    # 2.4 Feature Engineering
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype('int8')
    df['SCHEDULED_HOUR'] = (df['SCHEDULED_DEPARTURE'] // 100).astype('int8')

    # ---------------------------------------------------------
    # 3. OTIMIZAÇÃO DE TIPOS
    # ---------------------------------------------------------
    print("🔧 Otimizando Tipos de Dados (Downcasting)...")
    
    cols_to_keep = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ORIGIN_STATE',
        'SCHEDULED_HOUR', 'DISTANCE', 'ARRIVAL_DELAY', 
        'ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE', 'TARGET'
    ]
    
    df_opt = df[cols_to_keep].copy()

    # Conversão de Objetos para Categorias
    cat_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ORIGIN_STATE']
    for col in cat_cols:
        df_opt[col] = df_opt[col].astype('category')

    # Numéricos para tipos menores
    df_opt['MONTH'] = df_opt['MONTH'].astype('int8')
    df_opt['DAY'] = df_opt['DAY'].astype('int8')
    df_opt['DAY_OF_WEEK'] = df_opt['DAY_OF_WEEK'].astype('int8')
    df_opt['SCHEDULED_HOUR'] = df_opt['SCHEDULED_HOUR'].astype('int8')
    df_opt['TARGET'] = df_opt['TARGET'].astype('int8')
    
    df_opt['DISTANCE'] = df_opt['DISTANCE'].astype('float32')
    df_opt['ARRIVAL_DELAY'] = df_opt['ARRIVAL_DELAY'].astype('float32')
    df_opt['ORIGIN_LATITUDE'] = df_opt['ORIGIN_LATITUDE'].astype('float32')
    df_opt['ORIGIN_LONGITUDE'] = df_opt['ORIGIN_LONGITUDE'].astype('float32')

    df_opt = df_opt.dropna()

    # ---------------------------------------------------------
    # 4. AUDITORIA FINAL E RELATÓRIO
    # ---------------------------------------------------------
    memoria_depois = get_memory_usage_mb(df_opt)
    dtypes_depois = df_opt.dtypes.astype(str).to_dict()
    
    print("\n" + "="*50)
    print("RELATÓRIO DE ENGENHARIA DE DADOS")
    print("="*50)
    print(f"1. Memória Original (CSV Raw):   {memoria_antes:.2f} MB")
    print(f"2. Memória Final (CSV Otimizado): {memoria_depois:.2f} MB")
    print(f"3. Redução de Consumo em RAM:     {100 * (1 - memoria_depois/memoria_antes):.2f}%")
    print("-" * 50)
    
    print("\n[MUDANÇA DE TIPOS DE DADOS]")
    print(f"{'COLUNA':<25} | {'ANTES (Bruto)':<15} | {'DEPOIS (Otimizado)':<15}")
    print("-" * 60)
    
    exemplos = ['AIRLINE', 'MONTH', 'ARRIVAL_DELAY', 'ORIGIN_AIRPORT']
    for col in exemplos:
        t_antes = dtypes_antes.get(col, 'object')
        t_depois = dtypes_depois.get(col, 'Removido')
        print(f"{col:<25} | {t_antes:<15} | {t_depois:<15}")

    # ---------------------------------------------------------
    # 5. CARGA E GERAÇÃO DE ARTEFATOS
    # ---------------------------------------------------------
    # Salvar em CSV ao invés de Parquet
    csv_final_path = os.path.join(PROCESSED_DIR, 'flights_ready.csv')
    df_opt.to_csv(csv_final_path, index=False)
    print(f"\n💾 Arquivo CSV Processado salvo: {csv_final_path}")

    # Gerar Gráfico de Memória
    plot_path = os.path.join(IMG_DIR, 'comparacao_memoria_etl.png')
    plotar_comparacao_memoria(memoria_antes, memoria_depois, plot_path)

    tempo_total = time.time() - start_global
    print(f"\n✅ ETL Concluído em {tempo_total:.2f} segundos.")

if __name__ == "__main__":
    run_etl_com_auditoria()