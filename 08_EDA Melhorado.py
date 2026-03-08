"""
EDA Melhorado + Análise por Estado + Ajuste de Recall
Este script faz exatamente o que você pediu: gera a tabela de tipos/nulos, faz a análise de volume por estado e aplica o ajuste fino no Recall.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix

# Criar pasta para salvar imagens se não existir
os.makedirs('img_eda', exist_ok=True)
os.makedirs('img_supervisionado', exist_ok=True)

# --- 1. CARREGAMENTO E ANÁLISE DE ESTRUTURA ---
print("📂 Carregando dataset...")
try:
    df = pd.read_parquet('parquet/flights_ready.parquet')
except:
    # Fallback caso a pasta parquet não exista no diretório atual
    df = pd.read_parquet('flights_ready.parquet')

print("\n=== ANÁLISE DE VARIÁVEIS E TIPOS ===")
info_df = pd.DataFrame({
    'Tipo de Dado': df.dtypes,
    'Qtd Nulos': df.isnull().sum(),
    '% Nulos': (df.isnull().sum() / len(df)) * 100
}).sort_values(by='% Nulos', ascending=False)

print(info_df.head(10))

# --- 2. ANÁLISE POR ESTADO (VOLUME vs ATRASO) ---
print("\n📊 Gerando Análise por Estado...")

if 'ORIGIN_STATE' in df.columns:
    df_estado = df.groupby('ORIGIN_STATE').agg({
        'MONTH': 'count',           # Contagem de voos
        'ARRIVAL_DELAY': 'mean'     # Média de atraso
    }).reset_index()
    
    df_estado.columns = ['ESTADO', 'VOLUME_VOOS', 'MEDIA_ATRASO']
    df_estado = df_estado[df_estado['VOLUME_VOOS'] > 1000] 
    df_estado = df_estado.sort_values('MEDIA_ATRASO', ascending=False)

    fig, ax1 = plt.subplots(figsize=(14, 8))

    sns.barplot(data=df_estado, x='ESTADO', y='MEDIA_ATRASO', color='salmon', alpha=0.7, ax=ax1)
    ax1.set_ylabel('Média de Atraso (min)', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_title('Impacto do Volume no Atraso por Estado', fontsize=16)

    ax2 = ax1.twinx()
    sns.lineplot(data=df_estado, x='ESTADO', y='VOLUME_VOOS', color='blue', marker='o', ax=ax2, linewidth=2)
    ax2.set_ylabel('Quantidade de Voos', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.tight_layout()
    plt.savefig('img_eda/analise_estados_volume.png')
    print("✅ Gráfico de estados salvo em img_eda/!")
    # plt.show() # Comentei para não travar a execução se rodar direto
else:
    print("⚠️ Coluna ORIGIN_STATE não encontrada. Pulando gráfico de estados.")

# --- 3. PRÉ-PROCESSAMENTO (CORREÇÃO DO ERRO) ---
print("\n⚙️ Convertendo texto em números (Label Encoding)...")

# Garantir que a TARGET existe. Se não existir, cria agora.
if 'TARGET' not in df.columns:
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# Codificação das variáveis categóricas
cols_to_encode = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
le = LabelEncoder()

for col in cols_to_encode:
    # Converte para string primeiro para evitar erro se tiver misturado
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    print(f"   > Coluna {col} codificada.")

# --- 4. MODELAGEM E AJUSTE FINO DO RECALL ---
print("\n🤖 Iniciando Modelagem com Ajuste de Threshold...")

cols_features = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'HOUR', 'DISTANCE']

# Verificar se todas as colunas existem
missing_cols = [c for c in cols_features if c not in df.columns]
if missing_cols:
    print(f"❌ Erro: Colunas faltando no dataset: {missing_cols}")
    # Tentar corrigir HOUR se faltar (geralmente extraído de SCHEDULED_DEPARTURE)
    if 'HOUR' in missing_cols and 'SCHEDULED_DEPARTURE' in df.columns:
        print("   -> Criando coluna HOUR a partir de SCHEDULED_DEPARTURE...")
        df['HOUR'] = (df['SCHEDULED_DEPARTURE'] // 100).astype(int)
        cols_features = [c for c in cols_features if c != 'HOUR'] + ['HOUR']

X = df[cols_features]
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo Balanceado
rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Probabilidades
y_proba = rf.predict_proba(X_test)[:, 1]

# --- O GRANDE TRUQUE: ESCOLHER O LIMIAR ---
LIMIAR_PERSONALIZADO = 0.45 
y_pred_custom = (y_proba >= LIMIAR_PERSONALIZADO).astype(int)

print(f"\n✅ Resultados com Limiar {LIMIAR_PERSONALIZADO}:")
print(classification_report(y_test, y_pred_custom))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Gráfico Curva Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Random Forest', markevery=500) # markevery para não ficar pesado
plt.title('Curva Precision-Recall')
plt.xlabel('Recall (Capacidade de detectar atrasos)')
plt.ylabel('Precision (Confiabilidade do alerta)')
plt.legend()
plt.grid()
plt.savefig('img_supervisionado/precision_recall_curve.png')
print("✅ Gráfico Precision-Recall salvo!")

print("\n🚀 Script finalizado com sucesso!")