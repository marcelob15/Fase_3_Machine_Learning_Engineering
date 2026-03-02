# Fase 3: Machine Learning Engineering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este projeto faz parte da Pós-Tech em **Machine Learning Engineering**. O objetivo é analisar dados históricos de voos dos EUA para prever atrasos e identificar padrões geográficos de congestionamento, utilizando técnicas de aprendizado supervisionado e não supervisionado.

---

## 📋 Sobre o Projeto

O transporte aéreo é uma parte vital da infraestrutura global, mas os atrasos de voos impactam milhões de passageiros todos os anos. Neste projeto, utilizamos o conjunto de dados público **Flight Delays and Cancellations** para desenvolver análises e modelos preditivos.

### 🎯 Objetivos Principais

- **EDA:** Exploração estatística e visualização de dados  
- **Aprendizado Supervisionado:** Prever se um voo vai atrasar (Classificação)  
- **Aprendizado Não Supervisionado:** Agrupar aeroportos com perfis semelhantes (Clusterização)  
- **Engenharia de Dados:** Pipeline ETL otimizado (CSV → Parquet)  

---

## 🏗️ Estrutura do Pipeline

O projeto foi modularizado em um pipeline de 7 etapas para garantir eficiência, organização e escalabilidade:

| Ordem | Script | Descrição |
|------:|--------|-----------|
| 01 | `01_eda.py` | Análise Exploratória de Dados inicial e identificação de pontos críticos |
| 02 | `02_graficos.py` | Geração de visualizações para entendimento de padrões e sazonalidade |
| 03 | `03_ETL.py` | Extração, Transformação e Carga inicial dos dados |
| 04 | `04_Validacao.py` | Testes preliminares e validação de algoritmos |
| 05 | `05_Limpeza.py` | Pipeline de engenharia de dados (CSV → Parquet) com tipagem otimizada |
| 06 | `06_Supervisionado_classificacao_atraso.py` | Treinamento do modelo de classificação (Random Forest) |
| 07 | `07_Nao_Supervisionado_clusterizacao_geografica.py` | Agrupamento de aeroportos via K-Means |

---

## 🛠️ Instalação e Configuração

### 1. Clonar o Repositório
```bash
git clone <url-do-repositorio>
cd <nome-da-pasta>
```

### 2. Criar Ambiente Virtual
```bash
python -m venv .venv
```

### 3. Ativar Ambiente Virtual

Windows (CMD):
```bash
.venv\Scripts\activate
```

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

Linux / macOS:
```bash
source .venv/bin/activate
```

### 4. Instalar Dependências
```bash
pip install -r requirements.txt
```

É necessário fazer download do arquivo flights.csv para a pasta `dataset`. Pois ser tamanho é 564 MB.
Link
https://drive.google.com/file/d/1ceyRYUzkF22E_PvwKBF0s2oyhWZpRGGN/view?usp=drive_link

Os demias arquivos já estão pasta `dataset`
airlines.csv
airports.csv
dicionario_dados_flights.pdf

---

## 🚀 Como Executar

Siga a ordem sequencial dos scripts para garantir que os arquivos processados e imagens sejam gerados corretamente:

Para melhorar a melhorar o processamento, o arquivo flights.csv foi transformado para o formato Parquet.
Com isto ganhamos velocidade no processamento. O arquivo 03_ETL.py faz esta trasnformação.


```bash
python 01_eda.py
python 02_graficos.py
python 03_ETL.py
python 04_Validacao.py
python 05_Limpeza.py
python 06_Supervisionado_classificacao_atraso.py
python 07_Nao_Supervisionado_clusterizacao_geografica.py
```

---

## 📊 Resultados e Conclusões

### 🤖 Aprendizado Supervisionado (Classificação)

Utilizamos o algoritmo Random Forest Classifier com balanceamento de classes para prever atrasos superiores a 15 minutos (padrão FAA).

- ROC-AUC Score: 0.6719  
- Recall (Atrasos): 0.65 (O modelo identifica 65% dos atrasos reais)  
- Feature Importance: A variável HORA (engenharia de feature baseada no horário de partida) foi o preditor mais forte (~48%), confirmando o efeito cascata de atrasos ao longo do dia  

### 🧩 Aprendizado Não Supervisionado (Clusterização)

O algoritmo K-Means agrupou os aeroportos em 4 clusters distintos baseados em localização geográfica e taxa de atraso média.

Insight: Identificamos "zonas de risco" concentradas no Nordeste e em grandes hubs centrais como Chicago (ORD).

---

## 📂 Organização de Arquivos

```
.
├── dataset/                # Contém os arquivos CSV originais (flights.csv via download no link)
├── parquet/                # Armazena o dataset processado e otimizado para ML
├── img_eda/                # Visualizações da análise exploratória
├── img_supervisionado/     # Matriz de confusão e gráficos de importância de variáveis
├── img_nao_supervisionado/ # Mapas de clusterização geográfica
├── 01_eda.py
├── 02_graficos.py
├── 03_ETL.py
├── 04_Pre_Random_Forest.py
├── 05_Limpeza_Pesada.py
├── 06_Supervisionado_classificacao_atraso.py
├── 07_Nao_Supervisionado_clusterizacao_geografica.py
├── requirements.txt
└── README.md
```

---

## 📖 Dicionário de Dados

Principais colunas utilizadas baseadas no arquivo `dicionario_dados_flights.pdf`:

| Coluna | Descrição | Tipo |
|--------|-----------|------|
| YEAR, MONTH, DAY | Data do voo | Inteiro |
| DAY_OF_WEEK | Dia da semana (1=Seg, 7=Dom) | Inteiro |
| AIRLINE | Código da companhia aérea (ex: AA, DL) | Categórica |
| ORIGIN_AIRPORT | Código IATA do aeroporto de origem | Categórica |
| DESTINATION_AIRPORT | Código IATA do aeroporto de destino | Categórica |
| DEPARTURE_DELAY | Atraso na partida (em minutos) | Numérico |
| ARRIVAL_DELAY | Atraso na chegada (em minutos) | Numérico |
| CANCELLED | Indica se o voo foi cancelado (1= sim, 0= não) | Binária |
| DISTANCE | Distância entre origem e destino (em milhas) | Numérico |
| AIR_TIME | Tempo no ar (em minutos) | Numérico |