# Fase 3: Machine Learning Engineering ✈️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este projeto faz parte da Pós-Tech em **Machine Learning Engineering**. O objetivo é analisar dados históricos de voos dos EUA para prever atrasos e identificar padrões geográficos de congestionamento, utilizando técnicas de aprendizado supervisionado e não supervisionado.

---

## 📋 Sobre o Projeto

O transporte aéreo é uma parte vital da infraestrutura global, mas os atrasos impactam milhões de passageiros todos os anos. Neste projeto, utilizamos o dataset público **Flight Delays and Cancellations** para desenvolver análises estatísticas, modelos preditivos e agrupamentos geográficos estratégicos.

### 🎯 Objetivos Principais

- **EDA (Exploração de Dados):** Identificação de padrões, outliers e tendências temporais.
- **Aprendizado Supervisionado:** Classificação de voos com atraso superior a 15 minutos (padrão FAA).
- **Aprendizado Não Supervisionado:** Clusterização de aeroportos por perfil de risco.
- **Engenharia de Dados:** Pipeline ETL otimizado (CSV → Parquet) para ganho de performance.

---

## 🏗️ Estrutura do Pipeline

O projeto foi dividido em 7 etapas sequenciais para garantir consistência e reprodutibilidade:

| Ordem | Script | Descrição |
|------:|--------|-----------|
| 01 | `01_eda.py` | Análise exploratória inicial |
| 02 | `02_graficos.py` | Geração de visualizações |
| 03 | `03_ETL.py` | Extração e transformação inicial |
| 04 | `04_Validacao.py` | Validação preliminar de algoritmos |
| 05 | `05_Limpeza.py` | Conversão otimizada CSV → Parquet |
| 06 | `06_Supervisionado_classificacao_atraso.py` | Modelo Random Forest |
| 07 | `07_Nao_Supervisionado_clusterizacao_geografica.py` | Clusterização com K-Means |

---

## 🛠️ Instalação e Configuração

### 1️⃣ Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd <nome-da-pasta>
```

### 2️⃣ Criar Ambiente Virtual

```bash
python -m venv .venv
```

### 3️⃣ Ativar Ambiente Virtual

**Windows (CMD)**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell)**
```bash
.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 4️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## 📥 Download do Dataset Principal

Devido ao limite de tamanho do GitHub, o arquivo principal `flights.csv` (≈ 564 MB) deve ser baixado manualmente.

🔗 **Download direto:**  
[Baixar flights.csv](https://drive.google.com/file/d/1ceyRYUzkF22E_PvwKBF0s2oyhWZpRGGN/view?usp=drive_link)

### 📁 Estrutura esperada da pasta `dataset/`

```
dataset/
├── flights.csv
├── airlines.csv
├── airports.csv
└── dicionario_dados_flights.pdf
```

> Os arquivos `airlines.csv`, `airports.csv` e o dicionário já estão incluídos no repositório.

---

## 🚀 Como Executar

Execute os scripts na ordem numérica:

```bash
python 01_eda.py
python 02_graficos.py
python 03_ETL.py
python 04_Validacao.py
python 05_Limpeza.py
python 06_Supervisionado_classificacao_atraso.py
python 07_Nao_Supervisionado_clusterizacao_geografica.py
```

A partir do script 03_ETL.py, os dados passam a ser armazenados no formato **Parquet** em vez de CSV. Essa mudança foi adotada para melhorar significativamente o desempenho do processamento de dados.

---

## 📂 Organização do Projeto

```
.
├── dataset/
├── parquet/
├── img_eda/
├── img_supervisionado/
├── img_nao_supervisionado/
├── 01_eda.py
├── 02_graficos.py
├── 03_ETL.py
├── 04_Validacao.py
├── 05_Limpeza.py
├── 06_Supervisionado_classificacao_atraso.py
├── 07_Nao_Supervisionado_clusterizacao_geografica.py
├── requirements.txt
└── README.md
```

---


## 📊 Resultados e Conclusões

---

### 📈 Análise Exploratória de Dados (EDA)

Nesta etapa, processamos **5.8 milhões de registros** para entender o comportamento da malha aérea. O foco foi identificar variáveis preditoras para a modelagem supervisionada. Os artefatos visuais gerados (`img_eda/`) revelaram os seguintes padrões:

- **`grafico_01_distribuicao.png` (Target Definition)**
  - **Insight:** A distribuição de atrasos possui uma cauda longa à direita (assimetria positiva).
  - **Decisão Técnica:** Definimos o *target* binário com corte em **15 minutos** (padrão FAA), pois tratar o problema como regressão pura seria prejudicado pelos *outliers* extremos (>3 horas), que representam 0.81% dos dados.

- **`grafico_02_companhias.png` (Operational Efficiency)**
  - **Insight:** Há uma clara distinção de performance entre modelos de negócio. Companhias *Low-Cost* (ex: Spirit, Frontier) apresentam taxas de atraso sistematicamente maiores que as *Legacy Carriers* (ex: Delta, Alaska), tornando a variável `AIRLINE` um forte preditor.

- **`grafico_03_aeroportos.png` (Infrastructure Bottlenecks)**
  - **Insight:** Aeroportos que funcionam como grandes *Hubs* de conexão (ex: **ORD/Chicago**, **EWR/Newark**) aparecem consistentemente como gargalos logísticos. A saturação da infraestrutura nestes locais aumenta a probabilidade de atraso independentemente da companhia aérea.

- **`grafico_04_horario.png` (Propagation Effect)**
  - **Insight:** A probabilidade de atraso não é linear. Observamos o **"Efeito Bola de Neve"**: voos pela manhã (06h-09h) têm pontualidade alta, enquanto voos noturnos (20h-23h) sofrem com o acúmulo de atrasos anteriores.
  - **Decisão Técnica:** A criação da feature `SCHEDULED_HOUR` (extraída do horário de partida) é indispensável para o modelo.


#### **A. Variáveis Ignoradas (E por quê?)**
Estas variáveis foram removidas para evitar **Data Leakage (Vazamento de Dados)**. Elas contêm informações que só ocorrem *após* o evento que queremos prever.
*   `DEPARTURE_TIME`, `DEPARTURE_DELAY`: Se o avião já saiu atrasado, é óbvio que chegará atrasado.
*   `WHEELS_OFF`, `WHEELS_ON`: Horários exatos da decolagem/pouso.
*   `TAXI_OUT`, `TAXI_IN`: Tempo de manobra na pista.
*   `ELAPSED_TIME`, `AIR_TIME`: Duração real do voo.
*   `CANCELLATION_REASON`: Só existe se o voo for cancelado.
*   `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, etc.: Explicam o atraso *depois* que ele ocorreu.

#### **B. Variáveis Consideradas (Features)**
Informações disponíveis no momento do planejamento/compra.
*   **Temporais:** `MONTH`, `DAY`, `DAY_OF_WEEK`, `HOUR` (Extraído de Scheduled Departure). *Tratamento: Mantidos numéricos.*
*   **Geográficas/Operacionais:** `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`. *Tratamento: Label Encoding (transformados em números inteiros).*
*   **Físicas:** `DISTANCE`. *Tratamento: Mantido numérico (normalizado em alguns modelos).*


---

### 🤖 Aprendizado Supervisionado (Classificação)

Modelo utilizado: **Random Forest Classifier**  
Estratégia para desbalanceamento: `class_weight='balanced'`

#### 📈 Métricas Principais

- **ROC-AUC Score:** 0.6719  
- **Recall (Atrasos):** 0.65  

O modelo demonstra boa capacidade de separação entre voos pontuais e atrasados.

#### 🔎 Interpretabilidade

- **Feature Importance:**  
  `img_supervisionado/feature_importance.png`  
  A variável `HOUR` representa ~48% da importância total, indicando que atrasos são cumulativos ao longo do dia.

- **Matriz de Confusão:**  
  `img_supervisionado/matriz_confusao.png`  
  Mostra equilíbrio entre falsos positivos e falsos negativos.

---

### 🧩 Aprendizado Não Supervisionado (Clusterização)

Algoritmo: **K-Means**

Foram identificados 4 clusters distintos com base em:
- Localização geográfica
- Taxa média de atraso
- Volume operacional

📍 **Insights:**
- Concentração de risco no Nordeste dos EUA
- Impacto significativo de grandes hubs como Chicago (ORD)

Visualização disponível em:
```
img_nao_supervisionado/mapa_clusters_geograficos.png
```

---


## 📖 Dicionário de Dados

| Coluna | Descrição | Tipo |
|--------|-----------|------|
| YEAR, MONTH, DAY | Data do voo | Inteiro |
| DAY_OF_WEEK | Dia da semana (1=Seg, 7=Dom) | Inteiro |
| AIRLINE | Código da companhia aérea | Categórica |
| ORIGIN_AIRPORT | Código IATA de origem | Categórica |
| DESTINATION_AIRPORT | Código IATA de destino | Categórica |
| DEPARTURE_DELAY | Atraso na partida (minutos) | Numérico |
| ARRIVAL_DELAY | Atraso na chegada (minutos) | Numérico |
| CANCELLED | 1 = cancelado | Binária |
| DISTANCE | Distância em milhas | Numérico |
| AIR_TIME | Tempo de voo em minutos | Numérico |