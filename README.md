# Tech Challenge - Fase 3: Machine Learning Engineering ✈️

Este projeto faz parte da Pós-Tech em Machine Learning Engineering. O objetivo é analisar dados históricos de voos dos EUA para prever atrasos e identificar padrões geográficos de congestionamento, utilizando técnicas de aprendizado supervisionado e não supervisionado.

## 📋 Estrutura do Projeto

O projeto foi modularizado em um pipeline de 7 etapas para garantir eficiência, organização e escalabilidade:

1.  **`01_eda.py`**: Análise Exploratória de Dados inicial e identificação de pontos críticos.
2.  **`02_graficos.py`**: Geração de visualizações para entendimento de padrões e sazonalidade.
3.  **`03_ETL.py`**: Extração, Transformação e Carga inicial dos dados.
4.  **`04_Pre_Random_Forest.py`**: Testes preliminares e validação de algoritmos.
5.  **`05_Limpeza_Pesada.py`**: Pipeline de engenharia de dados (CSV para Parquet) com tipagem otimizada.
6.  **`06_Supervisionado_classificacao_atraso.py`**: Treinamento do modelo de classificação (Random Forest).
7.  **`07_Nao_Supervisionado_clusterizacao_geografica.py`**: Agrupamento de aeroportos via K-Means.

---

## 🛠️ Instalação e Configuração

### 1. Criar Ambiente Virtual
```bash
python -m venv venv

2. Ativar Ambiente Virtual
Windows (CMD): venv\Scripts\activate

Windows (PowerShell): venv\Scripts\Activate.ps1

Linux / macOS: source venv/bin/activate

3. Instalar Dependências

python -m pip install pandas numpy matplotlib seaborn pyarrow scikit-learn
pip freeze > requirements.txt

🚀 Como Executar
Siga a ordem sequencial dos scripts para garantir que os arquivos processados e imagens sejam gerados corretamente:

python 01_eda.py
python 02_graficos.py
python 03_ETL.py
python 04_Pre_Random_Forest.py
python 05_Limpeza_Pesada.py
python 06_Supervisionado_classificacao_atraso.py
python 07_Nao_Supervisionado_clusterizacao_geografica.py


O Gemini disse
Para fechar o seu projeto com chave de ouro, o README.md precisa ser claro, profissional e destacar o rigor técnico que você aplicou. Abaixo está uma estrutura completa que documenta desde a instalação até as conclusões do modelo.

Crie um arquivo chamado README.md na raiz do seu projeto e cole o conteúdo abaixo:

Markdown
# Tech Challenge - Fase 3: Machine Learning Engineering ✈️

Este projeto faz parte da Pós-Tech em Machine Learning Engineering. O objetivo é analisar dados históricos de voos dos EUA para prever atrasos e identificar padrões geográficos de congestionamento, utilizando técnicas de aprendizado supervisionado e não supervisionado.

## 📋 Estrutura do Projeto

O projeto foi modularizado em um pipeline de 7 etapas para garantir eficiência, organização e escalabilidade:

1.  **`01_eda.py`**: Análise Exploratória de Dados inicial e identificação de pontos críticos.
2.  **`02_graficos.py`**: Geração de visualizações para entendimento de padrões e sazonalidade.
3.  **`03_ETL.py`**: Extração, Transformação e Carga inicial dos dados.
4.  **`04_Pre_Random_Forest.py`**: Testes preliminares e validação de algoritmos.
5.  **`05_Limpeza_Pesada.py`**: Pipeline de engenharia de dados (CSV para Parquet) com tipagem otimizada.
6.  **`06_Supervisionado_classificacao_atraso.py`**: Treinamento do modelo de classificação (Random Forest).
7.  **`07_Nao_Supervisionado_clusterizacao_geografica.py`**: Agrupamento de aeroportos via K-Means.

---

## 🛠️ Instalação e Configuração

### 1. Criar Ambiente Virtual
```bash
python -m venv venv
2. Ativar Ambiente Virtual
Windows (CMD): venv\Scripts\activate

Windows (PowerShell): venv\Scripts\Activate.ps1

Linux / macOS: source venv/bin/activate

3. Instalar Dependências
Bash
python -m pip install pandas numpy matplotlib seaborn pyarrow scikit-learn
pip freeze > requirements.txt
🚀 Como Executar
Siga a ordem sequencial dos scripts para garantir que os arquivos processados e imagens sejam gerados corretamente:

Bash
python 01_eda.py
python 02_graficos.py
python 03_ETL.py
python 04_Pre_Random_Forest.py
python 05_Limpeza_Pesada.py
python 06_Supervisionado_classificacao_atraso.py
python 07_Nao_Supervisionado_clusterizacao_geografica.py
📊 Resultados e Conclusões
Aprendizado Supervisionado (Classificação)
Utilizamos o algoritmo Random Forest Classifier com balanceamento de classes para prever atrasos superiores a 15 minutos (padrão FAA).

ROC-AUC Score: 0.6719

Recall (Atrasos): 0.65 (O modelo identifica 65% dos atrasos reais).

Feature Importance: A variável HORA (HOUR) foi o preditor mais forte (~48%), confirmando o efeito cascata de atrasos ao longo do dia.

Aprendizado Não Supervisionado (Clusterização)
O algoritmo K-Means agrupou os aeroportos em 4 clusters distintos baseados em localização geográfica e taxa de atraso média.

Identificamos "zonas de risco" concentradas no Nordeste e em grandes hubs centrais como Chicago (ORD).

📂 Organização de Arquivos
/dataset: Contém os arquivos CSV originais (não inclusos no Git devido ao tamanho).

/parquet: Armazena o dataset processado e otimizado.

/img_eda: Visualizações da análise exploratória.

/img_supervisionado: Matriz de confusão e gráficos de importância de variáveis.

/img_nao_supervisionado: Mapas de clusterização geográfica.

Desenvolvido como parte do currículo de Machine Learning Engineering.



### Dica para o Repositório:
Ao enviar para o GitHub, lembre-se de criar um arquivo `.gitignore` para não subir o arquivo `flights.csv` (que tem 500MB) nem a pasta `venv/`.

**Exemplo de `.gitignore`:**
```text
venv/
__pycache__/
*.csv
*.parquet
.DS_Store
