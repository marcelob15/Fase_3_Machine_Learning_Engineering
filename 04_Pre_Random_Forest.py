import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Carga Ultra-Rápida
print("Carregando dados preparados...")

flights_processed_path = os.path.join(base_dir, 'parquet', 'flights_processed.parquet')
df = pd.read_parquet(flights_processed_path)

# 2. Prepara Variáveis para Supervisionado
le = LabelEncoder()
for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['TARGET', 'LATITUDE', 'LONGITUDE'])
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Modelo Supervisionado (Random Forest)
print("Treinando Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")

# 4. Modelo Não Supervisionado (Clusterização de Aeroportos)
print("Executando K-Means...")
# Agrupar por aeroporto usando as coordenadas e a média de atraso real (que calculamos antes)
# Aqui podes usar o df original antes do LabelEncoding se preferires, ou agrupar os pontos únicos
df_geo = df[['LATITUDE', 'LONGITUDE']].drop_duplicates()
kmeans = KMeans(n_clusters=5, random_state=42).fit(df_geo)
df_geo['CLUSTER'] = kmeans.labels_

print("Modelagem concluída!")