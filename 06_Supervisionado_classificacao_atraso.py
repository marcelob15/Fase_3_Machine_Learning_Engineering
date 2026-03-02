import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Configurações de Diretório
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'img_supervisionado')
os.makedirs(output_dir, exist_ok=True)

# 1. Carga do Parquet
print("📂 Carregando dados para Classificação...")
flights_ready_path = os.path.join(base_dir, 'parquet', 'flights_ready.parquet')
df = pd.read_parquet(flights_ready_path)

# 2. Label Encoding
le = LabelEncoder()
for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
    df[col] = le.fit_transform(df[col])

# 3. Treinamento (Random Forest)
print("🤖 Treinando Random Forest (Supervisionado)...")
X = df[['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'HOUR', 'DISTANCE']]
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 4. Métricas
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
print(f"\n✅ ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred))

# 5. Gráfico: Matriz de Confusão

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Predição de Atraso')
plt.savefig(f'{output_dir}/matriz_confusao.png')

# 6. Gráfico: Feature Importance

df_imp = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', palette='magma', legend=False)
plt.title('Importância das Variáveis na Predição', fontsize=14, fontweight='bold')
plt.savefig(f'{output_dir}/feature_importance.png', bbox_inches='tight')

print(f"✨ Modelagem Supervisionada concluída! Imagens em: {output_dir}")