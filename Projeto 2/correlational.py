import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import seaborn as sns # Para um plot mais bonito da matriz de correlação
from model_tools import load_csv, normalize, create_nonlinear_features

# --- Início do seu código modificado ---
Xy_raw = load_csv('dados/Concreto - treino.csv')
y_target = Xy_raw[:, -1].reshape(-1, 1) # Garante que y_target seja (N, 1)
X_original_features = Xy_raw[:, :-1]

# 1. Criar features não lineares
X_engineered = create_nonlinear_features(X_original_features)

# 2. Combinar as features engenheiradas com a variável alvo para calcular a correlação
#    É importante que a correlação seja calculada ANTES da normalização Z-score
#    se você quiser a correlação nos valores "reais" das features.
#    Ou, se quiser a correlação das features já normalizadas (o que também é válido),
#    faça a normalização primeiro e depois calcule a correlação.
#    Vamos calcular nas features engenheiradas ANTES da normalização Z-score
#    para entender as relações intrínsecas.

data_for_corr = np.hstack((X_engineered, y_target))

# 3. Calcular a matriz de correlação usando Pandas para facilidade
#    Pandas lida bem com os rótulos e calcula a correlação de Pearson corretamente.
num_original_features = X_original_features.shape[1]
num_engineered_features = X_engineered.shape[1]

# Gerar nomes das features baseado na implementação real de create_nonlinear_features
feature_names = []

# Features originais
for i in range(num_original_features):
    feature_names.append(f'F{i+1}')

# Features quadráticas (x²)
for i in range(num_original_features):
    feature_names.append(f'F{i+1}_sq')

# Features cúbicas (x³)
for i in range(num_original_features):
    feature_names.append(f'F{i+1}_cub')

# Features logarítmicas (log(x))
for i in range(num_original_features):
    feature_names.append(f'F{i+1}_log')

# Features de interação (xi * xj para i < j)
for i in range(num_original_features):
    for j in range(i+1, num_original_features):
        feature_names.append(f'F{i+1}_x_F{j+1}')

# Verificar se o número de nomes corresponde ao número de features engenheiradas
print(f"Número de features engenheiradas: {num_engineered_features}")
print(f"Número de nomes gerados: {len(feature_names)}")

# Ajustar se necessário (fallback para nomes genéricos)
if len(feature_names) != num_engineered_features:
    feature_names = [f'Feature_{i+1}' for i in range(num_engineered_features)]

final_feature_names = feature_names + ['Target_y']


df_for_corr = pd.DataFrame(data_for_corr, columns=final_feature_names)
corr_matrix = df_for_corr.corr() # Calcula a matriz de correlação de Pearson

print("Correlation matrix shape:", corr_matrix.shape)
print("Correlation matrix:\n", corr_matrix)

# 4. Plotar a matriz de correlação
plt.figure(figsize=(12, 10)) # Ajuste o tamanho conforme o número de features
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)
plt.title("Matriz de Correlação das Features e Variável Alvo")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout() # Ajusta o layout para não cortar os rótulos
# Crie o diretório se não existir
Path("graficos").mkdir(parents=True, exist_ok=True)
plt.savefig("graficos/correlation_matrix_features_target.png", dpi=300)
plt.close()

# 5. Normalizar as features engenheiradas para o modelo
#    (esta parte é para o seu fluxo de modelagem, não para a matriz de correlação acima)
X_norm, mean_norm, std_norm = normalize(X_engineered)
# y_target já está separado

print("\n--- Matriz X_norm pronta para o modelo ---")
print("X_norm shape:", X_norm.shape)
