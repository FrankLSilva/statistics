import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, Birch
from sklearn.mixture import GaussianMixture

# ---------- 1. Carregar dados ----------
df = pd.read_csv('fatura.csv')

# Se necessário, ative a linha abaixo para converter valores com vírgula
# df['Total'] = df['Total'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))

# ---------- 2. Preparar dados ----------
X = df[['Total']].values
X_scaled = StandardScaler().fit_transform(X)

# Criar eixo Y artificial (ruído leve para visualização 2D)
Y_fake = np.random.normal(loc=0, scale=0.1, size=X_scaled.shape[0])
X_2d = np.hstack((X_scaled, Y_fake.reshape(-1, 1)))

# ---------- 3. Algoritmos ----------
algorithms = [
    ('KMeans', KMeans(n_clusters=3, random_state=0)),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=3)),
    ('Agglomerative', AgglomerativeClustering(n_clusters=3)),
    ('OPTICS', OPTICS(min_samples=3)),
    ('Birch', Birch(n_clusters=3)),
    ('GaussianMixture', GaussianMixture(n_components=3, random_state=0))
]

# ---------- 4. Plotar e armazenar resultados ----------
fig, axs = plt.subplots(2, 3, figsize=(18, 8))
axs = axs.flatten()
results = []

for i, (name, algorithm) in enumerate(algorithms):
    try:
        labels = algorithm.fit_predict(X_scaled)
        axs[i].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=50, alpha=0.8, edgecolor='k')
        axs[i].set_title(f"{i+1}. {name}")
        axs[i].set_xlabel('Faturamento (escalado)')
        axs[i].set_ylabel('Eixo Y artificial')

        df_copy = df.copy()
        df_copy['cluster'] = labels
        results.append((name, df_copy))

    except Exception as e:
        axs[i].text(0.5, 0.5, f'Erro: {e}', ha='center', va='center')
        axs[i].set_title(f"{i+1}. {name}")
        print(f'Erro ao processar {name}: {e}')

plt.tight_layout()
plt.suptitle("Dispersão 2D dos Clusters com Faturamento", fontsize=16, y=1.02)
plt.show()

# ---------- 5. Escolher quais exportar ----------
print("\nAlgoritmos disponíveis para exportação:")
for i, (name, _) in enumerate(results):
    print(f"{i+1} - {name}")

ids = input("\nDigite os números dos algoritmos que deseja exportar (ex: 1,3,6): ")
ids = [int(x.strip()) for x in ids.split(',') if x.strip().isdigit()]

for idx in ids:
    if 1 <= idx <= len(results):
        name, df_result = results[idx - 1]
        filename = f'clusters_{name}.csv'
        df_result.to_csv(filename, index=False)
        print(f'CSV exportado: {filename}')
    else:
        print(f"Índice inválido: {idx}")
