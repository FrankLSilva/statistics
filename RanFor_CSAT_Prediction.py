import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Random Forest
# Previsão de CSAT para o mês seguinte
# [Necessário: Coluna de DATA da criação do CSAT + Coluna do CSAT + Remover nulos do CSAT]

# Variáveis (nomear as colunas do treinamento de modelo)
DATASET = 'csat.csv'
DATE = 'Data criação'
CSAT = 'CSAT'

# Carregar dataset / converter
df = pd.read_csv(DATASET)
df[DATE] = pd.to_datetime(df[DATE], dayfirst=True)

# Agrupar por mês e calcular a média do CSAT
df_monthly = df.groupby(df[DATE].dt.to_period('M'))[CSAT].mean().reset_index()
df_monthly[DATE] = df_monthly[DATE].astype(str)  # Converter de Period para string

# Criar variáveis de lag (CSAT dos meses anteriores)
df_monthly['CSAT_1M_Atras'] = df_monthly[CSAT].shift(1)
df_monthly['CSAT_2M_Atras'] = df_monthly[CSAT].shift(2)
df_monthly['CSAT_3M_Atras'] = df_monthly[CSAT].shift(3)

# Exibir tamanho do dataframe após a criação das variáveis de lag
print(f"Tamanho após criar variáveis de lag: {len(df_monthly)}")

# eparar features e target
X = df_monthly[['CSAT_1M_Atras', 'CSAT_2M_Atras', 'CSAT_3M_Atras']]
y = df_monthly[CSAT]

# Verificar se há dados suficientes para o treinamento
if len(X) < 2:  # Se não houver dados suficientes, não dividir em treino e teste
    print("Não há dados suficientes para realizar o treinamento.")

else:
    # Separar dados em treino e teste (últimos 3 meses para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar modelo Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Fazer previsão para os últimos meses do dataset
    y_pred = rf.predict(X_test)

    # Avaliar erro
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n📉 Erro Médio Absoluto (MAE): {mae:.2f}")

    # Exibir CSAT de todos os meses anteriores
    print("\n📊 CSAT dos meses anteriores:")
    for i in range(len(df_monthly)):  # Mostrar todos os meses anteriores
        print(f"{df_monthly[DATE].iloc[i]}: {df_monthly[CSAT].iloc[i]:.2f}")

    # Prever CSAT do próximo mês
    proxima_entrada = pd.DataFrame([X.iloc[-1].values], columns=X.columns)
    proxima_previsao = rf.predict(proxima_entrada)

    # Mostrar previsão
    print(f"\n🔮 Previsão do CSAT para o próximo mês: {proxima_previsao[0]:.2f}")