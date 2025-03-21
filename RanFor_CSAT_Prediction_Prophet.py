import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from prophet import Prophet

# Previsão de CSAT para o mês seguinte
# [Necessário: Coluna de DATA da criação do CSAT + Coluna do CSAT + Remover nulos do CSAT]

# Variáveis (nomear as colunas para o treinamento de modelo)
DATASET = 'csat.csv'
DATE = 'Data criação'
CSAT = 'CSAT'

# Carregar dataset / converter
df = pd.read_csv(DATASET)
df[DATE] = pd.to_datetime(df[DATE], dayfirst=True)

# Agrupar por mês e calcular a média do CSAT
df_monthly = df.groupby(df[DATE].dt.to_period('M'))[CSAT].mean().reset_index()
df_monthly[DATE] = df_monthly[DATE].astype(str)  # Converter de Period para string

df_prophet = df_monthly.rename(columns={DATE: 'ds', CSAT: 'y'})

# Separar dados em treino e teste para Prophet
train_size = int(len(df_prophet) * 0.8)
df_train = df_prophet.iloc[:train_size]
df_test = df_prophet.iloc[train_size:]

# Treinar modelo Prophet
model = Prophet()
model.fit(df_train)

# Criar dataframe para previsão
future = model.make_future_dataframe(periods=len(df_test), freq='M')
forecast = model.predict(future)

y_pred_prophet = forecast['yhat'].iloc[-len(df_test):].values
y_test_prophet = df_test['y'].values
mae_prophet = mean_absolute_error(y_test_prophet, y_pred_prophet)

# Criar variáveis de lag (CSAT dos meses anteriores)
df_monthly['CSAT_1M_Atras'] = df_monthly[CSAT].shift(1)
df_monthly['CSAT_2M_Atras'] = df_monthly[CSAT].shift(2)
df_monthly['CSAT_3M_Atras'] = df_monthly[CSAT].shift(3)

# Exibir tamanho do dataframe após a criação das variáveis de lag
print("\nRandom Forest | Previsão de CSAT")
print("--------------------------------")
print(f"\nTamanho após criar variáveis de lag: {len(df_monthly)}")

# Separar features e target
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
    y_pred_rf = rf.predict(X_test)

    # Avaliar erro
    mae_rf = mean_absolute_error(y_test, y_pred_rf)


    # Exibir CSAT de todos os meses anteriores
    print("\nCSAT dos meses anteriores:")
    for i in range(len(df_monthly)):  # Mostrar todos os meses anteriores
        print(f"{df_monthly[DATE].iloc[i]}: {df_monthly[CSAT].iloc[i]:.2f}")

    # Prever CSAT do próximo mês
    proxima_entrada = pd.DataFrame([X.iloc[-1].values], columns=X.columns)
    proxima_previsao_rf = rf.predict(proxima_entrada)

    # Mostrar previsão
    print(f"\nErro Médio Absoluto (MAE) - Random Forest: {mae_rf:.2f}")
    print(f"📊 Previsão do CSAT para o próximo mês (Random Forest): {proxima_previsao_rf[0]:.2f}")

    # Exibir previsão para o próximo mês com Prophet
    proxima_previsao = forecast[['ds', 'yhat']].iloc[-1]
    print(f"\nErro Médio Absoluto (MAE) - Prophet: {mae_prophet:.2f}")
    print(f"📊 Previsão do CSAT para o próximo mês (Prophet): {proxima_previsao['yhat']:.2f}")
