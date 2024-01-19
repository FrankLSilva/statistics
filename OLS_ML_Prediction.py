import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)
print("\nMACHINE LEARNING | REGRESSION + PREDICTION MODEL\n")

# ----------------------------- DATASET SAMPLE
# df = pd.read_csv('../dataset/sample_model.csv').dropna()
# df = pd.read_csv('../data/transport_E.csv')

print(f'{df.info()}\n')
print(f'{df.describe()}\n')

predict_column = 'Collects'
target_column = 'Faults'

# ----------------------------- DATA PREPARATION
predictor = df[predict_column] # Predictor variable
target = df[target_column] # Target variable

x = np.array(predictor) # Entry variable
x = x.reshape(-1, 1)
y = target

# ----------------------------- TRAIN / TEST / MODEL
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)


ploting = True
while ploting:
    plot_choose = int(input("\nChoose plot category:\n"
                            "1. Predictor KDE Plot\n"
                            "2. Scatter Plot\n"
                            "3. Scatter Plot + Regression\n"
                            "4. New Value + Prediction Deploy\n"))

    # ----------------------------- PREDICTOR KDE PLOT
    if plot_choose == 1:
        sns.histplot(data=df, x=predictor, kde=True)
        plt.show()

    # ----------------------------- SCATTER PLOT
    elif plot_choose == 2:
        plt.scatter(x, y, label="Dados reais históricos")
        plt.xlabel(predict_column)
        plt.ylabel(target_column)
        plt.legend()
        plt.show()

    # ----------------------------- SCATTER PLOT / REGRESSION
    elif plot_choose == 3:
        plt.scatter(x, y, label="Dados reais históricos")
        plt.plot(x, model.predict(x), color="red", label="Regressão")
        plt.xlabel(predict_column)
        plt.ylabel(target_column)
        plt.legend()
        plt.show()

    else:
        ploting = False

    # ----------------------------- DEPLOY
    deploy = True
    while deploy:
        if plot_choose == 4:
            new_value = int(input(f'Insert new value for Prediction On {predict_column}:\n'))
            v = np.array([[new_value]])
            prediction = model.predict(v)
            print(f'\nValue = {v} \nPrediction = {prediction}\n')
            print("-----------------------------------")
        else:
            deploy = False




