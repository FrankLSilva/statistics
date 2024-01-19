import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.compat import lzip

warnings.simplefilter(action='ignore', category=FutureWarning)
print("\nSTATS MODELS | OLS Linear Regression\n")

# ----------------------------- DATASETS
# df = sm.datasets.get_rdataset("Guerry", "HistData").data
# df = pd.read_csv("../Data/df_z.csv")
df = pd.read_csv("../Data/df_q.csv")

print(f'{df.columns}\n')

# ----------------------------- COLUMNS
dep = 'Suicides' # Dependent variable
col1 = 'Wealth'
col2 = 'Prostitutes'
col3 = 'Distance'
col4 = 'Literacy'

# ----------------------------- MODEL
df2 = df[[dep, col1, col2, col3, col4]].dropna()
y = df2[[dep]]
x = df2[[col2]]
print(f'{round(df2.describe())}\n')

# ----------------------------- FIT
mod = sm.OLS(y, x).fit()
print(mod.summary())

# ----------------------------- GRAPHS | PLOTS
exog_idx = col2 # Change col to drive the diagnostics from 1-3

ploting = True
while ploting:
    print(f"\nSelected Regression Exog: {col1}\n")
    plot_choose= int(input("Choose plot category:\n"
                       "1. Single Variable Regression Diagnostics {exog}\n"
                       "2. Plot fit against one regressor {exog}\n"
                       "3. Plot of influence in regression {exog}\n"
                       "4. Complete Pairgrid Plot {df2}\n"))

    doc_choose = input("Show DOC (0/1):\n").lower()

    plt.rc("figure", figsize=(12, 8))
    plt.rc("font", size=10)

    # ----------------------------- Single Variable Regression Diagnostics
    if plot_choose == 1:
        fig = sm.graphics.plot_regress_exog(mod, exog_idx)
        fig.tight_layout(pad=1.0)
        plt.show()
        if doc_choose == "1":
            print(sm.graphics.plot_regress_exog.__doc__)

    # ----------------------------- Fit Plot
    elif plot_choose == 2:
        fig = sm.graphics.plot_fit(mod, exog_idx)
        fig.tight_layout(pad=1.0)
        plt.show()
        if doc_choose == "1":
            print(sm.graphics.plot_fit.__doc__)

    # ----------------------------- Influence Plot
    elif plot_choose == 3:
        fig = sm.graphics.influence_plot(mod, exog_idx)
        fig.tight_layout(pad=1.0)
        plt.show()
        if doc_choose == "1":
            print(sm.graphics.influence_plot.__doc__)

    # ----------------------------- Complete Pairgrid Plot
    elif plot_choose == 4:
        g = sns.PairGrid(df2, diag_sharey=False, corner=True)
        g.fig.set_size_inches(12, 8)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot)
        plt.show()

    else:
        ploting = False