import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

print("\n----- OUTLIERS DROP 1 COL | Z-SCORE + IQR -----\n")
lin = "\n-----------------------------------------\n"

# ----------------------------- THRESHOLDS
z_score_threshold = 2
iqr_threshold = 1.5

# ----------------------------- DATASETS
df = pd.read_csv(f"../Data/Guerry_HistData.csv")
# df = sm.datasets.get_rdataset("Guerry", "HistData").data

print(f'{df.info()}\n')
print(f'{df.describe()}')
shape = df.shape

# ----------------------------- RUN
print(lin)
col_choose = input("Set column for Outliers check:\n")

ploting = True
while ploting:
    print(lin)
    plot_choose = int(input(f"Choose outcome for -> {col_choose}\n"
                            f"(Z-THR = {z_score_threshold}, IQR-THR = {iqr_threshold}):\n\n"
                            "1. View -> Boxplot\n"
                            "2. Generate -> Z-Score + IQR\n"
                            "3. Drop outliers -> Z-Score method\n"
                            "4. Drop outliers -> IQR method\n"
                            "5. View methods 'Key Differences'\n"))

    # ----------------------------- CHOOSE
    if plot_choose == 1:
        sns.boxplot(data=df[col_choose], width=.2)
        sns.stripplot(data=df[col_choose], size=4, linewidth=0, color=".2")
        plt.show()

    elif plot_choose == 2:

        # Z-Score Values
        print(lin)
        print("Z-Score values:")
        z = round(np.abs(stats.zscore(df[col_choose])), 2)
        print(z.tolist())

        # IQR Values
        Q1 = np.percentile(df[col_choose], 25, method='midpoint')
        Q3 = np.percentile(df[col_choose], 75, method='midpoint')
        IQR = Q3 - Q1

        # Define upper and lower bounds
        upper = Q3 + iqr_threshold * IQR
        lower = Q1 - iqr_threshold * IQR

        print(f"\nInter Quartile Range: {IQR}")
        print(f"Upper Bound: {upper}")
        print(f"Lower Bound: {lower}")

    elif plot_choose == 3:

        # Z-score Method:

        # Approach: The Z-score method standardizes the data by measuring how many...
        # standard deviations a data point is from the mean. Outliers are identified based on...
        # a specified Z-score threshold. Typically, values with an absolute Z-score above...
        # a certain threshold (e.g., 2 or 3) are considered outliers.

        # Assumption: This method assumes that the data follows a normal distribution.
        # It is more sensitive to extreme values compared to the IQR method.

        z_scores = np.abs(stats.zscore(df[col_choose]))
        outliers_mask = z_scores > z_score_threshold

        # Removing outliers using Z-Score method
        df_z = df[~outliers_mask]

        # Print Shapes
        print(lin)
        print(f"----> Old Shape: {shape}")
        print(f"----> New Shape (Z-Score): {df_z.shape}\n")
        print(f'{df_z.info()}\n')
        print(f'{df_z.describe()}\n\n')
        print("New Dataframe -> df_z")
        print(df_z.head())

        # CSV Export
        export_path = "../Data/df_z.csv"
        ex_choose = input("\nExport to CSV? (Y/N)\n").lower()
        if ex_choose == "y":
            df_z.to_csv(export_path, index=False)


    elif plot_choose == 4:

        # Interquartile Range (IQR) Method:

        # Approach: The IQR method is based on the quartiles of the data.
        # It calculates the interquartile range, which is the range between the first quartile (Q1)...
        # and the third quartile (Q3). Outliers are then identified as values ...
        # that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

        # Assumption: This method assumes that the data follows a roughly...
        # normal distribution and is less sensitive to extreme values.

        Q1 = np.percentile(df[col_choose], 25, method='midpoint')
        Q3 = np.percentile(df[col_choose], 75, method='midpoint')
        IQR = Q3 - Q1

        # Define upper and lower bounds
        upper = Q3 + iqr_threshold * IQR
        lower = Q1 - iqr_threshold * IQR

        # Arrays to indicate outliers rows
        upper_array = np.where(df[col_choose] >= upper)[0]
        lower_array = np.where(df[col_choose] <= lower)[0]

        # Create a new DataFrame df_q without outliers
        df_q = df.drop(index=np.concatenate([upper_array, lower_array])).copy()

        # Print Shapes
        print(lin)
        print(f"----> Old Shape: {shape}")
        print(f"----> New Shape (IQR): {df_q.shape}\n")
        print(f'{df_q.info()}\n')
        print(f'{df_q.describe()}\n\n')
        print("New Dataframe -> df_q")
        print(df_q.head())

        # CSV Export
        export_path = "../Data/df_q.csv"
        ex_choose = input("\nExport to CSV? (Y/N)\n").lower()
        if ex_choose == "y":
            df_q.to_csv(export_path, index=False)


    elif plot_choose == 5:
        print(lin)
        print("Key Differences:\n"

              "-> The Z-score method is more sensitive to extreme values and assumes a normal distribution.\n"
              "-> Z-score is a parametric method that assumes the data follows a normal distribution.\n"
              "-> Z-score is useful when the distribution of the data is known or approximately known,\n "
              "while IQR is more robust in the presence of non-normality.\n\n"

              "-> The IQR method is based on quartiles and is less affected by extreme values.\n"
              "-> IQR is a non-parametric method, meaning it makes fewer assumptions about the underlying distribution of the data.\n\n"

              "How to choose:\n"
              "-> In practice, the choice between these methods depends on the characteristics of your data and the underlying assumptions you are willing to make.\n"
              "-> It's often a good idea to compare results from both methods and choose the one that is more appropriate for your specific dataset.")

    else:
        ploting = False