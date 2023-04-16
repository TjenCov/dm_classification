import pandas as pd
from preprocessing import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


def scatter(data):
    pd.plotting.scatter_matrix(data, alpha=0.2, figsize=(15,15))
    plt.show()


def eda(data):
    """
    Simple eda function to find missing values etc
    :param data: general dataset
    :return:
    """
    total_customers = data.shape[0]
    for column in list(data.columns):
        print(f"{column}:\n")
        print(data[column].describe())
        customers_no = data[column].isna().sum()
        print(f"Percent of customers without a {column}: {customers_no/total_customers}")
        print(f"value counts:\n{data[column].value_counts()}")
        print("\n")


if __name__ == "__main__":
    data = load_dataset('existing-customers.csv')
    #scatter(data)
    # eda_age(data)
    # eda_education(data)
    # eda_workclass(data)
    eda(data)