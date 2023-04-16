import pandas as pd
from sklearn import preprocessing

COLUMNS_TO_IGNORE = ["capital-gain", "capital-loss"]
UNETHICAL_COLUMNS = ["race", "sex", "native-country"]
LABEL_ENCODED = ["occupation", "workclass", "relationship", "marital-status", "education"]
encoders = []
FILEPATH = "./data/"
ETHICAL_CONCERNS = True



def load_dataset(file_name, sep=','):
    """
    Loads data set in from csv
    :param file_name:
    :param sep:
    :return: dataframe
    """
    """
    Loads a single dataset stored in a csv to a dataframe
    :param file_name:
    :return: dataframe
    """
    file_path = f"{FILEPATH}{file_name}"
    data = pd.read_csv(file_path, sep=sep)
    return data

def filter_columns(data):
    """
    Filters out any columns that are indicated in the COLUMNS_TO_IGNORE list
    :param data:
    :return:
    """
    data.drop(columns=COLUMNS_TO_IGNORE, inplace=True)
    return data


def label_encode(data, data_type):
    """
    Label encodes the columns indicated in the LABEL_ENCODED list
    :param data:
    :param data_type: should be either 'train' or 'predict'
    :return:
    """
    labels = LABEL_ENCODED
    if not ETHICAL_CONCERNS:
        labels += UNETHICAL_COLUMNS

    for column in labels:
        encoders.append(preprocessing.LabelEncoder())
        data[column] = encoders[-1].fit_transform(data[column])

    if data_type == 'train':
        data["class"] = data["class"].replace(">50K", 1)
        data["class"] = data["class"].replace("<=50K", 0)


    return data

def fill_nans(data):
    """
    Fills the nans
    :param data:
    :return:
    """
    data["occupation"] = data["occupation"].fillna("other")
    data["workclass"] = data["workclass"].fillna("Private")
    if not ETHICAL_CONCERNS:
        data["native-country"] = data["native-country"].fillna("United-States")

    return data


def normalize(data):
    """
    Normalizes certain columns
    :param data:
    :return:
    """
    data["capital-gain"] = (data["capital-gain"] - data["capital-gain"].min()) / (data["capital-gain"].max() - data["capital-gain"].min())
    data["capital-loss"] = (data["capital-loss"] - data["capital-loss"].min()) / (data["capital-loss"].max() - data["capital-loss"].min())

    return data


def make_more_ethical(data):
    """
    Removes the sensitive columns indicated in the ETHICAL_COLUMNS list.
    :param data:
    :return:
    """
    data.drop(columns=UNETHICAL_COLUMNS, inplace=True)
    return data


def preprocess(data, data_type='train'):
    """
    Preprocessing pipeline
    :param data:
    :param data_type: should be either 'train' or 'predict'
    :return:
    """
    if ETHICAL_CONCERNS:
        data = make_more_ethical(data)
    data = fill_nans(data)
    data = normalize(data)
    data = label_encode(data, data_type)
    return data




if __name__ == "__main__":
    data = load_dataset('existing-customers.csv')
