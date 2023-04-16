import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessing import load_dataset, preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt


MODEL = "RF"    # RF/AB/


def split_data(data):
    """
    Splits data in train and test set
    :param data: data to be split
    :return:
    """
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


def train_adaboost(data):
    """
    Trains the AdaBoost classifier
    :param data: training data
    :return:
    """
    data.drop(columns=["RowID"], inplace=True)
    X_train, X_test, y_train, y_test = split_data(data)
    clf = AdaBoostClassifier(learning_rate=0.8, n_estimators=50, random_state=0)
    clf.fit(X_train, y_train)


    for k, feature in enumerate(clf.feature_importances_):
        print(f"{data.columns[k]}: {feature}")



    results = pd.DataFrame()
    results['p'] = clf.predict(X_test)
    results['class'] = y_test

    print(f"accuracy score: {accuracy_score(results['class'],results['p'])}")
    print(f"precision score: {precision_score(results['class'],results['p'])}")
    print(f"recall score: {recall_score(results['class'],results['p'])}")

    return clf, results


def train_randomForest(data):
    """
    Trains the random forst classifier
    :param data: training data
    :return:
    """
    data.drop(columns=["RowID"], inplace=True)
    X_train, X_test, y_train, y_test = split_data(data)
    clf = RandomForestClassifier( n_estimators=100, max_depth=16, max_features=5, random_state=0)
    clf.fit(X_train, y_train)


    for k, feature in enumerate(clf.feature_importances_):
        print(f"{data.columns[k]}: {feature}")


    results = pd.DataFrame()
    results['p'] = clf.predict(X_test)
    results['class'] = y_test

    print(f"accuracy score: {accuracy_score(results['class'], results['p'])}")
    print(f"precision score: {precision_score(results['class'], results['p'])}")
    print(f"recall score: {recall_score(results['class'],results['p'])}")


    return clf, results


def predict(model, data):
    """
    Predicts label of the data using the given model
    :param model: sklearn classifier model
    :param data: data to be classified
    :return:
    """
    ids = data["RowID"].copy()
    data.drop(columns=["RowID"], inplace=True)
    data['p'] = model.predict(data)
    data["RowID"] = ids
    return data


def calculate_revenue(data, train_results):
    """
    Estimates total revenue using results of both training and prediction
    :param data: final dataframe containing the predicted label
    :param train_results: results of training phase
    :return:
    """

    precision_wealthy = len(train_results[(train_results['class'] == 1) & (train_results['p'] == 1)]) / len(train_results[train_results['p'] == 1])
    precision_poor = len(train_results[(train_results['class'] == 0) & (train_results['p'] == 1)]) / len(train_results[train_results['p'] == 1])

    balance = -(data['p'].sum() * 10)     # subtract total mailing cost
    balance += data['p'].sum() * precision_wealthy * 0.1 * 980
    balance -= data['p'].sum() * precision_poor * 0.05 * -310

    print(f"Estimated balance: {balance}")


def save_predictions(data):
    """
    Filters out the potential customers with p = 1 and stores their ids in output.txt
    :param data: final dataframe
    :return:
    """
    ids = data[data['p'] == 1]["RowID"].copy()
    output = ""
    for id in ids:
        output += f"{id}\n"
    with open("output.txt", 'w') as f:
        f.write(output)


if __name__ == "__main__":
    train_data = load_dataset('existing-customers.csv')
    predict_data = load_dataset('potential-customers.csv')

    train_data = preprocess(train_data, 'train')
    predict_data = preprocess(predict_data, 'predict')

    model = None
    train_results = None
    if MODEL == 'AB':
        model, train_results = train_adaboost(train_data)
    elif MODEL == 'RF':
        model, train_results = train_randomForest(train_data)


    predict_data = predict(model, predict_data)
    calculate_revenue(predict_data, train_results)
    save_predictions(predict_data)

