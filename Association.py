import pandas as pd
from collections import defaultdict
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def candidate_generation(baskets, k):
    freqSets = defaultdict(int)

    for basket in baskets:
        u = getUnion(basket, 1)
        print(u)


def getUnion(itemSet, length):
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

def find_rules(data):
    c = candidate_generation(baskets, 1)
    return





if __name__ == '__main__':
    dataset = pd.read_csv('data/dataset.csv')
    baskets = dataset.groupby("user_id").product_id.apply(list).tolist()
    a = TransactionEncoder()
    a_data = a.fit(baskets).transform(baskets)
    df = pd.DataFrame(a_data, columns=a.columns_)
    df = df.replace(False, 0)
    df = df.replace(True,1)
    df = apriori(df, min_support=0.01, verbose=1)
    df_ar = association_rules(df, metric="confidence", min_threshold=0.01)
    df_ar