import itertools
import pandas as pd
import numpy as np
from typing import Tuple


def get_combinations(_items, _k: int, parent=[]):
    _test = []
    if _k == 0:
        return parent
    items_copy: list = _items.copy()
    for x in _items:
        items_copy.remove(x)
        next_parent = parent.copy()
        next_parent.append(x)
        supp: Tuple[int, float] = compute_support(tuple(next_parent), df)
        if supp[1] >= min_supp:
            combs = get_combinations(items_copy, _k - 1, next_parent)
            if combs:
                if type(combs[0]) is list or type(combs[0]) is tuple:
                    for comb in combs:
                        _test.append(tuple(comb))
                else:
                    _test.append(combs)
    return _test


def compute_support(_itemset: tuple, _df) -> Tuple[int, float]:
    filtered_itemset = _df
    for number in _itemset:
        filtered_itemset = filtered_itemset.loc[_df[number] == 1]
    itemset_count = len(filtered_itemset.index)

    return itemset_count, itemset_count / rows_count


def transform_table(_df):
    _numberOfItems = _df.max().max()
    columns = range(1, _numberOfItems + 1)
    data = np.zeros(shape=(len(_df.index), _numberOfItems), dtype='int')
    row_index = 0
    for row in _df.values:
        for number in row:
            data[row_index, number - 1] = 1
        row_index += 1
    return pd.DataFrame(data=data, columns=columns)


if __name__ == "__main__":
    t = [1, 2, 3, 4, 5, 6]
    c0 = list(itertools.combinations(t, 3))
    for row in c0:
        print(row)

    min_supp: float = 0.95  # support
    min_conf: float = 0.5  # confidence
    df = pd.read_csv("chess.dat", sep="\s+", header=None)
    number_of_items = df.max().max()
    rows_count = len(df.index)
    items = list(range(1, number_of_items + 1))
    testItems = list(["B", "M", "F", "Y", "C"])
    freq_itemsets = []
    df = transform_table(df)

    # test = get_combinations(testItems.copy(), 3)

    k = 1
    itemsets = [(1, 1)]
    while itemsets:
        itemsets = get_combinations(items, k)
        freq_dic = {}
        for itemset in itemsets:
            supp: float = compute_support(itemset, df)
            freq_dic[tuple(itemset)] = supp
            print("Itemset:" + str(itemset) + " Sup:" + f"{supp[1]:.2f}")
        print()
        freq_itemsets.append(freq_dic)
        k += 1

    print("Freq. itemsets: " + str(sum([i.__len__() for i in freq_itemsets])))
    print()

    rules = []
    count = 0
    k = 1
    for dic in freq_itemsets[1:]:
        rules_dic = {}
        for itemset, supp in dic.items():
            for item in itemset:
                itemset_copy = list(itemset)
                itemset_copy.remove(item)
                conf = supp[1] / freq_itemsets[k-1][tuple(itemset_copy)][1]
                if conf >= min_conf:
                    print(str(itemset_copy) + "->" + str(item) + " Conf:" + f"{conf:.2f}")
                    rules_dic[tuple(itemset_copy)] = item
                    count += 1
        rules.append(rules_dic)
        k += 1
        print()

    # print("Rules: " + str(sum([i.__len__() for i in rules])))
    print("Rules: " + str(count))
