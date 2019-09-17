import itertools
import pandas as pd

t = [1, 2, 3, 4, 5, 6]
c = list(itertools.combinations(t, 3))
for row in c:
    print(row)

df = pd.read_csv("chess.dat", sep="\s+", header=None)
numberOfItems = df.max().max()
items = list(range(1, numberOfItems + 1))
rules = list(itertools.combinations(items, 2))
