import pandas as pd
import ast


df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\theme_kwds.csv")

for i,row in df.iterrows():
    print(row['theme'])
    kwd_lst = []
    kwds = ast.literal_eval(row['counts'])
    for k in kwds:
        if('http' not in k):
            kwd_lst.append(k)
    print(kwd_lst)