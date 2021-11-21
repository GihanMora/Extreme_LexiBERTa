import os
import pandas as pd
import ast



so = []
g = []
rel = []
ra = []
phy = []
months = ['January','February','March','April','May','June','July','August']

for fn in range(1,9):
    df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds\predictions"+str(fn)+"_"+str(fn+1)+"_processed.csv")


    res = {'Sexual Orientation': 0, 'Gender': 0, 'Religion/creed': 0, 'Race/ethnicity': 0, 'Physical/disability': 0}


    print(df.columns)
    for i,row in df.iterrows():
        x = ast.literal_eval(row['profiles'])

        # print(row['profiles'])
        st_pr = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
        val = list(st_pr.keys())
        for kk in val:
            res[kk] = res[kk]+x[kk]


    so.append(res['Sexual Orientation'])
    g.append(res['Gender'])
    rel.append(res['Religion/creed'])
    ra.append(res['Race/ethnicity'])
    phy.append(res['Physical/disability'])

res_d = pd.DataFrame()


res_d['month'] = months
res_d['Sexual Orientation'] = so
res_d['Gender'] = g
res_d['Religion/creed'] = rel
res_d['Race/ethnicity'] = ra
res_d['Physical/disability'] = phy




res_d.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\\theme_strengths.csv")




