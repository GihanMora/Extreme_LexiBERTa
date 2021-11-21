import os
import pandas as pd
import ast
# root_path = r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds"
# dfs = os.listdir(root_path)
# dfs_list = []
# for csvf in dfs:
#     df = pd.read_csv(os.path.join(root_path,csvf))
#     print(df.head())
#     dfs_list.append(df)
#
#
# dddf = pd.concat(dfs_list, ignore_index=True)
#
# dddf.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds_2021_all.csv")

# df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds_2021_all.csv")
df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds\preds_all_with_meta.csv")
res = {'Sexual Orientation': 0, 'Gender': 0, 'Religion/creed': 0, 'Race/ethnicity': 0, 'Physical/disability': 0}
res_k = {'Sexual Orientation': [], 'Gender': [], 'Religion/creed': [], 'Race/ethnicity': [], 'Physical/disability': []}
print(df.columns)
for i,row in df.iterrows():
    x = ast.literal_eval(row['profiles'])
    y = ast.literal_eval(row['keywords'])
    # print(row['profiles'])
    st_pr = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    val = list(st_pr.keys())
    print(val[-1])
    res[val[-1]] = res[val[-1]]+1
    res_k[val[-1]] = res_k[val[-1]] + y

print(res)
print(res_k)
res_d = pd.DataFrame()
res_k_d = pd.DataFrame()

res_d['theme'] = list(res.keys())
res_d['counts'] = list(res.values())
res_d.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\\theme_counts_all.csv")


res_k_d['theme'] = list(res_k.keys())
res_k_d['counts'] = [str(i) for i in list(res_k.values())]
res_k_d.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\\theme_kwds_all.csv")


