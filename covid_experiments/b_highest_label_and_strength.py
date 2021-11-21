import ast
import pandas as pd

df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds\preds_all_with_meta.csv")
# df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds_2021_all_meta.csv")

print(df.columns)
max_labels = []
max_values = []
for i, row in df.iterrows():

    x = ast.literal_eval(row['profiles'])
    sorted_dict = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    print(sorted_dict)
    max_label = list(sorted_dict.keys())[-1]
    max_value = sorted_dict[max_label]
    max_labels.append(max_label)
    max_values.append(max_value)


df['max_label'] = max_labels
df['max_value'] = max_values

df.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\processed_preds_2021_all_meta_maxes_all.csv")