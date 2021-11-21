import pandas as pd
import ast
train = pd.read_csv(r"E:\Projects\DSI Gihan Prev\Datasets\covid\covid_vaccine\covidvaccine.csv")
print(train.columns)
print(train.head())
texts = []
labels = []
#
for i,row in train.iterrows():
#
    txt = row['text']
    texts.append(txt)
    labels.append(-1)
#
out_df = pd.DataFrame()
#
out_df['text'] = texts
out_df['label'] = labels
out_df['label'] = out_df['label'].astype(int)
#
# #
# #
# out_df.to_csv(r"E:\Projects\DSI Gihan Prev\Datasets\HateXplain Dataset\validation_hon.csv")