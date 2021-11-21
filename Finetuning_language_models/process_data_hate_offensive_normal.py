import pandas as pd
import ast
train = pd.read_csv(r"E:\Projects\DSI Gihan Prev\Datasets\HateXplain Dataset\validation_processed.csv")
print(train.columns)
print(train['class'])
print(train['post_tokens'])
print(train.head())
texts = []
labels = []

for i,row in train.iterrows():

    txt = ' '.join(ast.literal_eval(row['post_tokens']))
    texts.append(txt)
    labels.append(row['label'])

out_df = pd.DataFrame()

out_df['text'] = texts
out_df['label'] = labels
out_df['label'] = out_df['label'].astype(int)

#
#
out_df.to_csv(r"E:\Projects\DSI Gihan Prev\Datasets\HateXplain Dataset\validation_hon.csv")