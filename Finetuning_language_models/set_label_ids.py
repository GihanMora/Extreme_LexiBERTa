import pandas as pd

df = pd.read_csv(r"E:\Projects\DSI_Gihan\Datasets\HateXplain Dataset\test_fixed.csv")
l_dict = {'Religion/creed':1, 'Race/ethnicity':2, 'Gender':3,'Sexual Orientation':4,'Physical/disability':5}
print(df.columns)
print(df['labels'].unique())

out_df = pd.DataFrame()
txts = []
lbs = []
for i,row in df.iterrows():
    txts.append(row['text'])
    lbs.append(l_dict[row['labels']])



out_df['text'] = txts
out_df['label'] = lbs


out_df.to_csv(r"E:\Projects\DSI_Gihan\Datasets\HateXplain Dataset\ready_for_testing.csv")