import pandas as pd
import ast
train = pd.read_csv(r"E:\Projects\DSI_Gihan\Datasets\HateXplain Dataset\validation_processed.csv")
print(train.columns)
print(train['class'])
print(train['post_tokens'])

texts = []
labels_multi = []
labels = []

for i,row in train.iterrows():
    class_labels = ast.literal_eval(row['class'])
    if('Miscellaneous' in class_labels):
        class_labels.remove('Miscellaneous')
    if(len(class_labels)):
        for each_l in class_labels:
            txt = ' '.join(ast.literal_eval(row['post_tokens']))
            texts.append(txt)
            labels_multi.append(class_labels)
            labels.append(each_l)
out_df = pd.DataFrame()

out_df['text'] = texts
out_df['labels'] = labels
out_df['labels_multi'] = labels_multi


out_df.to_csv(r"E:\Projects\DSI_Gihan\Datasets\HateXplain Dataset\validation_fixed.csv")