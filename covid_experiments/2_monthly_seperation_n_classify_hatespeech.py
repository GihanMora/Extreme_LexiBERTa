import pandas as pd
import datetime
from dateutil import parser

from Finetuning_language_models.predict_using_finetuned_model_hon import predict_hon

df = pd.read_csv(r"E:\Projects\DSI_Gihan\Datasets\covid\covid_vaccine\covidvaccine.csv", parse_dates=True)

print(df.head())
print(df.columns)

print(len(df))


# df['date'] = pd.to_datetime(df['date'])
dates = []
for d in df['date']:
    try:
        dates.append(parser.parse(d))
    except:
        dates.append(datetime.datetime(2000, 1, 1, 0, 0, 0))
df['date'] = dates
for k in range(8,9):
    range_start = datetime.datetime(2021, k, 1, 0, 0, 0)
    range_end = datetime.datetime(2021, k+1, 1, 0, 0, 0)


    df = df[range_start <= df.date]
    df = df[df.date <= range_end]
    print(df.head())
    print(len(df))

    train = df

    texts = []
    labels = []
    #
    for i,row in train.iterrows():
    #
        txt = row['text']
        texts.append(txt)
        labels.append(0)
    #
    out_df = pd.DataFrame()
    #
    out_df['text'] = texts
    out_df['label'] = labels
    out_df['label'] = out_df['label'].astype(int)

    out_df.to_csv(r'E:\Projects\DSI Gihan Prev\covid_experiments\predictions\\temp_input'+str(k)+'_'+str(k+1)+'.csv')

    predictions = predict_hon(r'E:\Projects\DSI Gihan Prev\covid_experiments\predictions\\temp_input'+str(k)+'_'+str(k+1)+'.csv')

    out_df['pred_id'] = predictions['label_ids']
    out_df['pred_label'] = predictions['labels']

    out_df.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\predictions\\predictions"+str(k)+'_'+str(k+1)+".csv")