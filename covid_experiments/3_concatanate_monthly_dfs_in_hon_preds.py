import pandas as pd



df_list = []
for i in range(1,8):
    # if(i in [10,22,36]):continue
    df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds\predictions"+str(i)+"_"+str(i+1)+"_processed.csv")
    df_list.append(df)

out_df = pd.concat(df_list)


out_df.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds\preds_all.csv")

ddff = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\processed_preds\preds_all.csv")

print(len(ddff[ddff.pred_label=='offensive']))