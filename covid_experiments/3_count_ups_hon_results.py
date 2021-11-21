import pandas as pd


months = ['Jan','Feb','Mar','Apl','May','Jun','Jul','Aug']
tweet_counts = []
h_count = []
o_count = []
n_count = []
for m in range(1,9):
    cov_df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\predictions\predictions"+str(m)+"_"+str(m+1)+".csv")
    print(len(cov_df))
    # print(len(cov_df[cov_df['pred_label']=='normal']))
    tweet_counts.append(len(cov_df))
    n_count.append(len(cov_df[cov_df['pred_label']=='normal']))
    h_count.append(len(cov_df[cov_df['pred_label'] == 'hatespeech']))
    o_count.append(len(cov_df[cov_df['pred_label'] == 'offensive']))


count_df = pd.DataFrame()
count_df['months'] = months
count_df['tweet_counts'] = tweet_counts
count_df['h_count'] = h_count
count_df['o_count'] = o_count
count_df['n_count'] = n_count

count_df.to_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\\monthly_counts.csv")