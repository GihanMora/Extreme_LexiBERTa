import pandas as pd



# df_list = []
# for i in range(1,49):
#     # if(i > 42):continue
#     df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\processed_preds\processed_trump"+str(i)+".csv")
#     df_list.append(df)
#
# out_df = pd.concat(df_list)
#
#
# # out_df.to_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\hon_preds\election_hon_trump_all.csv")
# out_df.to_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\processed_preds\processed_trump_all.csv")
# # ddff = pd.read_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\hon_preds\election_hon_trump_all.csv")
# #
# # print(len(ddff[ddff.pred_label=='offensive']))


trump = pd.read_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\processed_preds\processed_trump_all.csv")
baiden = pd.read_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\processed_preds\processed_biden_all.csv")

trump['tag'] = ['trump']*len(trump)
baiden['tag'] = ['baiden']*len(baiden)
out_df = pd.concat([trump,baiden])

out_df.to_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\processed_preds\processed_trump_biden_all.csv")