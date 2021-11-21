import pandas as pd

from Finetuning_language_models.predict_using_finetuned_model_hon import predict_hon
for i in range(39,50):
    try:
        out_df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\Datasets\us_election\processed_biden"+str(i)+".csv")

        predictions = predict_hon(r"E:\Projects\DSI Gihan Prev\Datasets\us_election\processed_biden"+str(i)+".csv")

        out_df['pred_id'] = predictions['label_ids']
        out_df['pred_label'] = predictions['labels']

        out_df.to_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\hon_preds\\election_hon_biden"+str(i)+".csv")
    except PermissionError:
        out_df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\Datasets\us_election\processed_biden" + str(i) + ".csv")

        predictions = predict_hon(r"E:\Projects\DSI Gihan Prev\Datasets\us_election\processed_biden" + str(i) + ".csv")

        out_df['pred_id'] = predictions['label_ids']
        out_df['pred_label'] = predictions['labels']

        out_df.to_csv(
            r"E:\Projects\DSI Gihan Prev\us_election_experiments\hon_preds\\election_hon_biden" + str(i) + ".csv")

