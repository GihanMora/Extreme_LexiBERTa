import pandas as pd
import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix

def compute_metrics(pred, ground_labels):
    labels_all = ground_labels
    preds_all = list(pred)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all)
    acc = accuracy_score(labels_all, preds_all)
    confusion_mat = confusion_matrix(labels_all, preds_all)

    out_dict = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusiton_mat': confusion_mat
    }
    for k in out_dict.keys():
        print(k)
        print(out_dict[k])




df = pd.read_csv(r"E:\Projects\DSI_Gihan\Evaluations\predictions\hateXplain_test.csv")
print(df.columns)
l_dict = {'Religion/creed':1, 'Race/ethnicity':2, 'Gender':3,'Sexual Orientation':4,'Physical/disability':5}
actual = []
predicted = []
for i,row in df.iterrows():
    actual.append(row['label'])
    predicted.append(l_dict[row['predictions']])

# print(actual)
# print(predicted)

compute_metrics(predicted,actual)