import pandas as pd
from transformers import AutoTokenizer, AutoModel
from Core.profile_builder import build_profile
import ast


model_path = r"E:\Projects\Extreme_LexiBERTa\Finetuning_language_models\model"
vocab_path = r"E:\Projects\Extreme_LexiBERTa\Vocabularies\vocabulary_v1.csv"



tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


df = pd.read_csv(vocab_path)
df = df.dropna()
df['embedding'] = [ast.literal_eval(i) for i in df['embedding'].values.tolist()]

# sentence = "always thought that nigger was a faggot"
sentence = "Muslims are so disgusting"
sentence = "muslims are very friendly"
pred = build_profile(sentence,1,df,tokenizer,model,keyword_extraction=True,modifier_detection=False)

print(pred)

# for m in range(1,50):
#     # if(m in [1,2,3,4,5,6,7,8,9,10,22,36]):
#     # cov_df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\predictions\predictions"+str(m)+"_"+str(m+1)+".csv")
#     cov_df = pd.read_csv(
#         r"E:\Projects\DSI Gihan Prev\us_election_experiments\hon_preds\election_hon_trump" + str(m)  + ".csv")
#
#     hate_sp = cov_df.loc[cov_df['pred_label'].isin(['hatespeech', 'offensive'])]
#     print(hate_sp.head())
#     preds = []
#     kwds = []
#
#     for i,row in  hate_sp.iterrows():
#         pred = build_profile(row['text'], 1, df, tokenizer, model, keyword_extraction=True, modifier_detection=False)
#         preds.append(pred[0])
#         kwds.append(pred[1])
#
#     hate_sp['profiles'] = preds
#     hate_sp['keywords'] = kwds
#
#
#     hate_sp.to_csv(r"E:\Projects\DSI Gihan Prev\us_election_experiments\processed_preds\\processed_trump" + str(m)  + ".csv")