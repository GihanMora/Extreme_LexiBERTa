import pandas as pd
import ast

import torch
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv(r"E:\Projects\Extreame_LexiBERTa\Datasets\HateXplain Dataset\train_processed.csv")
print(df.columns)
print(df['phrases'])
print(df['class'].unique())

phrases = []
labels = []
lexicon = {'Race/ethnicity':[],
           'Gender':[],
           'Religion/creed':[],
           'Sexual Orientation':[],
           'Physical/disability':[]
           }
for i,row in df.iterrows():
    class_labels = ast.literal_eval(row['class'])
    if('Miscellaneous' in class_labels):
        class_labels.remove('Miscellaneous')
    if(len(class_labels)):
        for each_l in class_labels:
            if(row['phrases']):
                lexicon[each_l].extend(ast.literal_eval(row['phrases']))
            phrases.append(row['phrases'])
            labels.append(each_l)
            # print(row['phrases'],each_l)
            # labels.append(each_l)
# out_df = pd.DataFrame()
print(lexicon)

for key in list(lexicon.keys()):
    print(key)
    print(lexicon[key])
    print(len(list(set(lexicon[key]))))
# out_df['text'] = texts
# out_df['labels'] = labels
# out_df['labels_multi'] = labels_multi


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

tokenizer = AutoTokenizer.from_pretrained("E:\Projects\Extreame_LexiBERTa\Finetuning_language_models\model")
model = AutoModel.from_pretrained(r"E:\Projects\Extreame_LexiBERTa\Finetuning_language_models\model")


tokens = []
embedding = []
label = []

for each_key in lexicon:
  words_list = list(set(lexicon[each_key]))

  for wd in words_list:
      print(wd)
      sentences = [wd]
      encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
      with torch.no_grad():
          model_output = model(**encoded_input)
      sentence_embeddings_raw = mean_pooling(model_output, encoded_input['attention_mask'])
      sentence_embeddings = sentence_embeddings_raw.tolist()[0]
      tk = wd
      emb = sentence_embeddings
      e_lbl = each_key

      tokens.append(tk)
      embedding.append(emb)
      label.append(e_lbl)



out_df = pd.DataFrame()
out_df['token'] = tokens
out_df['embedding'] = embedding
out_df['label'] = label


# existing_def = pd.read_csv()

out_df.to_csv(r"E:\Projects\Extreame_LexiBERTa\Vocabularies\vocabulary_v1.csv")