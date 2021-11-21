import pandas as pd
from transformers import AutoTokenizer, AutoModel
from Core.profile_builder import build_profile
import ast
from datetime import datetime

model_path = r"E:\Projects\DSI_Gihan\Finetuning_language_models\model"
vocab_path = r"E:\Projects\DSI_Gihan\Vocabularies\vocabulary_v1.csv"



tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


vocab_df = pd.read_csv(vocab_path)
vocab_df = vocab_df.dropna()
vocab_df['embedding'] = [ast.literal_eval(i) for i in vocab_df['embedding'].values.tolist()]



test_df = pd.read_csv(r"E:\Projects\DSI_Gihan\Datasets\HateXplain Dataset\ready_for_testing.csv")

# print(test_df.head())

predictions = []

for i,row in test_df.iterrows():
    # print(row['text'])
    sentence = row['text']
    t1 = datetime.now()
    pred = build_profile(sentence, 1, vocab_df, tokenizer, model, keyword_extraction=False, modifier_detection=False)
    print(pred)
    pred = pred[0]

    pred_out = {k: v for k, v in sorted(pred.items(), key=lambda x: x[1])}
    # print(pred)

    pred = list(pred_out.keys())[-1]
    print(pred)
    predictions.append(pred)

    t2 = datetime.now()
    diff = t2 - t1
    print('time to build profile ', diff)


out_dff = test_df
print(len(out_dff))
out_dff['predictions'] = predictions

out_dff.to_csv(r"E:\Projects\DSI_Gihan\Evaluations\predictions/hateXplain_test.csv")