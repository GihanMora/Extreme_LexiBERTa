import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from datasets import load_dataset
def predict_hon(df_path):
    emotions = load_dataset('csv', data_files={'test':[df_path]},cache_dir=r"E:\Projects\DSI Gihan Prev\Datasets\us_election")


    model_name = "E:\Projects\DSI Gihan Prev\Finetuning_language_models\hon_model"


    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)



    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])


    trainer = Trainer(model=model)
    trainer.model = model.cuda()
    preds_output = trainer.predict(emotions_encoded["test"])

    y_preds = np.argmax(preds_output.predictions, axis=1)

    print(y_preds)

    label_dict = {
        'hatespeech':0,
        'normal':1,
        'offensive':2,
        }

    labels = y_preds
    predictions = []
    for lbl in labels:
        keys = [k for k, v in label_dict.items() if v == lbl]
        print(keys)
        predictions.append(keys[0])
    # print(y.label_ids)

    return {'label_ids':labels.tolist(),'labels':predictions}