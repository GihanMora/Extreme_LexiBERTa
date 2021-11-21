from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from datasets import load_dataset
emotions = load_dataset('csv', data_files={'train': [r"E:\Projects\DSI_Gihan\propaganda_detection\ready_for_training.csv"],
                                          'validation': [r"E:\Projects\DSI_Gihan\propaganda_detection\ready_for_dev.csv"],
                                           'test':[r"E:\Projects\DSI_Gihan\propaganda_detection\ready_for_test.csv"]})


model_name = "E:\Projects\DSI_Gihan\propaganda_detection\propaganda_model"


model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)



emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])


trainer = Trainer(model=model)
trainer.model = model.cuda()
y = trainer.predict(emotions_encoded["test"])



label_dict = {
    'Black-and-white Fallacy/Dictatorship':0,
    'Slogans':1,
    'Name calling/Labeling':2,
    'Loaded Language':3,
    'Smears':4,
    'Causal Oversimplification':5,
    'Exaggeration/Minimisation':6,
    'Appeal to fear/prejudice':7,
    'Reductio ad hitlerum':8,
    'Repetition':9,
    'Glittering generalities (Virtue)':10,
    "Misrepresentation of Someone's Position (Straw Man)":11,
    'Doubt':12,
    'Obfuscation, Intentional vagueness, Confusion':13,
    'Whataboutism':14,
    'Flag-waving':15,
    'Thought-terminating cliché':16,
    'Presenting Irrelevant Data (Red Herring)':17,
    'Appeal to authority':18,
    'Bandwagon':19}

labels = y.label_ids

for lbl in labels:
    keys = [k for k, v in label_dict.items() if v == lbl]
    print(keys)
# print(y.label_ids)