import pandas as pd


df = pd.read_csv(r"E:\Projects\DSI_Gihan\propaganda_detection\data_processed_test.csv")

print(df.columns)

print(df['technique'].unique())

texts = []
labels = []


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



for i,row in df.iterrows():
    try:
        texts.append(row['sentence'].strip())
        labels.append(label_dict[row['technique']])
    except Exception:
        continue




out_df = pd.DataFrame()
out_df['text'] = texts
out_df['label'] = labels

out_df.to_csv(r"E:\Projects\DSI_Gihan\propaganda_detection\ready_for_test.csv")
