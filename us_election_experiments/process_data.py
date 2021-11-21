import pandas as pd

for i in range(50):
    df = pd.read_csv(r"E:\Projects\DSI Gihan Prev\Datasets\us_election\hashtag_joebiden.csv", lineterminator='\n', nrows=20000, skiprows=range(1, 20000*i))
    print(df.columns)
    print(len(df))
    df['text'] = df['tweet']
    df['label'] = [0]*len(df['tweet'])

    df.to_csv(r"E:\Projects\DSI Gihan Prev\Datasets\us_election\\processed_biden"+str(i)+".csv")