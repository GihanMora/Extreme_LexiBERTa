import ast
import nltk
import pandas as pd

f = open(r"E:\Projects\DSI_Gihan\propaganda_detection\2021_task_6_training_set_task2.txt","r",encoding="utf-8")

text = f.read()
text = ast.literal_eval(text)
print(text)

print(len(text))




dict_list = []
for i in text:
    print(i['id'])
    # print(i['text'])
    sentences =  nltk.sent_tokenize(i['text'])
    print(sentences)
    for k in i['labels']:
        print(k)
        curnt_dic = k
        for s in sentences:
            if(k['text_fragment'] in s):
                curnt_dic['sentence'] = s
        curnt_dic['id'] = i['id']
        curnt_dic['text'] = i['text']
        dict_list.append(curnt_dic)



    # break

print(dict_list)
df = pd.DataFrame(dict_list)

df.to_csv("E:\Projects\DSI_Gihan\propaganda_detection\data_processed_dev.csv")