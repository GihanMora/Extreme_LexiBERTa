import numpy
from datetime import datetime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from Core.scoring import calculate_scores


def get_nearest_neighbours(embeding,df):
    t1 = datetime.now()
    tuples = []

    for i, row_e in df.iterrows():
        dis = cosine_similarity([row_e['embedding']], embeding)
        tuples.append([row_e['token'], row_e['label'], dis, row_e['embedding']])

    s_tup = sorted(tuples, key=lambda x: x[2])  # sort tuples based on the cosine distance
    neaarest_neighbs_words = []
    neaarest_neighbs_embs = []
    neaarest_neighbs_labels = []
    for i, m in enumerate(s_tup[::-1]):
        # print(m)
        if (i < 50):  # getting the nearest 100 neighbours
            neaarest_neighbs_words.append(m[0])
            neaarest_neighbs_embs.append(m[3])
            neaarest_neighbs_labels.append(m[1])
    n_score_dict = calculate_scores(neaarest_neighbs_words, neaarest_neighbs_labels)
    # neaarest_neighbs_words.append('sentence')
    # neaarest_neighbs_embs.append(numpy.array(embeding[0]))
    # neaarest_neighbs_labels.append('input')
    print(Counter(neaarest_neighbs_labels))
    t2 = datetime.now()
    diff = t2 - t1
    print('time nn and score', diff)

    return [n_score_dict,{'words':neaarest_neighbs_words,'embs':neaarest_neighbs_embs,'labels':neaarest_neighbs_labels}]