import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#plot the vocabulary
import ast
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.decomposition import PCA #Grab PCA functions

# def visualize_embs(labels, embs, words):
#     # fourteen 2 colors
#     label_color_dict = {'input': 'black',
#                         'Religion/creed': 'red',
#                         'Race/ethnicity': 'green',
#                         'Gender': 'blue',
#                         'Sexual Orientation': 'purple',
#                         'Physical/disability': 'gray',
#                         }
#     # #vocabulary_visualization_8_colors
#     # label_color_dict = {'anticipation':'green','anger':'red','disgust':'orange',
#     #                     'fear':'brown','sadness':'black','joy':'yellow','surprise':'purple',
#     #                     'trust':'gray'}
#
#     # #vocabulary_visualization_2_colors
#     # label_color_dict = {'anticipation':'green','anger':'red','disgust':'red',
#     #                     'fear':'red','sadness':'red','joy':'green','surprise':'green',
#     #                     'trust':'green'}
#     X = embs
#     pca = PCA(n_components=2)
#     result = pca.fit_transform(X)
#
#     filtered_words = []
#     filtered_emb = []
#     filtered_label = []
#
#     #seperate
#
#     # for i, j in enumerate(result):
#     #     if (j[0] < 1 and labels[i] in ['positive', 'joy', 'anticipation', 'distraction', 'trust', 'surprise',
#     #                                    'joy_ecstasy', 'admire', 'acceptance', 'interest_vigilance',
#     #                                    'amazement_surprise', 'input']):
#     #         filtered_emb.append(j)
#     #         filtered_label.append(labels[i])
#     #
#     #         filtered_words.append(words[i])
#     #
#     #     if (j[0] > -1 and labels[i] in ['negative', 'sadness', 'disgust', 'anger', 'fear', 'disgust_loathing',
#     #                                     'boredom', 'input']):
#     #         filtered_emb.append(j)
#     #         filtered_label.append(labels[i])
#     #         filtered_words.append(words[i])
#
#
#     result = np.array([x for x in filtered_emb])
#     labels = filtered_label
#     words = filtered_words
#
#     cvec = [label_color_dict[label] for label in labels]
#     # print('filtered words')
#     # print(filtered_words)
#     # print('filtered labels')
#     # print(filtered_label)
#     # fig, ax = plt.subplots()
#     # ax.plot(result[:, 0], result[:, 1], 'o')
#     # ax.set_title('Tweets')
#     # plt.show()
#
#     # Create the scatter plot
#     plt.figure(figsize=(8, 8))
#     plt.scatter(result[:, 0], result[:, 1], c=cvec, edgecolor='', alpha=0.5)
#
#     # selected_names = ['joy']
#     # names = ['sadness','joy','surprise','trust','anticipation','anger','disgust','fear']
#     # some_labels = []
#     # for ll in ['joy','anticipation','distraction','trust','surprise','joy_ecstasy','admire','acceptance','interest_vigilance','amazement_surprise','sadness','disgust','anger','fear','disgust_loathing','boredom']:
#     #   some_labels.extend([i for i in labels if i==ll])
#     # names = labels
#     # random.shuffle(words)
#     names = words
#     selected_names = ['sentence']
#     # selected_names = words
#     # names = words
#     # Add the labels
#     # for name in names:
#     # #
#     # #     # Get the index of the name
#     #     i = names.index(name)
#     # #
#     #     # Add the text label
#     #     labelpad = 0.01   # Adjust this based on your dataset
#     #     plt.text(result[i,0]+labelpad, result[i,1]+labelpad, name, fontsize=8)
#     #     # plt.text(result[i,0]+labelpad, result[i,1]+labelpad, labels[i], fontsize=8)
#
#     #     # Mark the labeled observations with a star marker
#     #     if(name in selected_names):
#     #       plt.scatter(result[i,0], result[i,1],
#     #                   c=cvec[i], vmin=min(cvec), vmax=max(cvec),
#     #                   edgecolor='', marker='*', s=100)
#
#     # Add the labels vocabulary
#     # names = words
#     # selected_names = names
#
#     # for i in range(0,len(names),50):
#     # #
#     # #     # Get the index of the name
#     #     name = names[i]
#     # #
#     #     # Add the text label
#     #     labelpad = 0.01   # Adjust this based on your dataset
#     #     # plt.text(result[i,0]+labelpad, result[i,1]+labelpad, name, fontsize=8)
#     #     plt.text(result[i,0]+labelpad, result[i,1]+labelpad, labels[i], fontsize=8)
#
#     #     # Mark the labeled observations with a star marker
#     #     if(name in selected_names):
#     #       plt.scatter(result[i,0], result[i,1],
#     #                   c=cvec[i], vmin=min(cvec), vmax=max(cvec),
#     #                   edgecolor='', marker='*', s=100)
#
#     # Add the axis labels
#     plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
#     plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))
#     plt.title('Embedding Space')
#     # Done
#     plt.show()

def tsne_vocab_plot(vocab_df):
    from sklearn.manifold import TSNE
    import time
    print(vocab_df.columns)
    X = [ast.literal_eval(i) for i in vocab_df['embedding']]
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    vocab_df['tsne-2d-one'] = tsne_results[:, 0]
    vocab_df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", len(vocab_df['label'].unique())),
        # palette=sns.color_palette("hls",14),
        # palette=['green','green','green','green','green','red','green','red','red','red','red','red','green','green'],
        # palette=['#C0EB84','#148B0E','#52A22A','#3B780C','#A5E250','#fdc70c','#895207','#BF3804','#DF2902','#EF2101','#ed683c','#f3903f','#40760B','#295115'],
        data=vocab_df,
        legend="full",
        alpha=0.8
    )

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'] + .02, point['y'], str(point['val']))

    emos = vocab_df['label'].unique()
    print(emos)
    vl_df = pd.DataFrame()
    for ee in emos:
        # print(ee)
        l_df = vocab_df[vocab_df['label'] == ee][:3]
        # print(l_df.head())
        vl_df = pd.concat([vl_df, l_df], ignore_index=True)

    # vl_df = pd.concat(emo_df_list, axis=1,ignore_index=True)

    # print(vl_df)
    label_point(vl_df['tsne-2d-one'], vl_df['tsne-2d-two'], vl_df['label'], plt.gca())
    plt.savefig(r'E:\Projects\Extreame_LexiBERTa\Vocabularies\vocab_v1.png')

    plt.show()


vocab_df = pd.read_csv(r"E:\Projects\Extreame_LexiBERTa\Vocabularies\vocabulary_v1.csv")
print('emo',len(vocab_df))
print(vocab_df.head())


words = vocab_df['token'].tolist()
embeddings = [ast.literal_eval(i) for i in vocab_df['embedding'].tolist()]
eight_label = vocab_df['label'].tolist()

# visualize_embs(eight_label,embeddings,words)
tsne_vocab_plot(vocab_df)