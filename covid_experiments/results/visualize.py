import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_pie():
    data = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\theme_counts.csv")
    x = data["theme"]
    y = data["counts"]



    #define data
    data = y
    labels = x

    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]

    #create pie chart
    g = plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    plt.title('Hate-speech Themes for Covid-19 Vaccination Tweets')
    plt.legend(loc = "lower right")
    plt.show()

def draw_histo():
    data = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\monthly_counts.csv")
    x = data["months"]
    y1 = data["tweet_counts"]
    y2 = data["h_count"]
    y3 = data["n_count"]
    y4 = data["o_count"]

    labels = x
    men_means = y1
    women_means = y2

    x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    width = 0.35
    colors = sns.color_palette('pastel')[0:5]

    fig, ax = plt.subplots()
    # rects2 = ax.bar(x, y4, width, label='Offensive')
    rects1 = ax.bar(x - width/2 , y4, width, label='Offensive')
    rects3 = ax.bar(x + width/2 , y2, width, label='Hatespeech')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('#tweets')
    ax.set_title('Classes of Tweets per month in 2021')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()



    #
    #
    # # define data
    # data = y
    # labels = x
    #
    # # define Seaborn color palette to use
    # colors = sns.color_palette('pastel')[0:5]
    #
    # # create pie chart
    # g = plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
    # plt.title('Hate-speech Themes for Covid-19 Vaccination Tweets')
    # plt.legend(loc="lower right")
    # plt.show()


def draw_strengths():
    data = pd.read_csv(r"E:\Projects\DSI Gihan Prev\covid_experiments\results\theme_strengths.csv")
    x = data["month"]
    y1 = data['Sexual Orientation']
    y2 = data['Gender']
    y3 = data['Religion/creed']
    y4 = data['Race/ethnicity']
    y5 = data['Physical/disability']

    labels = x

    x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    width = 0.2
    colors = sns.color_palette('pastel')[0:5]

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - 0.4 ,y1,width=width,  label='Sexual Orientation', align='center')
    rects2 = ax.bar(x - 0.2,y2,width=width,  label='Gender', align='center')
    rects3 = ax.bar(x , y3,width=width, label='Religion/creed', align='center')
    rects4 = ax.bar(x + 0.2,y4,width=width,  label='Race/ethnicity', align='center')
    rects5 = ax.bar(x + 0.4,y5,width=width,  label='Physical/disability', align='center')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Theme significance')
    ax.set_title('Themes of hatespeech per month in 2021')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # ax.bar_label(rects1, padding=10)
    # ax.bar_label(rects2, padding=10)
    # ax.bar_label(rects3, padding=10)
    # ax.bar_label(rects4, padding=10)
    # ax.bar_label(rects5, padding=10)

    fig.tight_layout()
    plt.figure(figsize=(20,8))



    plt.show()



    #
    #
    # # define data
    # data = y
    # labels = x
    #
    # # define Seaborn color palette to use
    # colors = sns.color_palette('pastel')[0:5]
    #
    # # create pie chart
    # g = plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
    # plt.title('Hate-speech Themes for Covid-19 Vaccination Tweets')
    # plt.legend(loc="lower right")
    # plt.show()

# draw_histo()
draw_strengths()