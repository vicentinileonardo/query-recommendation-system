import matplotlib.pyplot as plt
import numpy as np
from mplfinance.original_flavor import candlestick2_ohlc
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle

def plot_charts(path):
    # read the file
    with open(path, 'r') as f:
        lines = f.readlines()

    # parse the file
    n_queries = []
    n_users = []
    time_taken = []
    mae = []
    rmse = []
    mape = []
    mre = []
    for line in lines:
        if line.startswith('Configuration:'):
            #n_queries.append(int(line.split('N_QUERIES = ')[1].split(',')[0]))
            n_users.append(int(line.split('N_USERS = ')[1].split('  ')[0]))
        elif line.startswith('Time taken:'):
            time_taken.append(float(line.split('Time taken: ')[1].split('   ')[0]))
        elif line.startswith('MAE:'):
            mae.append(float(line.split('MAE: ')[1].split('   ')[0]))
        elif line.startswith('RMSE:'):
            rmse.append(float(line.split('RMSE: ')[1].split('   ')[0]))
        elif line.startswith('MAPE:'):
            mape.append(float(line.split('MAPE: ')[1].split('   ')[0]))
        elif line.startswith('MRE:'):
            mre.append(float(line.split('MRE: ')[1].split('   ')[0]))

    # keep firt 6 values
    n_users = n_users[:5]
    mae = mae[:5]
    rmse = rmse[:5]
    mape = mape[:5]


    # plot the chart of mae vs n_queries and n_users
    #plt.plot(n_queries, mae, label='MAE')
    plt.plot(n_users, mae, label='MAE')
    plt.plot(n_users, rmse, label='RMSE')
    plt.plot(n_users, mape, label='MAPE')
    # add light grey hlines for every unit
    for i in range(2, 18, 2):
        plt.axhline(y=i, color='lightgrey', linestyle='-')

    plt.xlabel('n_users')
    plt.ylabel('Values')
    # bold title
    plt.title('Expanded Item-Item Collaborative Filtering' + '\nPerformance metrics vs n_users')



    plt.figtext(0.90, 0.025, 'n_items = 7669 (fixed)', wrap=True, horizontalalignment='center', fontsize=10)

    #expand the plot
    plt.subplots_adjust(bottom=0.15)

    plt.ylim(bottom=0)
    # float values on the y axis, with 1 decimal
    plt.yticks(np.arange(0, 18, step=2), ['{:.1f}'.format(x) for x in np.arange(0, 18, step=2)])
    #plt.yticks(np.arange(0.0, 10.0, 1.0))
    plt.xticks(np.arange(0, 101, 10))


    # set dpi
    plt.rcParams['figure.dpi'] = 300


    plt.legend()

    # save the chart
    plt.savefig('../data/movies_item_item_cf/expanded_item_item_users.png', dpi=300)

    plt.show()

def plot_charts_topk(path):
    # read the file
    with open(path, 'r') as f:
        lines = f.readlines()

    # parse the file
    TOP_K = []
    compact_mean = []
    compact_min = []
    compact_max = []
    hybrid_mean = [0.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    hybrid_min = [0 for x in range(1, 15)]
    hybrid_max = [1 for x in range(1, 15)]

    for line in lines:
        if line.startswith('TOP_K: '):
            TOP_K.append(int(line.split('TOP_K: ')[1].split('  ')[0]))
        elif line.startswith('Jaccard similarity compact: '):
            compact_mean.append(float(line.split('Jaccard similarity compact: ')[1].split('   ')[0]))
        elif line.startswith('Lowest jaccard similarity compact: '):
            compact_min.append(float(line.split('Lowest jaccard similarity compact: ')[1].split('   ')[0]))
        elif line.startswith('Highest jaccard similarity compact: '):
            compact_max.append(float(line.split('Highest jaccard similarity compact: ')[1].split('   ')[0]))
        elif line.startswith('Jaccard similarity hybrid: '):
            hybrid_mean.append(float(line.split('Jaccard similarity hybrid: ')[1].split('   ')[0]))
        elif line.startswith('Lowest jaccard similarity hybrid: '):
            hybrid_min.append(float(line.split('Lowest jaccard similarity hybrid: ')[1].split('   ')[0]))
        elif line.startswith('Highest jaccard similarity hybrid: '):
            hybrid_max.append(float(line.split('Highest jaccard similarity hybrid: ')[1].split('   ')[0]))

    print(TOP_K)
    print(compact_mean)
    print(compact_min)
    print(compact_max)
    print(hybrid_mean)
    print(hybrid_min)
    print(hybrid_max)

    # plot candlestick chart of jaccard similarity, using mean, min and max
    fig, ax = plt.subplots()
    ax.set_title('Jaccard similarity between items')
    ax.set_xlabel('TOP_K')
    ax.set_ylabel('Jaccard similarity')
    ax.set_ylim(-1, 2)
    ax.set_xticks(TOP_K)
    ax.set_xticklabels(TOP_K)
    ax.grid(True)

    # plot the candlestick chart

    candlestick2_ohlc(ax, compact_min, compact_max, compact_mean, compact_mean, width=0.6, colorup='g', colordown='r', alpha=0.75)
    candlestick2_ohlc(ax, hybrid_min, hybrid_max, hybrid_mean, hybrid_mean, width=0.6, colorup='b', colordown='b', alpha=0.75)





    # set dpi
    plt.rcParams['figure.dpi'] = 300


    plt.legend()

    # save the chart
    plt.savefig('../data/jaccard.png', dpi=300)

    plt.show()

def alternative_plot_charts_topk(csv_path):
    # Import
    df = pd.read_csv(csv_path)

    # x: top k
    # y: jaccard similarity
    # hue: compact or hybrid

    # Draw
    plt.figure(figsize=(12, 9), dpi=72)
    mpg_boxplot = sns.boxplot(x="top_k", y="jaccard_similarity_value", hue="algorithm_type", data=df, palette="pastel")
    mpg_stripplot = sns.stripplot(
        x="top_k",
        y="jaccard_similarity_value",
        hue="algorithm_type",
        data=df,
        color="black",
        dodge=True,  # this correctly aligns the dots with the individual box plots
        jitter=True,
        alpha=0.7,
    )

    # add some vertical lines to ease separation of groups
    for i in range(len(df["top_k"].unique()) - 1):
        plt.vlines(i + 0.5, -3, 45, linestyles="solid", colors="gray", alpha=0.2)

    # Decorate
    plt.ylim(-1, 2)

    plt.gca().set_yticks([0.0, 0.5, 1.0])
    plt.title("Dot-Box Plot of Top-K queries", fontsize=22)
    plt.xlabel("Top-K", fontsize=18)
    plt.ylabel("Jaccard similarity", fontsize=18)
    # since we have two Seaborn plots with "hue", we'll have two legends with the same information
    # remove one of the redundant legends:
    handles, labels = mpg_stripplot.get_legend_handles_labels()
    n_classes = df["algorithm_type"].nunique()
    plt.legend(handles[0:n_classes], labels[0:n_classes], title="algorithm_type")

    # save the chart
    plt.savefig('../data/PART_B/jaccard.png', dpi=300)

    plt.show()




if __name__ == "__main__":
    #plot_charts('../data/movies_item_item_cf/performance.txt')
    #plot_charts_topk('../data/jaccard_top_k.txt')
    alternative_plot_charts_topk('../data/PART_B/df_jaccard_similarity.csv')
