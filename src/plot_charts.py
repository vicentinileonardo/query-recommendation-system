import matplotlib.pyplot as plt
import numpy as np

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
            n_queries.append(int(line.split('N_QUERIES = ')[1].split(',')[0]))
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

    # plot the chart of mae vs n_queries and n_users
    #plt.plot(n_queries, mae, label='MAE')
    plt.plot(n_queries, mae, label='MAE')
    plt.plot(n_queries, rmse, label='RMSE')
    plt.plot(n_queries, mape, label='MAPE')
    # add light grey hlines for every unit
    for i in range(2, 36, 2):
        plt.axhline(y=i, color='lightgrey', linestyle='-')

    plt.xlabel('n_queries')
    plt.ylabel('Values')
    # bold title
    plt.title('Compact User-User Collaborative Filtering' + '\nPerformance metrics vs n_queries')



    plt.figtext(0.90, 0.025, 'n_users = 2500 (fixed)', wrap=True, horizontalalignment='center', fontsize=10)

    #expand the plot
    plt.subplots_adjust(bottom=0.15)

    plt.ylim(bottom=0)
    # float values on the y axis, with 1 decimal
    plt.yticks(np.arange(0, 36, step=2), ['{:.1f}'.format(x) for x in np.arange(0, 36, step=2)])
    #plt.yticks(np.arange(0.0, 10.0, 1.0))
    plt.xticks(np.arange(0, 101, 10))


    # set dpi
    plt.rcParams['figure.dpi'] = 300


    plt.legend()

    # save the chart
    plt.savefig('../data/compact_user_user/compact_user_user_queries.png', dpi=300)

    plt.show()



if __name__ == "__main__":
    plot_charts('../data/compact_user_user/performance_queries.txt')
