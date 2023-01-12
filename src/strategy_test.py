import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math

# set pandas to print the entire dataframe when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def import_data(matrix_type="partial"):
    if matrix_type == "partial":
        path = '../data/_utility_matrix.csv'
    elif matrix_type == "real_complete":
        path = '../data/_utility_matrix_complete.csv'
    else:
        print('Wrong matrix type!')
        return
    utility_matrix = pd.read_csv(path, index_col=0)
    return utility_matrix

def preprocess_data(utility_matrix, n_queries=100, n_users=100):
    utility_matrix = utility_matrix.copy()
    utility_matrix = utility_matrix.T
    utility_matrix = utility_matrix.iloc[:n_queries, :n_users]
    utility_matrix = utility_matrix.fillna(np.nan)

    # example dataset, from the course slides
    '''
    dummy_um = pd.DataFrame([[1, np.nan, 3, np.nan, np.nan, 5, np.nan, np.nan, 5, np.nan, 4, np.nan],
                           [np.nan, np.nan, 5, 4, np.nan, np.nan, 4, np.nan, np.nan, 2, 1, 3],
                           [2, 4, np.nan, 1, 2, np.nan, 3, np.nan, 4, 3, 5, np.nan],
                           [np.nan, 2, 4, np.nan, 5, np.nan, np.nan, 4, np.nan, np.nan, 2, np.nan],
                           [np.nan, np.nan, 4, 3, 4, 2, np.nan, np.nan, np.nan, np.nan, 2, 5],
                           [1, np.nan, 3, np.nan, 3, np.nan, np.nan, 2, np.nan, np.nan, 4, np.nan]])
    
    dummy_um.index = utility_matrix.index[:6]
    dummy_um.columns = utility_matrix.columns[:12]
    utility_matrix = dummy_um
    '''

    # maybe, order the rows, using row.count to count the most non nan values
    #utility_matrix = utility_matrix.reindex(utility_matrix.count(axis=1).sort_values(ascending=False).index)

    utility_matrix_before_pp = utility_matrix.copy()

    # Addressing issue #01 check if row mean == 0 (only 0s and NAs), if that is true,  change the 0.0s with something else
    for query in range(utility_matrix.shape[0]):
        if utility_matrix.iloc[query, :].mean() == 0:
            for user in range(utility_matrix.shape[1]):
                if utility_matrix.iloc[query, user] == 0:  # 0s, not NAs
                    utility_matrix.iloc[query, user] = round((1 * utility_matrix.iloc[:, user].mean()) % 10) # to fix, not use the modulo
                    #utility_matrix.iloc[query, user] = 5

    # Centered matrix: subtract the mean of each row (query) from the ratings
    centered_matrix = utility_matrix.copy()
    for row in range(utility_matrix.shape[0]):
        for col in range(utility_matrix.shape[1]):
            if utility_matrix.iloc[row, col] != np.nan:
                centered_matrix.iloc[row, col] = utility_matrix.iloc[row, col] - utility_matrix.iloc[row, :].mean()



    #Note: merge the above loops into one


    # filling NAs of the centered matrix
    # option 1: fill na with 0s
    #centered_matrix = centered_matrix.fillna(0)

    # option 2: fill na cells with the mean of their row (query)
    centered_matrix = centered_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

    partial_utility_matrix = utility_matrix.copy()

    return utility_matrix_before_pp, partial_utility_matrix, centered_matrix

def calculate_rating(utility_matrix, centered_matrix, counter, user=4, query=0, top_n=2):
    # cosine similarity between rows
    similarities = cosine_similarity(centered_matrix, centered_matrix)

    # selecting the column related to the similarities of the specific query against the others
    similarities = similarities[query, :]
    print('similarities:', similarities)

    # Sort the similarities in descending order
    sorted_similarities = np.argsort(similarities)[::-1]
    print('sorted_sim:', sorted_similarities)

    # Select the top N similar queries, excluding the query itself
    top_n_similarities = sorted_similarities[1:top_n+1]
    print('top similarities: ', top_n_similarities)

    rating = 0
    for i in top_n_similarities:
        # handle nan values, ref. slide 32 of RecSys slide deck, using the mean of the query
        if np.isnan(utility_matrix.iloc[i, user]):
            rating += similarities[i] * utility_matrix.iloc[i, :].mean()
        else:
            rating += similarities[i] * utility_matrix.iloc[i, user]
    # print('sum', similarities[top_n_similarities].sum())

    # handle division by zero, even after the preprocessing, this could happen only if the matrix is very small
    if similarities[top_n_similarities].sum() == 0:
        # print('inside exception')

        # option 1: return the mean of the query
        rating = utility_matrix.iloc[query, :].mean()

        # option 2: return the mean of the user
        # rating = utility_matrix.iloc[:, user].mean()

        # option 3
        # rating = 0

        counter["c_exception"] += 1
    else:  # normal behaviour
        rating /= similarities[top_n_similarities].sum()
        counter["c_normal"] += 1
    rating = round(rating)

    if rating > 100:
        rating = 100
    elif rating < 0:
        rating = 0

    return rating, counter

def collaborative_filtering(utility_matrix, centered_matrix, top_n=2):
    # loop all the cell of utility matrix with nan and calculate the rating
    complete_utility_matrix = utility_matrix.copy()
    print('Complete matrix shape:', complete_utility_matrix.shape)
    ratings_list = []
    counter = {"c_exception": 0, "c_normal": 0}
    for row in range(utility_matrix.shape[0]):
        for col in range(utility_matrix.shape[1]):
            if np.isnan(utility_matrix.iloc[row, col]):
                print('row:', row, 'col:', col)
                rating, counter_exception = calculate_rating(utility_matrix, centered_matrix, counter, user=col, query=row, top_n=top_n)
                complete_utility_matrix.iloc[row, col] = rating
                # append the rating also to a list
                ratings_list.append(rating)
        print('Row: ', row, 'Estimated percentage of completion: ', round((row / utility_matrix.shape[0]) * 100, 2), '%')
    print('Counter: ', counter)
    return complete_utility_matrix, ratings_list

def compute_difference(utility_matrix_complete, n_queries=100, n_users=100):

    real_utility_matrix_complete = import_data("real_complete")
    real_utility_matrix_complete = real_utility_matrix_complete.T
    real_utility_matrix_complete = real_utility_matrix_complete.iloc[:n_queries, :n_users]

    difference_utility_matrix = (real_utility_matrix_complete - utility_matrix_complete)
    return difference_utility_matrix

# testing performances
def calculate_mae(real_utility_matrix_complete, utility_matrix_complete):
    # calculate the MAE
    mae = 0
    for row in range(utility_matrix_complete.shape[0]):
        for col in range(utility_matrix_complete.shape[1]):
            if np.isnan(utility_matrix_complete.iloc[row, col]):
                mae += abs(real_utility_matrix_complete.iloc[row, col] - utility_matrix_complete.iloc[row, col])
    mae /= utility_matrix_complete.isna().sum().sum()
    return mae

    # alternative, to be tested
    # mae = mean_absolute_error(real_utility_matrix_complete, complete_utility_matrix)

def calculate_rmse(real_utility_matrix_complete, utility_matrix_complete):
    # calculate the RMSE
    rmse = 0
    for row in range(utility_matrix_complete.shape[0]):
        for col in range(utility_matrix_complete.shape[1]):
            if np.isnan(utility_matrix_complete.iloc[row, col]):
                rmse += (real_utility_matrix_complete.iloc[row, col] - utility_matrix_complete.iloc[row, col]) ** 2
    rmse /= utility_matrix_complete.isna().sum().sum()
    rmse = math.sqrt(rmse)
    return rmse

    # alternative, to be tested
    # rmse = sqrt(mean_squared_error(real_utility_matrix_complete, complete_utility_matrix))

def calculate_mape(real_utility_matrix_complete, utility_matrix_complete):
    # calculate the MAPE
    mape = 0
    for row in range(utility_matrix_complete.shape[0]):
        for col in range(utility_matrix_complete.shape[1]):
            if np.isnan(utility_matrix_complete.iloc[row, col]):
                mape += abs(real_utility_matrix_complete.iloc[row, col] - utility_matrix_complete.iloc[row, col]) / real_utility_matrix_complete.iloc[row, col]
    mape /= utility_matrix_complete.isna().sum().sum()
    mape *= 100
    return mape

    # alternative, to be tested
    # mape = np.mean(np.abs((real_utility_matrix_complete - complete_utility_matrix) / real_utility_matrix_complete)) * 100

def calculate_mre(real_utility_matrix_complete, utility_matrix_complete):
    # calculate the MRE
    mre = 0
    for row in range(utility_matrix_complete.shape[0]):
        for col in range(utility_matrix_complete.shape[1]):
            if np.isnan(utility_matrix_complete.iloc[row, col]):
                mre += abs(real_utility_matrix_complete.iloc[row, col] - utility_matrix_complete.iloc[row, col]) / real_utility_matrix_complete.iloc[row, col]
    mre /= utility_matrix_complete.isna().sum().sum()
    return mre

    # alternative, to be tested
    # mre = np.mean(np.abs((real_utility_matrix_complete - complete_utility_matrix) / real_utility_matrix_complete))

def print_data_to_html(utility_matrix_before_pp, partial_utility_matrix, centered_matrix, utility_matrix_complete, difference_utility_matrix):
    with open('../data/viz.html', 'w') as f:
        f.write("<html><head><title>Visualization</title></head><body>")
        f.write("<h1>Utility Matrix Before Preprocessing</h1>")
        f.write(utility_matrix_before_pp.to_html())
        f.write("<h1>Partial Utility Matrix</h1>")
        f.write(partial_utility_matrix.to_html())
        f.write("<h1>Centered Matrix</h1>")
        f.write(centered_matrix.to_html())
        f.write("<h1>Computed Complete Utility Matrix</h1>")
        f.write(utility_matrix_complete.to_html())
        f.write("<h1>Difference Utility Matrix</h1>")
        f.write(difference_utility_matrix.to_html())

# function that plot the heatmap of the utility matrix
def plot_heatmap(utility_matrix, n_rows, n_columns, title):
    utility_matrix = utility_matrix.copy()
    utility_matrix = utility_matrix.iloc[:n_rows, :n_columns]
    plt.figure(figsize=(12,10), dpi=75)
    sns.heatmap(utility_matrix, cmap='coolwarm', center=0, annot=True)

    # Decorate
    plt.title(title, fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


    # save the plot
    plt.savefig('../data/heatmap.png')
    plt.show()


def get_top_k_queries(partial_utility_matrix, utility_matrix_complete, user=0, top_k=10):
    if partial_utility_matrix.shape != utility_matrix_complete.shape:
        print('The shape of the two matrices are different! Something went wrong!')
        return

    # create a list of tuples (query, rating)
    query_rating_list = []
    for query in range(partial_utility_matrix.shape[0]):
        # if the query is not rated by the user
        if np.isnan(partial_utility_matrix.iloc[query, user]):
            query_rating_list.append((query, utility_matrix_complete.iloc[query, user]))

    # sort the list of tuples by rating
    query_rating_list.sort(key=lambda x: x[1], reverse=True)

    # return the top N queries
    return query_rating_list[:top_k]


def save_top_k_queries(partial_utility_matrix, utility_matrix_complete, top_k=10, n_users=10):
    path = '../data/top_' + str(top_k) + '_queries_n_' + str(n_users) + '_users.txt'

    with open(path, 'w') as f:
        for user in range(n_users):
            top_k_queries = get_top_k_queries(partial_utility_matrix, utility_matrix_complete, user, top_k)

            row_string = 'Top ' + str(top_k) + ' queries for user [' + str(user) + ']: ' + str(top_k_queries) + ' \n'
            f.write(row_string)

if __name__ == '__main__':

    # load the data
    start_time_load = time.time()
    partial_utility_matrix = import_data('partial')
    end_time_load = time.time()
    print('Time taken to load the partial and complete (real) utility matrix:', end_time_load - start_time_load)

    # preprocess the data
    start_time_pp = time.time()
    utility_matrix_before_pp, partial_utility_matrix, centered_matrix = preprocess_data(partial_utility_matrix, n_queries=100, n_users=200)
    end_time_pp = time.time()
    print('Time taken to preprocess the partial utility matrix:', end_time_pp - start_time_pp)

    # complete the utility matrix with the ratings
    start_time_cf = time.time()
    utility_matrix_complete, ratings_list = collaborative_filtering(partial_utility_matrix, centered_matrix, top_n=2)
    end_time_cf = time.time()
    print('Time taken to fill the matrix with CF:', end_time_cf - start_time_cf)

    difference_utility_matrix = compute_difference(utility_matrix_complete, n_queries=85, n_users=85)

    # print the data to html, for testing purposes
    print_data_to_html(utility_matrix_before_pp, partial_utility_matrix, centered_matrix, utility_matrix_complete, difference_utility_matrix)


    # plot the heatmap of the difference utility matrix
    plot_heatmap(difference_utility_matrix, 50, 50, 'Difference')

    '''
    # calculate the performances
    mae = calculate_mae(utility_matrix_complete, utility_matrix_complete)
    rmse = calculate_rmse(utility_matrix_complete, utility_matrix_complete)
    mape = calculate_mape(utility_matrix_complete, utility_matrix_complete)
    mre = calculate_mre(utility_matrix_complete, utility_matrix_complete)

    print('MAE: ', mae)
    print('RMSE: ', rmse)
    print('MAPE: ', mape)
    print('MRE: ', mre)
    '''

    # save the top k queries
    save_top_k_queries(partial_utility_matrix, utility_matrix_complete, top_k=10, n_users=10)
